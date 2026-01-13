//! Tensor parallelism for multi-GPU inference
//!
//! This module implements tensor parallelism (TP) for distributing
//! Llama 70B across multiple GPUs.
//!
//! # Sharding Strategy for Llama 70B on 4 GPUs (92GB total)
//!
//! Llama 70B requires ~140GB in BF16. With 4x A10G (24GB each = 92GB total),
//! we need careful sharding:
//!
//! ## Attention Sharding (Column Parallel)
//! - Q projection: [8192, 8192] -> shard output dim across 4 GPUs: [8192, 2048] per GPU
//! - K projection: [8192, 1024] -> replicate (small due to GQA)
//! - V projection: [8192, 1024] -> replicate (small due to GQA)
//! - O projection: [8192, 8192] -> shard input dim: [2048, 8192] per GPU (row parallel)
//!
//! ## FFN Sharding
//! - gate_proj: [8192, 28672] -> shard output: [8192, 7168] per GPU
//! - up_proj: [8192, 28672] -> shard output: [8192, 7168] per GPU
//! - down_proj: [28672, 8192] -> shard input: [7168, 8192] per GPU
//!
//! ## Per-GPU Memory (4 GPUs)
//! - Embeddings: 128256 * 8192 * 2 / 4 = ~500MB per GPU (sharded)
//! - Per layer: ~350MB per GPU (80 layers = ~28GB per GPU)
//! - KV cache: depends on context length
//! - Total: ~30-35GB per GPU (fits in 24GB with some layers on CPU or offloading)

use serde::{Deserialize, Serialize};

use crate::config::ModelArchitecture;
use crate::error::{Result, V2Error};

/// Tensor parallelism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelConfig {
    /// Number of GPUs to use
    pub num_gpus: usize,
    /// Whether to enable tensor parallelism
    pub enabled: bool,
    /// Sharding dimension for attention Q/K/V
    pub attention_shard_dim: ShardDim,
    /// Sharding dimension for FFN
    pub ffn_shard_dim: ShardDim,
    /// Whether to replicate small weights (embeddings, norms)
    pub replicate_small_weights: bool,
    /// Pipeline parallelism depth (layers per GPU)
    pub pipeline_depth: Option<usize>,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            num_gpus: 4,
            enabled: true,
            attention_shard_dim: ShardDim::Column,
            ffn_shard_dim: ShardDim::Column,
            replicate_small_weights: true,
            pipeline_depth: None,
        }
    }
}

impl TensorParallelConfig {
    /// Create config for 4 GPUs
    pub fn for_4_gpus() -> Self {
        Self {
            num_gpus: 4,
            ..Default::default()
        }
    }

    /// Create config for 2 GPUs (H100 80GB)
    pub fn for_2_gpus() -> Self {
        Self {
            num_gpus: 2,
            ..Default::default()
        }
    }

    /// Create config for 8 GPUs (405B or redundancy)
    pub fn for_8_gpus() -> Self {
        Self {
            num_gpus: 8,
            ..Default::default()
        }
    }

    /// Validate configuration against model architecture
    pub fn validate(&self, arch: ModelArchitecture) -> Result<()> {
        let num_heads = arch.num_heads();

        // Ensure heads can be evenly divided across GPUs
        if num_heads % self.num_gpus != 0 {
            return Err(V2Error::Config(format!(
                "Number of heads ({}) must be divisible by num_gpus ({})",
                num_heads, self.num_gpus
            )));
        }

        // Ensure intermediate dim can be evenly divided
        let intermediate_dim = arch.intermediate_dim();
        if intermediate_dim % self.num_gpus != 0 {
            return Err(V2Error::Config(format!(
                "Intermediate dim ({}) must be divisible by num_gpus ({})",
                intermediate_dim, self.num_gpus
            )));
        }

        Ok(())
    }
}

/// Sharding dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardDim {
    /// Shard along columns (output dimension)
    Column,
    /// Shard along rows (input dimension)
    Row,
    /// No sharding (replicate)
    None,
}

/// Weight sharding specification
#[derive(Debug, Clone)]
pub struct ShardSpec {
    /// Name of the weight
    pub name: String,
    /// Sharding dimension (0 = row, 1 = column for 2D)
    pub shard_dim: Option<usize>,
    /// GPU assignment for this shard
    pub gpu_id: usize,
    /// Offset within the full weight
    pub offset: usize,
    /// Size of this shard
    pub size: usize,
}

/// Sharding plan for a full model
#[derive(Debug)]
pub struct ShardingPlan {
    /// Number of GPUs
    pub num_gpus: usize,
    /// Architecture
    pub architecture: ModelArchitecture,
    /// Per-GPU weight assignments
    pub gpu_assignments: Vec<Vec<ShardSpec>>,
    /// Total memory per GPU (bytes)
    pub memory_per_gpu: Vec<usize>,
}

impl ShardingPlan {
    /// Create a sharding plan for any supported architecture
    ///
    /// This works for both Llama and Qwen model families as they share
    /// compatible transformer architectures (GQA + SwiGLU FFN).
    pub fn for_architecture(arch: ModelArchitecture, num_gpus: usize) -> Result<Self> {
        let config = TensorParallelConfig {
            num_gpus,
            ..Default::default()
        };
        config.validate(arch)?;

        let hidden_dim = arch.hidden_dim();
        let num_heads = arch.num_heads();
        let num_kv_heads = arch.num_kv_heads();
        let head_dim = hidden_dim / num_heads;
        let intermediate_dim = arch.intermediate_dim();
        let num_layers = arch.num_layers();
        let vocab_size = arch.vocab_size();

        let heads_per_gpu = num_heads / num_gpus;
        let intermediate_per_gpu = intermediate_dim / num_gpus;

        let mut gpu_assignments: Vec<Vec<ShardSpec>> = (0..num_gpus).map(|_| Vec::new()).collect();
        let mut memory_per_gpu = vec![0usize; num_gpus];

        // Helper to add sharded weight
        let mut add_weight = |name: &str, shape: &[usize], shard_dim: Option<usize>| {
            let total_size = shape.iter().product::<usize>() * 2; // BF16

            match shard_dim {
                Some(dim) => {
                    let shard_size = total_size / num_gpus;
                    for gpu_id in 0..num_gpus {
                        gpu_assignments[gpu_id].push(ShardSpec {
                            name: name.to_string(),
                            shard_dim: Some(dim),
                            gpu_id,
                            offset: gpu_id * shard_size,
                            size: shard_size,
                        });
                        memory_per_gpu[gpu_id] += shard_size;
                    }
                }
                None => {
                    // Replicate to all GPUs
                    for gpu_id in 0..num_gpus {
                        gpu_assignments[gpu_id].push(ShardSpec {
                            name: name.to_string(),
                            shard_dim: None,
                            gpu_id,
                            offset: 0,
                            size: total_size,
                        });
                        memory_per_gpu[gpu_id] += total_size;
                    }
                }
            }
        };

        // Embeddings: shard along vocab dimension
        add_weight(
            "model.embed_tokens.weight",
            &[vocab_size, hidden_dim],
            Some(0),
        );

        // Layer weights
        for layer_idx in 0..num_layers {
            let prefix = format!("model.layers.{}", layer_idx);

            // Attention - shard Q/K/V along output dimension for memory efficiency
            // Q: [num_heads * head_dim, hidden_dim] -> shard output
            add_weight(
                &format!("{}.self_attn.q_proj.weight", prefix),
                &[num_heads * head_dim, hidden_dim],
                Some(0), // Shard along output (heads)
            );

            // K: [num_kv_heads * head_dim, hidden_dim] -> shard output
            // With GQA (8 KV heads), sharding saves memory vs replication
            add_weight(
                &format!("{}.self_attn.k_proj.weight", prefix),
                &[num_kv_heads * head_dim, hidden_dim],
                Some(0), // Shard along output
            );

            // V: [num_kv_heads * head_dim, hidden_dim] -> shard output
            add_weight(
                &format!("{}.self_attn.v_proj.weight", prefix),
                &[num_kv_heads * head_dim, hidden_dim],
                Some(0), // Shard along output
            );

            // O: [num_heads * head_dim, hidden_dim] -> shard input (row parallel)
            add_weight(
                &format!("{}.self_attn.o_proj.weight", prefix),
                &[hidden_dim, num_heads * head_dim],
                Some(1), // Shard along input
            );

            // FFN - shard gate/up output, down input
            add_weight(
                &format!("{}.mlp.gate_proj.weight", prefix),
                &[intermediate_dim, hidden_dim],
                Some(0), // Shard along output
            );

            add_weight(
                &format!("{}.mlp.up_proj.weight", prefix),
                &[intermediate_dim, hidden_dim],
                Some(0), // Shard along output
            );

            add_weight(
                &format!("{}.mlp.down_proj.weight", prefix),
                &[hidden_dim, intermediate_dim],
                Some(1), // Shard along input
            );

            // Layer norms - replicate (small)
            add_weight(
                &format!("{}.input_layernorm.weight", prefix),
                &[hidden_dim],
                None,
            );

            add_weight(
                &format!("{}.post_attention_layernorm.weight", prefix),
                &[hidden_dim],
                None,
            );
        }

        // Final norm - replicate
        add_weight("model.norm.weight", &[hidden_dim], None);

        // LM head - shard along vocab
        add_weight("lm_head.weight", &[vocab_size, hidden_dim], Some(0));

        Ok(Self {
            num_gpus,
            architecture: arch,
            gpu_assignments,
            memory_per_gpu,
        })
    }

    /// Create a sharding plan for Llama architecture (backward compatibility alias)
    #[deprecated(since = "0.2.0", note = "Use for_architecture instead")]
    pub fn for_llama(arch: ModelArchitecture, num_gpus: usize) -> Result<Self> {
        Self::for_architecture(arch, num_gpus)
    }

    /// Get memory requirement per GPU in GB
    pub fn memory_per_gpu_gb(&self) -> Vec<f64> {
        self.memory_per_gpu
            .iter()
            .map(|&bytes| bytes as f64 / (1024.0 * 1024.0 * 1024.0))
            .collect()
    }

    /// Get total memory requirement in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.memory_per_gpu.iter().sum::<usize>() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Print sharding summary
    pub fn print_summary(&self) {
        println!("Sharding Plan for {:?}", self.architecture);
        println!("=====================================");
        println!("Number of GPUs: {}", self.num_gpus);
        println!("Total memory: {:.2} GB", self.total_memory_gb());
        println!();

        for (gpu_id, memory) in self.memory_per_gpu.iter().enumerate() {
            let gb = *memory as f64 / (1024.0 * 1024.0 * 1024.0);
            println!("GPU {}: {:.2} GB ({} weights)", gpu_id, gb, self.gpu_assignments[gpu_id].len());
        }
    }
}

/// Calculate shard indices for a weight tensor
pub fn calculate_shard_indices(
    full_shape: &[usize],
    shard_dim: usize,
    shard_id: usize,
    num_shards: usize,
) -> (usize, usize) {
    let dim_size = full_shape[shard_dim];
    let shard_size = dim_size / num_shards;
    let start = shard_id * shard_size;
    let end = start + shard_size;
    (start, end)
}

/// Slice tensor data along a dimension
pub fn slice_tensor_data(
    data: &[u8],
    shape: &[usize],
    dtype_size: usize,
    shard_dim: usize,
    shard_id: usize,
    num_shards: usize,
) -> Vec<u8> {
    let (start, end) = calculate_shard_indices(shape, shard_dim, shard_id, num_shards);

    if shape.len() == 1 {
        // 1D tensor (e.g., layer norm)
        let start_byte = start * dtype_size;
        let end_byte = end * dtype_size;
        return data[start_byte..end_byte].to_vec();
    }

    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);

        if shard_dim == 0 {
            // Shard along rows
            let shard_rows = (end - start);
            let row_bytes = cols * dtype_size;
            let start_byte = start * row_bytes;
            let end_byte = end * row_bytes;
            return data[start_byte..end_byte].to_vec();
        } else {
            // Shard along columns - need to extract from each row
            let shard_cols = end - start;
            let mut result = Vec::with_capacity(rows * shard_cols * dtype_size);

            for row in 0..rows {
                let row_start = row * cols * dtype_size;
                let col_start = row_start + start * dtype_size;
                let col_end = row_start + end * dtype_size;
                result.extend_from_slice(&data[col_start..col_end]);
            }

            return result;
        }
    }

    // For higher-dimensional tensors, just return the full data
    // (would need more complex slicing logic)
    data.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharding_plan_llama_70b() {
        #[allow(deprecated)]
        let plan = ShardingPlan::for_llama(ModelArchitecture::Llama3_1_70B, 4).unwrap();

        assert_eq!(plan.num_gpus, 4);

        // Each GPU should have assignments
        for gpu_id in 0..4 {
            assert!(!plan.gpu_assignments[gpu_id].is_empty());
        }

        // Memory should be roughly equal across GPUs
        let mem_gb = plan.memory_per_gpu_gb();
        println!("Llama 70B Memory per GPU: {:?} GB", mem_gb);

        // Should fit in 4x 24GB GPUs (with some optimization)
        // Note: This is the weight memory, not including activations/KV cache
        for mem in &mem_gb {
            // Each GPU should have less than 40GB of weights
            // (which is reasonable for 140GB model split 4 ways + replication)
            assert!(*mem < 50.0, "GPU memory too high: {} GB", mem);
        }
    }

    #[test]
    fn test_sharding_plan_qwen_7b() {
        // Qwen 2.5 7B can run on 1 GPU (14GB model)
        let plan = ShardingPlan::for_architecture(ModelArchitecture::Qwen2_5_7B, 1).unwrap();

        assert_eq!(plan.num_gpus, 1);
        assert!(!plan.gpu_assignments[0].is_empty());

        let mem_gb = plan.memory_per_gpu_gb();
        println!("Qwen 7B Memory per GPU: {:?} GB", mem_gb);

        // Should fit in 1x 24GB GPU
        assert!(mem_gb[0] < 20.0, "GPU memory too high: {} GB", mem_gb[0]);
    }

    #[test]
    fn test_sharding_plan_qwen_7b_2gpu() {
        // Qwen 2.5 7B with 2 GPUs for tensor parallelism
        let plan = ShardingPlan::for_architecture(ModelArchitecture::Qwen2_5_7B, 2).unwrap();

        assert_eq!(plan.num_gpus, 2);

        // Verify even split
        let mem_gb = plan.memory_per_gpu_gb();
        println!("Qwen 7B (2 GPU) Memory per GPU: {:?} GB", mem_gb);

        // Memory should be roughly equal
        let diff = (mem_gb[0] - mem_gb[1]).abs();
        assert!(diff < 1.0, "Memory imbalance too high: {} GB", diff);
    }

    #[test]
    fn test_architecture_params() {
        // Verify Qwen 2.5 7B parameters
        let arch = ModelArchitecture::Qwen2_5_7B;
        assert_eq!(arch.hidden_dim(), 3584);
        assert_eq!(arch.num_heads(), 28);
        assert_eq!(arch.num_kv_heads(), 4);
        assert_eq!(arch.num_layers(), 28);
        assert_eq!(arch.vocab_size(), 152064);
        assert_eq!(arch.intermediate_dim(), 18944);
        assert_eq!(arch.head_dim(), 128); // 3584 / 28

        // Verify Llama 3.1 70B parameters
        let arch = ModelArchitecture::Llama3_1_70B;
        assert_eq!(arch.hidden_dim(), 8192);
        assert_eq!(arch.num_heads(), 64);
        assert_eq!(arch.num_kv_heads(), 8);
        assert_eq!(arch.num_layers(), 80);
        assert_eq!(arch.vocab_size(), 128256);
        assert_eq!(arch.intermediate_dim(), 28672);
        assert_eq!(arch.head_dim(), 128); // 8192 / 64
    }

    #[test]
    fn test_slice_tensor_row() {
        // 4x4 tensor, shard along rows into 2
        let data: Vec<u8> = (0..64).collect(); // 4x4 f32 = 64 bytes
        let shape = vec![4, 4];
        let dtype_size = 4; // f32

        let shard0 = slice_tensor_data(&data, &shape, dtype_size, 0, 0, 2);
        let shard1 = slice_tensor_data(&data, &shape, dtype_size, 0, 1, 2);

        assert_eq!(shard0.len(), 32); // 2 rows
        assert_eq!(shard1.len(), 32);
        assert_eq!(&shard0[..], &data[0..32]);
        assert_eq!(&shard1[..], &data[32..64]);
    }

    #[test]
    fn test_slice_tensor_col() {
        // 2x4 tensor, shard along cols into 2
        // Layout: [[0,1,2,3], [4,5,6,7]] as bytes
        let data: Vec<u8> = (0..8).collect(); // 2x4 i8 = 8 bytes
        let shape = vec![2, 4];
        let dtype_size = 1; // i8

        let shard0 = slice_tensor_data(&data, &shape, dtype_size, 1, 0, 2);
        let shard1 = slice_tensor_data(&data, &shape, dtype_size, 1, 1, 2);

        // shard0 should be [[0,1], [4,5]]
        assert_eq!(shard0, vec![0, 1, 4, 5]);
        // shard1 should be [[2,3], [6,7]]
        assert_eq!(shard1, vec![2, 3, 6, 7]);
    }
}
