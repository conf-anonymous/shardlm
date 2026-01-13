//! Secure model weights for privacy-preserving inference
//!
//! This module provides CPU-resident weights wrapped in SecureLinear/SecureAttention/SecureFFN
//! for secure inference. The GPU weights are downloaded to CPU once at initialization,
//! then used for all secure inference requests.
//!
//! # Why CPU weights for secure inference?
//!
//! Secret sharing operates on shares that are element-wise additions. The server must:
//! 1. Receive client_share and server_share separately
//! 2. Compute Y_c = X_c · W and Y_s = X_s · W + b independently
//! 3. Return both output shares to client
//!
//! This requires CPU computation because:
//! - We need to process shares separately (not fused GPU kernels)
//! - The computation is linear (matmul) which is fast on CPU for inference
//! - GPU weights are BF16 and we need F32 for precision
//!
//! For production performance, consider:
//! - GPU-accelerated secure linear (see secure_linear.rs gpu module)
//! - Batched processing across multiple requests

use shardlm_v2_core::gpu::GpuDevice;
use shardlm_v2_model::{ShardedLayerWeights, ShardedModelWeights};
use shardlm_v2_sharing::{SecureAttention, SecureFFN, SecureLinear};

use crate::error::{Result, ServerError};

/// Secure layer weights (CPU-resident for share processing)
pub struct SecureLayerWeights {
    /// Secure attention (Q/K/V/O projections)
    pub attention: SecureAttention,
    /// Secure FFN (gate/up/down projections)
    pub ffn: SecureFFN,
    /// Input layer norm weights (for client-side RMSNorm)
    pub input_layernorm: Vec<f32>,
    /// Post-attention layer norm weights (for client-side RMSNorm)
    pub post_attn_layernorm: Vec<f32>,
}

/// Secure model weights (CPU-resident for share processing)
pub struct SecureModelWeights {
    /// Secure layers
    pub layers: Vec<SecureLayerWeights>,
    /// LM head projection (hidden_dim -> vocab_size)
    pub lm_head: SecureLinear,
    /// Final norm weights (for client-side RMSNorm)
    pub final_norm: Vec<f32>,
    /// Embedding table (vocab_size × hidden_dim) - for OT-based embedding lookup
    pub embeddings: Vec<f32>,
    /// Model dimensions
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    /// RoPE base frequency
    pub rope_theta: f32,
}

impl SecureModelWeights {
    /// Initialize secure weights from GPU model weights
    ///
    /// Downloads GPU weights to CPU and wraps them in SecureLinear structures.
    /// This is done once at model load time.
    #[cfg(feature = "cuda")]
    pub fn from_gpu_weights(
        gpu_weights: &ShardedModelWeights,
        devices: &[&GpuDevice],
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
        rope_theta: f32,
    ) -> Result<Self> {
        let num_layers = gpu_weights.layers.len();
        let num_gpus = devices.len();

        tracing::info!(
            "Initializing secure weights: {} layers, hidden_dim={}, {} GPUs",
            num_layers, hidden_dim, num_gpus
        );

        // Download embedding table from GPU (for OT-based lookup)
        // Embeddings are sharded along vocab dimension, need to concatenate
        tracing::info!("Downloading embedding table for OT...");
        let (embeddings, _embed_vocab_size) = Self::download_sharded_embeddings(
            &gpu_weights.embed_tokens, devices, hidden_dim
        )?;

        // Download and wrap each layer
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            if layer_idx % 10 == 0 || layer_idx == num_layers - 1 {
                tracing::info!("  Processing layer {}/{}", layer_idx + 1, num_layers);
            }

            let layer = Self::create_secure_layer(
                &gpu_weights.layers[layer_idx],
                devices,
                hidden_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_dim,
                rope_theta,
            )?;
            layers.push(layer);
        }

        // Download final norm
        tracing::info!("Downloading final norm...");
        let final_norm = Self::download_replicated_f32(&gpu_weights.norm, devices)?;

        // Download and create LM head SecureLinear
        // LM head is sharded along vocab dimension
        // Note: actual_vocab_size may differ from config vocab_size due to tied embeddings
        tracing::info!("Downloading LM head...");
        let (lm_head_weights, actual_vocab_size) = Self::download_sharded_lm_head(
            &gpu_weights.lm_head, devices, hidden_dim
        )?;
        let lm_head = SecureLinear::new(lm_head_weights, None, hidden_dim, actual_vocab_size);

        tracing::info!("Secure weights initialization complete (vocab_size={})", actual_vocab_size);

        Ok(Self {
            layers,
            lm_head,
            final_norm,
            embeddings,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: actual_vocab_size, // Use actual size from tensor
            num_layers,
            rope_theta,
        })
    }

    /// Download sharded embeddings and concatenate
    /// Returns (embeddings, actual_vocab_size)
    #[cfg(feature = "cuda")]
    fn download_sharded_embeddings(
        shards: &[shardlm_v2_core::gpu::CudaTensorBF16],
        devices: &[&GpuDevice],
        hidden_dim: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let num_gpus = shards.len();

        // Get actual vocab_per_gpu from tensor shape
        // Embedding tensor shape is [vocab_per_gpu, hidden_dim]
        let vocab_per_gpu = shards[0].shape[0];
        let actual_vocab_size = vocab_per_gpu * num_gpus;

        tracing::info!(
            "Embedding table shape: vocab_per_gpu={}, hidden_dim={}, actual_vocab_size={}",
            vocab_per_gpu, hidden_dim, actual_vocab_size
        );

        let mut embeddings = vec![0.0f32; actual_vocab_size * hidden_dim];

        for (gpu_id, shard) in shards.iter().enumerate() {
            // Download BF16 as bytes and convert to f32
            let bf16_bytes = shard.to_host(devices[gpu_id])
                .map_err(|e| ServerError::Internal(format!("Failed to download embeddings: {}", e)))?;

            let f32_data = bf16_bytes_to_f32(&bf16_bytes);

            // Copy to correct position in full embedding table
            let offset = gpu_id * vocab_per_gpu * hidden_dim;
            embeddings[offset..offset + f32_data.len()].copy_from_slice(&f32_data);
        }

        Ok((embeddings, actual_vocab_size))
    }

    /// Download sharded LM head and concatenate (transposed for matmul)
    /// Returns (weights, actual_vocab_size) since tied embeddings may have different size
    #[cfg(feature = "cuda")]
    fn download_sharded_lm_head(
        shards: &[shardlm_v2_core::gpu::CudaTensorBF16],
        devices: &[&GpuDevice],
        hidden_dim: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let num_gpus = shards.len();

        // Get actual vocab_per_gpu from tensor shape
        // LM head tensor shape is [vocab_per_gpu, hidden_dim]
        let vocab_per_gpu = shards[0].shape[0];
        let actual_vocab_size = vocab_per_gpu * num_gpus;

        tracing::info!(
            "LM head shape: vocab_per_gpu={}, hidden_dim={}, actual_vocab_size={}",
            vocab_per_gpu, hidden_dim, actual_vocab_size
        );

        // For SecureLinear we need [hidden_dim, vocab_size] (row-major)
        let mut weights = vec![0.0f32; hidden_dim * actual_vocab_size];

        for (gpu_id, shard) in shards.iter().enumerate() {
            let bf16_bytes = shard.to_host(devices[gpu_id])
                .map_err(|e| ServerError::Internal(format!("Failed to download LM head: {}", e)))?;

            let f32_data = bf16_bytes_to_f32(&bf16_bytes);

            // Shard shape is [vocab_per_gpu, hidden_dim]
            // We need to transpose and place in the right position
            let vocab_offset = gpu_id * vocab_per_gpu;

            for v in 0..vocab_per_gpu {
                for h in 0..hidden_dim {
                    // Source: [v, h] in shard
                    let src_idx = v * hidden_dim + h;
                    // Dest: [h, vocab_offset + v] in transposed output
                    let dst_idx = h * actual_vocab_size + (vocab_offset + v);
                    weights[dst_idx] = f32_data[src_idx];
                }
            }
        }

        Ok((weights, actual_vocab_size))
    }

    /// Download replicated F32 tensor (like layer norms)
    #[cfg(feature = "cuda")]
    fn download_replicated_f32(
        shards: &[shardlm_v2_core::gpu::CudaTensor],
        devices: &[&GpuDevice],
    ) -> Result<Vec<f32>> {
        // Replicated - just download from first GPU
        shards[0].to_f32_host(devices[0])
            .map_err(|e| ServerError::Internal(format!("Failed to download tensor: {}", e)))
    }

    /// Create a secure layer from GPU weights
    #[cfg(feature = "cuda")]
    fn create_secure_layer(
        layer: &ShardedLayerWeights,
        devices: &[&GpuDevice],
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        rope_theta: f32,
    ) -> Result<SecureLayerWeights> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        // Download and concatenate sharded projections
        // Q/K/V are sharded on output dimension (dim 0)
        // O is sharded on input dimension (dim 1)

        let q_proj_weights = Self::download_column_sharded_bf16(
            &layer.q_proj, devices, hidden_dim, q_dim
        )?;
        let k_proj_weights = Self::download_column_sharded_bf16(
            &layer.k_proj, devices, hidden_dim, kv_dim
        )?;
        let v_proj_weights = Self::download_column_sharded_bf16(
            &layer.v_proj, devices, hidden_dim, kv_dim
        )?;
        let o_proj_weights = Self::download_row_sharded_bf16(
            &layer.o_proj, devices, q_dim, hidden_dim
        )?;

        // Download biases if present (Qwen has them, Llama doesn't)
        let q_bias = layer.q_proj_bias.as_ref()
            .map(|b| Self::download_column_sharded_f32(b, devices, q_dim))
            .transpose()?;
        let k_bias = layer.k_proj_bias.as_ref()
            .map(|b| Self::download_column_sharded_f32(b, devices, kv_dim))
            .transpose()?;
        let v_bias = layer.v_proj_bias.as_ref()
            .map(|b| Self::download_column_sharded_f32(b, devices, kv_dim))
            .transpose()?;

        // Create SecureLinear for each projection
        let q_proj = SecureLinear::new(q_proj_weights, q_bias, hidden_dim, q_dim);
        let k_proj = SecureLinear::new(k_proj_weights, k_bias, hidden_dim, kv_dim);
        let v_proj = SecureLinear::new(v_proj_weights, v_bias, hidden_dim, kv_dim);
        let o_proj = SecureLinear::new(o_proj_weights, None, q_dim, hidden_dim);

        // Max sequence length for RoPE precomputation (8K tokens should be enough)
        const MAX_SEQ_LEN: usize = 8192;

        let attention = SecureAttention::new(
            q_proj, k_proj, v_proj, o_proj,
            num_heads, num_kv_heads, head_dim,
            rope_theta, MAX_SEQ_LEN,
        );

        // FFN projections
        // gate/up are sharded on output dimension (dim 0)
        // down is sharded on input dimension (dim 1)
        let gate_proj_weights = Self::download_column_sharded_bf16(
            &layer.gate_proj, devices, hidden_dim, intermediate_dim
        )?;
        let up_proj_weights = Self::download_column_sharded_bf16(
            &layer.up_proj, devices, hidden_dim, intermediate_dim
        )?;
        let down_proj_weights = Self::download_row_sharded_bf16(
            &layer.down_proj, devices, intermediate_dim, hidden_dim
        )?;

        let gate_proj = SecureLinear::new(gate_proj_weights, None, hidden_dim, intermediate_dim);
        let up_proj = SecureLinear::new(up_proj_weights, None, hidden_dim, intermediate_dim);
        let down_proj = SecureLinear::new(down_proj_weights, None, intermediate_dim, hidden_dim);

        let ffn = SecureFFN::new(gate_proj, up_proj, down_proj, hidden_dim, intermediate_dim);

        // Layer norms (replicated)
        let input_layernorm = Self::download_replicated_f32(&layer.input_layernorm, devices)?;
        let post_attn_layernorm = Self::download_replicated_f32(&layer.post_attn_layernorm, devices)?;

        Ok(SecureLayerWeights {
            attention,
            ffn,
            input_layernorm,
            post_attn_layernorm,
        })
    }

    /// Download column-sharded BF16 tensor (sharded on output dim)
    #[cfg(feature = "cuda")]
    fn download_column_sharded_bf16(
        shards: &[shardlm_v2_core::gpu::CudaTensorBF16],
        devices: &[&GpuDevice],
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        let num_gpus = shards.len();
        let out_per_gpu = out_features / num_gpus;

        // Shape is [out_per_gpu, in_features] per shard
        // We need [in_features, out_features] row-major for SecureLinear
        let mut weights = vec![0.0f32; in_features * out_features];

        for (gpu_id, shard) in shards.iter().enumerate() {
            let bf16_bytes = shard.to_host(devices[gpu_id])
                .map_err(|e| ServerError::Internal(format!("Failed to download tensor: {}", e)))?;

            let f32_data = bf16_bytes_to_f32(&bf16_bytes);
            let out_offset = gpu_id * out_per_gpu;

            // Source: [o, i] for o in 0..out_per_gpu, i in 0..in_features
            // Dest: [i, out_offset + o]
            for o in 0..out_per_gpu {
                for i in 0..in_features {
                    let src_idx = o * in_features + i;
                    let dst_idx = i * out_features + (out_offset + o);
                    weights[dst_idx] = f32_data[src_idx];
                }
            }
        }

        Ok(weights)
    }

    /// Download row-sharded BF16 tensor (sharded on input dim)
    #[cfg(feature = "cuda")]
    fn download_row_sharded_bf16(
        shards: &[shardlm_v2_core::gpu::CudaTensorBF16],
        devices: &[&GpuDevice],
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        let num_gpus = shards.len();
        let in_per_gpu = in_features / num_gpus;

        // Shape is [out_features, in_per_gpu] per shard
        // We need [in_features, out_features] row-major for SecureLinear
        let mut weights = vec![0.0f32; in_features * out_features];

        for (gpu_id, shard) in shards.iter().enumerate() {
            let bf16_bytes = shard.to_host(devices[gpu_id])
                .map_err(|e| ServerError::Internal(format!("Failed to download tensor: {}", e)))?;

            let f32_data = bf16_bytes_to_f32(&bf16_bytes);
            let in_offset = gpu_id * in_per_gpu;

            // Source: [o, i] for o in 0..out_features, i in 0..in_per_gpu
            // Dest: [in_offset + i, o]
            for o in 0..out_features {
                for i in 0..in_per_gpu {
                    let src_idx = o * in_per_gpu + i;
                    let dst_idx = (in_offset + i) * out_features + o;
                    weights[dst_idx] = f32_data[src_idx];
                }
            }
        }

        Ok(weights)
    }

    /// Download column-sharded F32 tensor (for biases)
    #[cfg(feature = "cuda")]
    fn download_column_sharded_f32(
        shards: &[shardlm_v2_core::gpu::CudaTensor],
        devices: &[&GpuDevice],
        out_features: usize,
    ) -> Result<Vec<f32>> {
        let num_gpus = shards.len();
        let out_per_gpu = out_features / num_gpus;

        let mut bias = vec![0.0f32; out_features];

        for (gpu_id, shard) in shards.iter().enumerate() {
            let f32_data = shard.to_f32_host(devices[gpu_id])
                .map_err(|e| ServerError::Internal(format!("Failed to download bias: {}", e)))?;

            let offset = gpu_id * out_per_gpu;
            bias[offset..offset + out_per_gpu].copy_from_slice(&f32_data);
        }

        Ok(bias)
    }

    /// Get layer weights
    pub fn layer(&self, idx: usize) -> &SecureLayerWeights {
        &self.layers[idx]
    }
}

/// Convert BF16 bytes to f32 vec
fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    use half::bf16;

    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect()
}

// =============================================================================
// GPU-ACCELERATED SECURE WEIGHTS
// =============================================================================

/// GPU-accelerated secure layer weights
#[cfg(feature = "cuda")]
pub struct GpuSecureLayerWeights {
    /// GPU-accelerated attention
    pub attention: shardlm_v2_sharing::GpuSecureAttention,
    /// GPU-accelerated FFN
    pub ffn: shardlm_v2_sharing::GpuSecureFFN,
    /// Input layer norm weights (for client-side RMSNorm, server applies gamma)
    pub input_layernorm: Vec<f32>,
    /// Post-attention layer norm weights
    pub post_attn_layernorm: Vec<f32>,
    /// Input layer norm weights on GPU (for fully-GPU inference path)
    pub input_layernorm_gpu: shardlm_v2_core::gpu::CudaTensor,
    /// Post-attention layer norm weights on GPU
    pub post_attn_layernorm_gpu: shardlm_v2_core::gpu::CudaTensor,
}

/// GPU-accelerated secure model weights
///
/// Uses cuBLAS for all matrix multiplications while maintaining security guarantees.
/// Weights are stored on GPU; shares are uploaded, processed, and downloaded per request.
///
/// When using multiple GPUs, layers are distributed across GPUs to avoid OOM:
/// - Layer i is on GPU (i % num_gpus)
/// - The `layer_gpu_ids` field tracks which GPU each layer is on
#[cfg(feature = "cuda")]
pub struct GpuSecureModelWeights {
    /// GPU-accelerated layers
    pub layers: Vec<GpuSecureLayerWeights>,
    /// Which GPU each layer is on (layer_gpu_ids[i] = GPU index for layer i)
    pub layer_gpu_ids: Vec<usize>,
    /// Number of GPUs used
    pub num_gpus: usize,
    /// GPU-accelerated LM head
    pub lm_head: shardlm_v2_sharing::GpuSecureLinear,
    /// Final norm weights (CPU, for gamma scaling)
    pub final_norm: Vec<f32>,
    /// Final norm weights on GPU (for fully-GPU inference path)
    pub final_norm_gpu: shardlm_v2_core::gpu::CudaTensor,
    /// Embedding table (CPU, for OT-based lookup)
    pub embeddings: Vec<f32>,
    /// Model dimensions
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    /// RoPE base frequency
    pub rope_theta: f32,
}

#[cfg(feature = "cuda")]
impl GpuSecureModelWeights {
    /// Initialize GPU-accelerated secure weights from GPU model weights
    ///
    /// Downloads GPU weights to CPU (for creating GpuSecureLinear), then
    /// uploads them back to GPU in the format needed for secure operations.
    pub fn from_gpu_weights(
        gpu_weights: &ShardedModelWeights,
        devices: &[&GpuDevice],
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        vocab_size: usize,
        rope_theta: f32,
    ) -> Result<Self> {
        let num_layers = gpu_weights.layers.len();
        let num_gpus = devices.len();

        // Distribute layers across GPUs to avoid OOM
        // Layer i goes to GPU (i % num_gpus)
        let layer_gpu_ids: Vec<usize> = (0..num_layers).map(|i| i % num_gpus).collect();

        // Use first GPU for embeddings and final norm (they're small)
        let device = devices[0];

        tracing::info!(
            "Initializing GPU-accelerated secure weights: {} layers, hidden_dim={}, distributed across {} GPUs",
            num_layers, hidden_dim, num_gpus
        );

        // Download embedding table
        tracing::info!("Downloading embedding table for GPU secure inference...");
        let (embeddings, _embed_vocab_size) = SecureModelWeights::download_sharded_embeddings(
            &gpu_weights.embed_tokens, devices, hidden_dim
        )?;

        // Process each layer, distributing across GPUs
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let gpu_id = layer_gpu_ids[layer_idx];
            let layer_device = devices[gpu_id];

            if layer_idx % 10 == 0 || layer_idx == num_layers - 1 {
                tracing::info!("  Processing layer {}/{} for GPU {} (distributed)", layer_idx + 1, num_layers, gpu_id);
            }

            let layer = Self::create_gpu_secure_layer(
                &gpu_weights.layers[layer_idx],
                devices,
                hidden_dim,
                num_heads,
                num_kv_heads,
                head_dim,
                intermediate_dim,
                rope_theta,
                layer_device,
            )?;
            layers.push(layer);
        }

        // Download final norm
        let final_norm = SecureModelWeights::download_replicated_f32(&gpu_weights.norm, devices)?;

        // Upload final norm to GPU for fully-GPU inference path
        let final_norm_gpu = shardlm_v2_core::gpu::CudaTensor::from_f32(
            device,
            vec![final_norm.len()],
            final_norm.clone(),
        ).map_err(|e| ServerError::Internal(format!("Failed to upload final norm to GPU: {}", e)))?;

        // Create GPU LM head
        tracing::info!("Creating GPU LM head...");
        let (lm_head_weights, actual_vocab_size) = SecureModelWeights::download_sharded_lm_head(
            &gpu_weights.lm_head, devices, hidden_dim
        )?;
        let lm_head = shardlm_v2_sharing::GpuSecureLinear::from_cpu(
            &lm_head_weights, None, hidden_dim, actual_vocab_size, device
        ).map_err(|e| ServerError::Internal(format!("Failed to create GPU LM head: {}", e)))?;

        tracing::info!("GPU-accelerated secure weights initialization complete (distributed across {} GPUs)", num_gpus);

        Ok(Self {
            layers,
            layer_gpu_ids,
            num_gpus,
            lm_head,
            final_norm,
            final_norm_gpu,
            embeddings,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            vocab_size: actual_vocab_size,
            num_layers,
            rope_theta,
        })
    }

    /// Get the GPU ID for a given layer
    pub fn layer_gpu_id(&self, layer_idx: usize) -> usize {
        self.layer_gpu_ids[layer_idx]
    }

    /// Create a GPU-accelerated secure layer
    fn create_gpu_secure_layer(
        layer: &ShardedLayerWeights,
        devices: &[&GpuDevice],
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        rope_theta: f32,
        device: &GpuDevice,
    ) -> Result<GpuSecureLayerWeights> {
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        const MAX_SEQ_LEN: usize = 8192;

        // Download attention projection weights
        let q_weights = SecureModelWeights::download_column_sharded_bf16(
            &layer.q_proj, devices, hidden_dim, q_dim
        )?;
        let k_weights = SecureModelWeights::download_column_sharded_bf16(
            &layer.k_proj, devices, hidden_dim, kv_dim
        )?;
        let v_weights = SecureModelWeights::download_column_sharded_bf16(
            &layer.v_proj, devices, hidden_dim, kv_dim
        )?;
        let o_weights = SecureModelWeights::download_row_sharded_bf16(
            &layer.o_proj, devices, q_dim, hidden_dim
        )?;

        // Download biases if present
        let q_bias = layer.q_proj_bias.as_ref()
            .map(|b| SecureModelWeights::download_column_sharded_f32(b, devices, q_dim))
            .transpose()?;
        let k_bias = layer.k_proj_bias.as_ref()
            .map(|b| SecureModelWeights::download_column_sharded_f32(b, devices, kv_dim))
            .transpose()?;
        let v_bias = layer.v_proj_bias.as_ref()
            .map(|b| SecureModelWeights::download_column_sharded_f32(b, devices, kv_dim))
            .transpose()?;

        // Create GPU attention
        let attention = shardlm_v2_sharing::GpuSecureAttention::from_cpu(
            &q_weights, &k_weights, &v_weights, &o_weights,
            q_bias.as_deref(), k_bias.as_deref(), v_bias.as_deref(),
            num_heads, num_kv_heads, head_dim,
            rope_theta, MAX_SEQ_LEN,
            device,
        ).map_err(|e| ServerError::Internal(format!("Failed to create GPU attention: {}", e)))?;

        // Download FFN weights
        let gate_weights = SecureModelWeights::download_column_sharded_bf16(
            &layer.gate_proj, devices, hidden_dim, intermediate_dim
        )?;
        let up_weights = SecureModelWeights::download_column_sharded_bf16(
            &layer.up_proj, devices, hidden_dim, intermediate_dim
        )?;
        let down_weights = SecureModelWeights::download_row_sharded_bf16(
            &layer.down_proj, devices, intermediate_dim, hidden_dim
        )?;

        // Create GPU FFN
        let ffn = shardlm_v2_sharing::GpuSecureFFN::from_cpu(
            &gate_weights, &up_weights, &down_weights,
            hidden_dim, intermediate_dim,
            device,
        ).map_err(|e| ServerError::Internal(format!("Failed to create GPU FFN: {}", e)))?;

        // Download layer norms (CPU, for gamma scaling)
        let input_layernorm = SecureModelWeights::download_replicated_f32(&layer.input_layernorm, devices)?;
        let post_attn_layernorm = SecureModelWeights::download_replicated_f32(&layer.post_attn_layernorm, devices)?;

        // Upload layer norms to GPU for fully-GPU inference path
        let input_layernorm_gpu = shardlm_v2_core::gpu::CudaTensor::from_f32(
            device,
            vec![input_layernorm.len()],
            input_layernorm.clone(),
        ).map_err(|e| ServerError::Internal(format!("Failed to upload input layernorm to GPU: {}", e)))?;

        let post_attn_layernorm_gpu = shardlm_v2_core::gpu::CudaTensor::from_f32(
            device,
            vec![post_attn_layernorm.len()],
            post_attn_layernorm.clone(),
        ).map_err(|e| ServerError::Internal(format!("Failed to upload post_attn layernorm to GPU: {}", e)))?;

        Ok(GpuSecureLayerWeights {
            attention,
            ffn,
            input_layernorm,
            post_attn_layernorm,
            input_layernorm_gpu,
            post_attn_layernorm_gpu,
        })
    }

    /// Get layer weights
    pub fn layer(&self, idx: usize) -> &GpuSecureLayerWeights {
        &self.layers[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        // Test conversion of known BF16 values
        let one_bf16 = half::bf16::from_f32(1.0);
        let bytes = one_bf16.to_bits().to_le_bytes();

        let f32_vec = bf16_bytes_to_f32(&bytes);
        assert!((f32_vec[0] - 1.0).abs() < 1e-3);
    }
}
