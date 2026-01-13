//! Optimized model loading with INT8 quantization and pipeline parallelism
//!
//! This loader extends ShardedModelLoader with memory optimization strategies:
//! - INT8 quantization for 50% weight memory reduction
//! - Pipeline parallelism for layer distribution
//! - CPU offloading for edge/middle layers

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_core::gpu::{GpuBuffer, GpuDevice, MultiGpuContext};
use shardlm_v2_core::memory::{MemoryConfig, MemoryEstimator, OffloadConfig, PipelineConfig};
use shardlm_v2_core::model::LlamaConfig;
use shardlm_v2_core::parallel::{slice_tensor_data, TensorParallelConfig};
use shardlm_v2_core::quantization::{quantize_bf16_to_int8_per_channel, QuantizationMode, QuantizedTensor};
use shardlm_v2_core::tensor::DType;

use crate::error::{ModelError, Result};

/// Quantized weight tensor (INT8 with scales)
pub struct QuantizedWeight {
    /// INT8 quantized data
    pub data: Vec<i8>,
    /// Scale factors (per output channel)
    pub scales: Vec<f32>,
    /// Original shape
    pub shape: Vec<usize>,
    /// GPU buffer for INT8 data (if loaded to GPU)
    pub gpu_buffer: Option<GpuBuffer>,
    /// GPU buffer for scales
    pub scale_buffer: Option<GpuBuffer>,
    /// Device ID (-1 for CPU)
    pub device_id: i32,
}

impl QuantizedWeight {
    /// Create from a QuantizedTensor
    fn from_quantized(qt: QuantizedTensor) -> Self {
        Self {
            data: qt.data,
            scales: qt.scales,
            shape: qt.shape,
            gpu_buffer: None,
            scale_buffer: None,
            device_id: -1, // CPU
        }
    }

    /// Get memory size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Check if on GPU
    pub fn is_on_gpu(&self) -> bool {
        self.device_id >= 0
    }
}

/// Optimized layer weights with quantization support
pub struct OptimizedLayerWeights {
    /// Layer index
    pub layer_idx: usize,
    /// Q projection (quantized)
    pub q_proj: Vec<QuantizedWeight>,
    /// K projection (quantized)
    pub k_proj: Vec<QuantizedWeight>,
    /// V projection (quantized)
    pub v_proj: Vec<QuantizedWeight>,
    /// O projection (quantized)
    pub o_proj: Vec<QuantizedWeight>,
    /// Gate projection (quantized)
    pub gate_proj: Vec<QuantizedWeight>,
    /// Up projection (quantized)
    pub up_proj: Vec<QuantizedWeight>,
    /// Down projection (quantized)
    pub down_proj: Vec<QuantizedWeight>,
    /// Input layer norm (BF16, small)
    pub input_layernorm: Vec<Vec<u8>>,
    /// Post-attention layer norm (BF16, small)
    pub post_attn_layernorm: Vec<Vec<u8>>,
    /// Whether this layer is offloaded to CPU
    pub is_offloaded: bool,
    /// Primary GPU for this layer (pipeline parallelism)
    pub primary_gpu: usize,
}

/// Optimized model weights
pub struct OptimizedModelWeights {
    /// Token embeddings (quantized, sharded)
    pub embed_tokens: Vec<QuantizedWeight>,
    /// Layer weights
    pub layers: Vec<OptimizedLayerWeights>,
    /// Final norm (BF16, replicated)
    pub norm: Vec<Vec<u8>>,
    /// LM head (quantized, sharded)
    pub lm_head: Vec<QuantizedWeight>,
    /// Memory configuration used
    pub memory_config: MemoryConfig,
    /// Pipeline configuration
    pub pipeline_config: Option<PipelineConfig>,
}

impl OptimizedModelWeights {
    /// Get total memory used (approximate)
    pub fn total_memory_bytes(&self) -> usize {
        let mut total = 0;

        // Embeddings
        for w in &self.embed_tokens {
            total += w.size_bytes();
        }

        // Layers
        for layer in &self.layers {
            for w in &layer.q_proj { total += w.size_bytes(); }
            for w in &layer.k_proj { total += w.size_bytes(); }
            for w in &layer.v_proj { total += w.size_bytes(); }
            for w in &layer.o_proj { total += w.size_bytes(); }
            for w in &layer.gate_proj { total += w.size_bytes(); }
            for w in &layer.up_proj { total += w.size_bytes(); }
            for w in &layer.down_proj { total += w.size_bytes(); }
            for b in &layer.input_layernorm { total += b.len(); }
            for b in &layer.post_attn_layernorm { total += b.len(); }
        }

        // Final norm and LM head
        for b in &self.norm { total += b.len(); }
        for w in &self.lm_head { total += w.size_bytes(); }

        total
    }

    /// Get number of offloaded layers
    pub fn num_offloaded_layers(&self) -> usize {
        self.layers.iter().filter(|l| l.is_offloaded).count()
    }
}

/// Optimized model loader with memory optimization strategies
pub struct OptimizedModelLoader {
    /// Model configuration
    config: LlamaConfig,
    /// Architecture
    arch: ModelArchitecture,
    /// Path to model directory
    model_dir: std::path::PathBuf,
    /// Memory-mapped files
    mmaps: HashMap<String, Mmap>,
    /// Memory configuration
    memory_config: MemoryConfig,
    /// Tensor parallel configuration
    tp_config: TensorParallelConfig,
    /// Pipeline configuration (if enabled)
    pipeline_config: Option<PipelineConfig>,
    /// Offload configuration (if enabled)
    offload_config: Option<OffloadConfig>,
    /// GPU context
    gpu_ctx: MultiGpuContext,
}

impl OptimizedModelLoader {
    /// Create a new optimized loader with memory configuration
    pub fn new(
        model_dir: impl AsRef<Path>,
        arch: ModelArchitecture,
        memory_config: MemoryConfig,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        if !model_dir.exists() {
            return Err(ModelError::ModelNotFound(model_dir.display().to_string()));
        }

        let config = LlamaConfig::from_architecture(arch);
        let num_gpus = memory_config.pipeline_stages.max(1);

        let tp_config = TensorParallelConfig {
            num_gpus,
            enabled: true,
            ..Default::default()
        };

        tp_config
            .validate(arch)
            .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;

        // Create pipeline config if enabled
        let pipeline_config = if memory_config.pipeline_parallel {
            Some(PipelineConfig::balanced(arch.num_layers(), num_gpus))
        } else {
            None
        };

        // Create offload config if enabled
        let offload_config = if memory_config.cpu_offload {
            Some(OffloadConfig::specific(
                memory_config.cpu_offload_layers.clone(),
                (0..arch.num_layers())
                    .filter(|l| !memory_config.cpu_offload_layers.contains(l))
                    .collect(),
            ))
        } else {
            None
        };

        // Initialize GPU context
        let gpu_ctx = MultiGpuContext::new(num_gpus)
            .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;

        Ok(Self {
            config,
            arch,
            model_dir,
            mmaps: HashMap::new(),
            memory_config,
            tp_config,
            pipeline_config,
            offload_config,
            gpu_ctx,
        })
    }

    /// Create with default 4x A10G configuration
    pub fn for_4x_a10g(model_dir: impl AsRef<Path>, arch: ModelArchitecture) -> Result<Self> {
        Self::new(model_dir, arch, MemoryConfig::for_4x_a10g())
    }

    /// Get memory estimator
    pub fn memory_estimator(&self) -> MemoryEstimator {
        MemoryEstimator::new(self.arch, self.memory_config.clone())
    }

    /// Print memory summary
    pub fn print_memory_summary(&self) {
        let estimator = self.memory_estimator();
        let summary = estimator.summary(self.tp_config.num_gpus);
        println!("{}", summary);
    }

    /// Load all safetensor files
    pub fn load_all_files(&mut self) -> Result<()> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(&self.model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "safetensors") {
                files.push(path);
            }
        }
        files.sort();

        println!("Found {} safetensor files", files.len());
        for file in &files {
            println!("  Loading: {}", file.display());
            let f = std::fs::File::open(file)?;
            let mmap = unsafe { Mmap::map(&f)? };
            let filename = file.file_name().unwrap().to_string_lossy().to_string();
            self.mmaps.insert(filename, mmap);
        }

        Ok(())
    }

    /// Get raw weight data by name
    fn get_weight_data(&self, name: &str) -> Result<(Vec<usize>, DType, Vec<u8>)> {
        for (filename, mmap) in &self.mmaps {
            let safetensors = SafeTensors::deserialize(mmap)
                .map_err(|e| ModelError::InvalidFormat(format!("{}: {}", filename, e)))?;

            if let Ok(tensor) = safetensors.tensor(name) {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let dtype = match tensor.dtype() {
                    safetensors::Dtype::BF16 => DType::BF16,
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::F32 => DType::F32,
                    dt => return Err(ModelError::UnsupportedDtype(format!("{:?}", dt))),
                };
                let data = tensor.data().to_vec();
                return Ok((shape, dtype, data));
            }
        }
        Err(ModelError::MissingWeight(name.to_string()))
    }

    /// Load and quantize a weight, shard across GPUs
    fn load_quantized_sharded(
        &self,
        name: &str,
        shard_dim: Option<usize>,
    ) -> Result<Vec<QuantizedWeight>> {
        let (shape, dtype, data) = self.get_weight_data(name)?;
        let num_gpus = self.tp_config.num_gpus;

        // Only BF16 supported for quantization currently
        if dtype != DType::BF16 {
            return Err(ModelError::UnsupportedDtype(format!(
                "Quantization requires BF16, got {:?}",
                dtype
            )));
        }

        let mut shards = Vec::with_capacity(num_gpus);

        match shard_dim {
            Some(dim) => {
                for gpu_id in 0..num_gpus {
                    let shard_data = slice_tensor_data(
                        &data,
                        &shape,
                        dtype.size_bytes(),
                        dim,
                        gpu_id,
                        num_gpus,
                    );

                    let mut shard_shape = shape.clone();
                    shard_shape[dim] /= num_gpus;

                    // Quantize to INT8
                    if matches!(self.memory_config.quantization, QuantizationMode::Int8PerChannel | QuantizationMode::Int8PerTensor) {
                        let quantized = quantize_bf16_to_int8_per_channel(&shard_data, &shard_shape);
                        shards.push(QuantizedWeight::from_quantized(quantized));
                    } else {
                        // No quantization - store as-is (would need different struct)
                        // For now, still use QuantizedWeight but with identity scaling
                        let fake_quantized = QuantizedTensor {
                            data: shard_data.iter().map(|&b| b as i8).collect(),
                            scales: vec![1.0],
                            shape: shard_shape,
                            mode: QuantizationMode::None,
                        };
                        shards.push(QuantizedWeight::from_quantized(fake_quantized));
                    }
                }
            }
            None => {
                // Replicate to all GPUs
                for _gpu_id in 0..num_gpus {
                    if matches!(self.memory_config.quantization, QuantizationMode::Int8PerChannel | QuantizationMode::Int8PerTensor) {
                        let quantized = quantize_bf16_to_int8_per_channel(&data, &shape);
                        shards.push(QuantizedWeight::from_quantized(quantized));
                    } else {
                        let fake_quantized = QuantizedTensor {
                            data: data.iter().map(|&b| b as i8).collect(),
                            scales: vec![1.0],
                            shape: shape.clone(),
                            mode: QuantizationMode::None,
                        };
                        shards.push(QuantizedWeight::from_quantized(fake_quantized));
                    }
                }
            }
        }

        Ok(shards)
    }

    /// Load raw weight data (for layer norms - keep in BF16)
    fn load_raw_sharded(&self, name: &str, shard_dim: Option<usize>) -> Result<Vec<Vec<u8>>> {
        let (shape, dtype, data) = self.get_weight_data(name)?;
        let num_gpus = self.tp_config.num_gpus;
        let mut shards = Vec::with_capacity(num_gpus);

        match shard_dim {
            Some(dim) => {
                for gpu_id in 0..num_gpus {
                    let shard_data = slice_tensor_data(
                        &data,
                        &shape,
                        dtype.size_bytes(),
                        dim,
                        gpu_id,
                        num_gpus,
                    );
                    shards.push(shard_data);
                }
            }
            None => {
                for _ in 0..num_gpus {
                    shards.push(data.clone());
                }
            }
        }

        Ok(shards)
    }

    /// Load a single layer with quantization
    fn load_layer(&self, layer_idx: usize) -> Result<OptimizedLayerWeights> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Determine primary GPU for this layer (pipeline parallelism)
        let primary_gpu = self
            .pipeline_config
            .as_ref()
            .map(|p| p.gpu_for_layer(layer_idx).unwrap_or(0))
            .unwrap_or(0);

        // Check if layer should be offloaded
        let is_offloaded = self
            .offload_config
            .as_ref()
            .map(|o| o.is_cpu_layer(layer_idx))
            .unwrap_or(false);

        println!(
            "  Loading layer {} (GPU {}, offloaded: {})...",
            layer_idx, primary_gpu, is_offloaded
        );

        // Load quantized weights
        let q_proj = self.load_quantized_sharded(
            &format!("{}.self_attn.q_proj.weight", prefix),
            Some(0),
        )?;
        let k_proj = self.load_quantized_sharded(
            &format!("{}.self_attn.k_proj.weight", prefix),
            Some(0),
        )?;
        let v_proj = self.load_quantized_sharded(
            &format!("{}.self_attn.v_proj.weight", prefix),
            Some(0),
        )?;
        let o_proj = self.load_quantized_sharded(
            &format!("{}.self_attn.o_proj.weight", prefix),
            Some(1),
        )?;
        let gate_proj = self.load_quantized_sharded(
            &format!("{}.mlp.gate_proj.weight", prefix),
            Some(0),
        )?;
        let up_proj = self.load_quantized_sharded(
            &format!("{}.mlp.up_proj.weight", prefix),
            Some(0),
        )?;
        let down_proj = self.load_quantized_sharded(
            &format!("{}.mlp.down_proj.weight", prefix),
            Some(1),
        )?;

        // Layer norms stay in BF16 (small, need precision)
        let input_layernorm = self.load_raw_sharded(
            &format!("{}.input_layernorm.weight", prefix),
            None,
        )?;
        let post_attn_layernorm = self.load_raw_sharded(
            &format!("{}.post_attention_layernorm.weight", prefix),
            None,
        )?;

        Ok(OptimizedLayerWeights {
            layer_idx,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            input_layernorm,
            post_attn_layernorm,
            is_offloaded,
            primary_gpu,
        })
    }

    /// Load all weights with optimization
    pub fn load_all_weights(&self) -> Result<OptimizedModelWeights> {
        let num_layers = self.config.num_layers;
        let num_gpus = self.tp_config.num_gpus;

        println!(
            "Loading Llama model with {} layers across {} GPUs",
            num_layers, num_gpus
        );
        println!("Quantization: {:?}", self.memory_config.quantization);
        println!("Pipeline parallel: {}", self.memory_config.pipeline_parallel);
        println!("CPU offload: {}", self.memory_config.cpu_offload);

        self.print_memory_summary();

        // Load embeddings (quantized, sharded)
        println!("Loading embeddings...");
        let embed_tokens = self.load_quantized_sharded("model.embed_tokens.weight", Some(0))?;

        // Load layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(self.load_layer(i)?);
        }

        // Load final norm (BF16, replicated)
        println!("Loading final norm...");
        let norm = self.load_raw_sharded("model.norm.weight", None)?;

        // Load LM head (quantized, sharded)
        println!("Loading LM head...");
        let lm_head = self.load_quantized_sharded("lm_head.weight", Some(0))?;

        println!("Model loading complete!");

        let weights = OptimizedModelWeights {
            embed_tokens,
            layers,
            norm,
            lm_head,
            memory_config: self.memory_config.clone(),
            pipeline_config: self.pipeline_config.clone(),
        };

        let total_gb = weights.total_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0);
        println!("Total weight memory: {:.2} GB", total_gb);
        println!("Offloaded layers: {}", weights.num_offloaded_layers());

        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_loader_creation() {
        let loader = OptimizedModelLoader::for_4x_a10g(
            "/tmp/fake-model",
            ModelArchitecture::Llama3_1_70B,
        );
        assert!(loader.is_err()); // Path doesn't exist
    }

    #[test]
    fn test_memory_config_for_a10g() {
        let config = MemoryConfig::for_4x_a10g();
        assert!(matches!(config.quantization, QuantizationMode::Int8PerChannel));
        assert!(config.pipeline_parallel);
        assert_eq!(config.pipeline_stages, 4);
    }
}
