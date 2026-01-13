//! Sharded model loading for tensor parallelism
//!
//! This module handles loading model weights across multiple GPUs
//! with tensor parallelism. Supports Llama and Qwen model families.
//!
//! Supports INT8 quantization for 50% memory reduction.
//! Supports weight caching for fast subsequent loads (~30-60s vs ~20-30min).

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;

use half::bf16;

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_core::gpu::{CudaTensor, CudaTensorBF16, GpuDevice, MultiGpuContext};
use shardlm_v2_core::model::LlamaConfig;
use shardlm_v2_core::parallel::{slice_tensor_data, ShardingPlan, TensorParallelConfig};
use shardlm_v2_core::quantization::{quantize_bf16_to_int8_per_channel, QuantizationMode};
use shardlm_v2_core::tensor::DType;

use crate::error::{ModelError, Result};
use crate::weight_cache::{CacheBuilder, CachedTensor, WeightCache};

/// CPU-resident layer weights for offloading
/// Stores BF16 data as raw bytes for efficient CPU storage
#[derive(Clone)]
pub struct CpuLayerWeights {
    /// Shape and data for each projection per GPU shard
    pub q_proj: Vec<(Vec<usize>, Vec<u8>)>,
    pub k_proj: Vec<(Vec<usize>, Vec<u8>)>,
    pub v_proj: Vec<(Vec<usize>, Vec<u8>)>,
    pub o_proj: Vec<(Vec<usize>, Vec<u8>)>,
    pub gate_proj: Vec<(Vec<usize>, Vec<u8>)>,
    pub up_proj: Vec<(Vec<usize>, Vec<u8>)>,
    pub down_proj: Vec<(Vec<usize>, Vec<u8>)>,
    /// Layer norms as FP32 bytes
    pub input_layernorm: Vec<(Vec<usize>, Vec<u8>)>,
    pub post_attn_layernorm: Vec<(Vec<usize>, Vec<u8>)>,
    /// QKV biases (optional - Qwen has them, Llama doesn't)
    pub q_proj_bias: Option<Vec<(Vec<usize>, Vec<u8>)>>,
    pub k_proj_bias: Option<Vec<(Vec<usize>, Vec<u8>)>>,
    pub v_proj_bias: Option<Vec<(Vec<usize>, Vec<u8>)>>,
}

/// Sharded layer weights on GPU
/// Weight matrices stored as BF16 on GPU to save memory (~70GB vs ~140GB for FP32)
/// BF16→FP32 conversion happens during matmul computation on GPU
pub struct ShardedLayerWeights {
    /// Attention Q projection shards (one per GPU) - BF16 for memory efficiency
    pub q_proj: Vec<CudaTensorBF16>,
    /// Attention K projection (replicated) - BF16 for memory efficiency
    pub k_proj: Vec<CudaTensorBF16>,
    /// Attention V projection (replicated) - BF16 for memory efficiency
    pub v_proj: Vec<CudaTensorBF16>,
    /// Attention O projection shards (one per GPU) - BF16 for memory efficiency
    pub o_proj: Vec<CudaTensorBF16>,
    /// FFN gate projection shards - BF16 for memory efficiency
    pub gate_proj: Vec<CudaTensorBF16>,
    /// FFN up projection shards - BF16 for memory efficiency
    pub up_proj: Vec<CudaTensorBF16>,
    /// FFN down projection shards - BF16 for memory efficiency
    pub down_proj: Vec<CudaTensorBF16>,
    /// Input layer norm (replicated) - FP32 for precision
    pub input_layernorm: Vec<CudaTensor>,
    /// Post-attention layer norm (replicated) - FP32 for precision
    pub post_attn_layernorm: Vec<CudaTensor>,
    /// QKV biases (optional - Qwen has them, Llama doesn't) - FP32 for precision
    pub q_proj_bias: Option<Vec<CudaTensor>>,
    pub k_proj_bias: Option<Vec<CudaTensor>>,
    pub v_proj_bias: Option<Vec<CudaTensor>>,
}

/// Sharded layer weights on GPU - FP32 version for maximum inference speed
/// Uses 2x memory but avoids BF16→FP32 conversion per matmul (25-50x faster!)
pub struct ShardedLayerWeightsF32 {
    /// Attention Q projection shards (one per GPU) - FP32 for speed
    pub q_proj: Vec<CudaTensor>,
    /// Attention K projection (replicated) - FP32 for speed
    pub k_proj: Vec<CudaTensor>,
    /// Attention V projection (replicated) - FP32 for speed
    pub v_proj: Vec<CudaTensor>,
    /// Attention O projection shards (one per GPU) - FP32 for speed
    pub o_proj: Vec<CudaTensor>,
    /// FFN gate projection shards - FP32 for speed
    pub gate_proj: Vec<CudaTensor>,
    /// FFN up projection shards - FP32 for speed
    pub up_proj: Vec<CudaTensor>,
    /// FFN down projection shards - FP32 for speed
    pub down_proj: Vec<CudaTensor>,
    /// Input layer norm (replicated) - FP32 for precision
    pub input_layernorm: Vec<CudaTensor>,
    /// Post-attention layer norm (replicated) - FP32 for precision
    pub post_attn_layernorm: Vec<CudaTensor>,
    /// QKV biases (optional - Qwen has them, Llama doesn't) - FP32
    pub q_proj_bias: Option<Vec<CudaTensor>>,
    pub k_proj_bias: Option<Vec<CudaTensor>>,
    pub v_proj_bias: Option<Vec<CudaTensor>>,
}

impl ShardedLayerWeightsF32 {
    /// Convert a BF16 layer to FP32 for maximum inference speed
    /// This is done once at load time, not per-token
    #[cfg(feature = "cuda")]
    pub fn from_bf16(
        bf16_layer: &ShardedLayerWeights,
        kernels: &[shardlm_v2_core::kernel::KernelContext],
    ) -> Result<Self> {
        // Helper to convert a Vec<CudaTensorBF16> to Vec<CudaTensor>
        let convert_tensors = |bf16_tensors: &[CudaTensorBF16]| -> Result<Vec<CudaTensor>> {
            bf16_tensors.iter().enumerate()
                .map(|(gpu_id, t)| {
                    kernels[gpu_id].bf16_to_f32_tensor(t)
                        .map_err(|e| ModelError::CudaError(e.to_string()))
                })
                .collect()
        };

        // Helper to copy FP32 tensors (layer norms are already FP32)
        let copy_f32_tensors = |f32_tensors: &[CudaTensor]| -> Result<Vec<CudaTensor>> {
            f32_tensors.iter().enumerate()
                .map(|(gpu_id, t)| {
                    kernels[gpu_id].copy_tensor(t)
                        .map_err(|e| ModelError::CudaError(e.to_string()))
                })
                .collect()
        };

        // Copy biases if present (Qwen has them, Llama doesn't)
        let q_proj_bias = bf16_layer.q_proj_bias.as_ref()
            .map(|tensors| copy_f32_tensors(tensors))
            .transpose()?;
        let k_proj_bias = bf16_layer.k_proj_bias.as_ref()
            .map(|tensors| copy_f32_tensors(tensors))
            .transpose()?;
        let v_proj_bias = bf16_layer.v_proj_bias.as_ref()
            .map(|tensors| copy_f32_tensors(tensors))
            .transpose()?;

        Ok(Self {
            q_proj: convert_tensors(&bf16_layer.q_proj)?,
            k_proj: convert_tensors(&bf16_layer.k_proj)?,
            v_proj: convert_tensors(&bf16_layer.v_proj)?,
            o_proj: convert_tensors(&bf16_layer.o_proj)?,
            gate_proj: convert_tensors(&bf16_layer.gate_proj)?,
            up_proj: convert_tensors(&bf16_layer.up_proj)?,
            down_proj: convert_tensors(&bf16_layer.down_proj)?,
            input_layernorm: copy_f32_tensors(&bf16_layer.input_layernorm)?,
            post_attn_layernorm: copy_f32_tensors(&bf16_layer.post_attn_layernorm)?,
            q_proj_bias,
            k_proj_bias,
            v_proj_bias,
        })
    }
}

/// Sharded model weights with CPU offloading
/// Embeddings and lm_head on GPU (frequently accessed)
/// Layers stored on CPU and loaded to GPU on-demand
pub struct ShardedModelWeights {
    /// Token embeddings shards - BF16 on GPU (always resident)
    pub embed_tokens: Vec<CudaTensorBF16>,
    /// Layer weights on CPU (offloaded, loaded on-demand)
    pub cpu_layers: Vec<CpuLayerWeights>,
    /// Currently loaded layer on GPU (only one at a time)
    pub layers: Vec<ShardedLayerWeights>,
    /// Final norm (replicated) - FP32 on GPU
    pub norm: Vec<CudaTensor>,
    /// LM head shards - BF16 on GPU (always resident)
    pub lm_head: Vec<CudaTensorBF16>,
}

impl ShardedModelWeights {
    /// Load a layer from CPU to GPU
    /// This transfers the layer weights from cpu_layers to a new ShardedLayerWeights on GPU.
    /// Projections stay as BF16 on GPU to save memory (~70GB vs ~140GB for FP32).
    /// Layer norms are converted to FP32 for precision.
    pub fn load_layer_to_gpu(
        &self,
        layer_idx: usize,
        gpu_ctx: &MultiGpuContext,
    ) -> Result<ShardedLayerWeights> {
        let cpu_layer = &self.cpu_layers[layer_idx];
        let num_gpus = gpu_ctx.num_gpus;

        // Helper to upload BF16 bytes directly as BF16 on GPU (saves 50% memory)
        let upload_bf16 = |shards: &Vec<(Vec<usize>, Vec<u8>)>| -> Result<Vec<CudaTensorBF16>> {
            let mut tensors = Vec::with_capacity(num_gpus);
            for (gpu_id, (shape, data)) in shards.iter().enumerate() {
                let device = gpu_ctx.device(gpu_id);
                // Upload BF16 bytes directly as BF16 on GPU
                let tensor = CudaTensorBF16::from_bf16_bytes(device, shape.clone(), data)
                    .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                tensors.push(tensor);
            }
            Ok(tensors)
        };

        // Helper to upload as FP32 (for layer norms which need precision)
        let upload_f32 = |shards: &Vec<(Vec<usize>, Vec<u8>)>| -> Result<Vec<CudaTensor>> {
            let mut tensors = Vec::with_capacity(num_gpus);
            for (gpu_id, (shape, data)) in shards.iter().enumerate() {
                let device = gpu_ctx.device(gpu_id);
                // from_bf16_bytes converts BF16→FP32 on CPU, then uploads FP32 to GPU
                let tensor = CudaTensor::from_bf16_bytes(device, shape.clone(), data)
                    .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                tensors.push(tensor);
            }
            Ok(tensors)
        };

        // Projections stored as BF16 on GPU (half memory usage)
        // Layer norms stored as FP32 for numerical precision
        // QKV biases stored as FP32 for precision (optional - Qwen has them, Llama doesn't)
        let q_proj_bias = cpu_layer.q_proj_bias.as_ref()
            .map(|b| upload_f32(b))
            .transpose()?;
        let k_proj_bias = cpu_layer.k_proj_bias.as_ref()
            .map(|b| upload_f32(b))
            .transpose()?;
        let v_proj_bias = cpu_layer.v_proj_bias.as_ref()
            .map(|b| upload_f32(b))
            .transpose()?;

        Ok(ShardedLayerWeights {
            q_proj: upload_bf16(&cpu_layer.q_proj)?,
            k_proj: upload_bf16(&cpu_layer.k_proj)?,
            v_proj: upload_bf16(&cpu_layer.v_proj)?,
            o_proj: upload_bf16(&cpu_layer.o_proj)?,
            gate_proj: upload_bf16(&cpu_layer.gate_proj)?,
            up_proj: upload_bf16(&cpu_layer.up_proj)?,
            down_proj: upload_bf16(&cpu_layer.down_proj)?,
            input_layernorm: upload_f32(&cpu_layer.input_layernorm)?,
            post_attn_layernorm: upload_f32(&cpu_layer.post_attn_layernorm)?,
            q_proj_bias,
            k_proj_bias,
            v_proj_bias,
        })
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.cpu_layers.len()
    }
}

/// Sharded model loader for multi-GPU inference
pub struct ShardedModelLoader {
    /// Model configuration
    config: LlamaConfig,
    /// Model architecture
    architecture: ModelArchitecture,
    /// Path to model directory
    model_dir: std::path::PathBuf,
    /// Memory-mapped files
    mmaps: HashMap<String, Mmap>,
    /// Tensor parallel configuration
    tp_config: TensorParallelConfig,
    /// Sharding plan
    sharding_plan: ShardingPlan,
    /// GPU context
    gpu_ctx: MultiGpuContext,
    /// Quantization mode (INT8 for memory efficiency)
    quantization: QuantizationMode,
    /// Weight cache for fast subsequent loads
    weight_cache: WeightCache,
}

impl ShardedModelLoader {
    /// Create a new sharded model loader (BF16, no quantization)
    pub fn new(
        model_dir: impl AsRef<Path>,
        arch: ModelArchitecture,
        num_gpus: usize,
    ) -> Result<Self> {
        Self::with_quantization(model_dir, arch, num_gpus, QuantizationMode::None)
    }

    /// Create a new sharded model loader with INT8 quantization
    ///
    /// INT8 quantization reduces memory by ~50%, allowing Llama 70B to fit
    /// on 4x A10G (24GB each) GPUs.
    pub fn new_int8(
        model_dir: impl AsRef<Path>,
        arch: ModelArchitecture,
        num_gpus: usize,
    ) -> Result<Self> {
        Self::with_quantization(model_dir, arch, num_gpus, QuantizationMode::Int8PerChannel)
    }

    /// Create a new sharded model loader with specified quantization
    pub fn with_quantization(
        model_dir: impl AsRef<Path>,
        arch: ModelArchitecture,
        num_gpus: usize,
        quantization: QuantizationMode,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        if !model_dir.exists() {
            return Err(ModelError::ModelNotFound(model_dir.display().to_string()));
        }

        let config = LlamaConfig::from_architecture(arch);
        let tp_config = TensorParallelConfig {
            num_gpus,
            enabled: true,
            ..Default::default()
        };

        // Validate and create sharding plan
        tp_config
            .validate(arch)
            .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;

        let sharding_plan = ShardingPlan::for_architecture(arch, num_gpus)
            .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;

        // Initialize GPU context
        let gpu_ctx = MultiGpuContext::new(num_gpus)
            .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;

        // Create weight cache
        let weight_cache = WeightCache::new(&model_dir, num_gpus);

        Ok(Self {
            config,
            architecture: arch,
            model_dir,
            mmaps: HashMap::new(),
            tp_config,
            sharding_plan,
            gpu_ctx,
            quantization,
            weight_cache,
        })
    }

    /// Get quantization mode
    pub fn quantization(&self) -> QuantizationMode {
        self.quantization
    }

    /// Get model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Get sharding plan
    pub fn sharding_plan(&self) -> &ShardingPlan {
        &self.sharding_plan
    }

    /// Get GPU context
    pub fn gpu_ctx(&self) -> &MultiGpuContext {
        &self.gpu_ctx
    }

    /// Check if a valid weight cache exists
    pub fn has_cache(&self) -> bool {
        self.weight_cache.exists()
    }

    /// Delete the weight cache
    pub fn delete_cache(&self) -> Result<()> {
        self.weight_cache.delete()
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &Path {
        self.weight_cache.cache_dir()
    }

    /// List safetensor files in model directory
    pub fn list_safetensor_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(&self.model_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "safetensors") {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    /// Memory-map a safetensor file
    pub fn mmap_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        self.mmaps.insert(filename, mmap);

        Ok(())
    }

    /// Load all safetensor files
    pub fn load_all_files(&mut self) -> Result<()> {
        let files = self.list_safetensor_files()?;

        println!("Found {} safetensor files", files.len());
        for file in &files {
            println!("  Loading: {}", file.display());
            self.mmap_file(file)?;
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

    /// Load a weight matrix as BF16 on GPU (half memory usage)
    ///
    /// Stores weights directly as BF16 on GPU to save memory.
    /// Conversion to FP32 happens during matmul computation.
    fn load_sharded_weight_bf16_native(
        &self,
        name: &str,
        shard_dim: Option<usize>,
    ) -> Result<Vec<CudaTensorBF16>> {
        let (shape, dtype, data) = self.get_weight_data(name)?;
        let num_gpus = self.tp_config.num_gpus;
        let mut shards = Vec::with_capacity(num_gpus);

        // BF16 weights stored directly as BF16 on GPU (no quantization for now)
        // This saves 50% memory vs FP32 while avoiding the complexity of INT8
        match shard_dim {
            Some(dim) => {
                for gpu_id in 0..num_gpus {
                    let shard_data =
                        slice_tensor_data(&data, &shape, dtype.size_bytes(), dim, gpu_id, num_gpus);

                    let mut shard_shape = shape.clone();
                    shard_shape[dim] /= num_gpus;

                    let device = self.gpu_ctx.device(gpu_id);
                    // Store as BF16 on GPU (half memory)
                    let tensor = CudaTensorBF16::from_bf16_bytes(device, shard_shape, &shard_data)
                        .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                    shards.push(tensor);
                }
            }
            None => {
                // Replicate to all GPUs
                for gpu_id in 0..num_gpus {
                    let device = self.gpu_ctx.device(gpu_id);
                    let tensor = CudaTensorBF16::from_bf16_bytes(device, shape.clone(), &data)
                        .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                    shards.push(tensor);
                }
            }
        }

        Ok(shards)
    }

    /// Load a weight matrix to CPU memory (for offloading)
    /// Returns (shape, data) pairs for each GPU shard
    fn load_sharded_weight_to_cpu(
        &self,
        name: &str,
        shard_dim: Option<usize>,
    ) -> Result<Vec<(Vec<usize>, Vec<u8>)>> {
        let (shape, dtype, data) = self.get_weight_data(name)?;
        let num_gpus = self.tp_config.num_gpus;
        let mut shards = Vec::with_capacity(num_gpus);

        match shard_dim {
            Some(dim) => {
                for gpu_id in 0..num_gpus {
                    let shard_data =
                        slice_tensor_data(&data, &shape, dtype.size_bytes(), dim, gpu_id, num_gpus);
                    let mut shard_shape = shape.clone();
                    shard_shape[dim] /= num_gpus;
                    shards.push((shard_shape, shard_data));
                }
            }
            None => {
                // Replicate to all
                for _ in 0..num_gpus {
                    shards.push((shape.clone(), data.clone()));
                }
            }
        }

        Ok(shards)
    }

    /// Load a layer's weights to CPU (for offloading)
    fn load_layer_to_cpu(&self, layer_idx: usize) -> Result<CpuLayerWeights> {
        let prefix = format!("model.layers.{}", layer_idx);

        // Try to load QKV biases (Qwen has them, Llama doesn't)
        let q_proj_bias = self.load_sharded_weight_to_cpu(
            &format!("{}.self_attn.q_proj.bias", prefix), Some(0)
        ).ok();
        let k_proj_bias = self.load_sharded_weight_to_cpu(
            &format!("{}.self_attn.k_proj.bias", prefix), Some(0)
        ).ok();
        let v_proj_bias = self.load_sharded_weight_to_cpu(
            &format!("{}.self_attn.v_proj.bias", prefix), Some(0)
        ).ok();

        Ok(CpuLayerWeights {
            q_proj: self.load_sharded_weight_to_cpu(&format!("{}.self_attn.q_proj.weight", prefix), Some(0))?,
            k_proj: self.load_sharded_weight_to_cpu(&format!("{}.self_attn.k_proj.weight", prefix), Some(0))?,
            v_proj: self.load_sharded_weight_to_cpu(&format!("{}.self_attn.v_proj.weight", prefix), Some(0))?,
            o_proj: self.load_sharded_weight_to_cpu(&format!("{}.self_attn.o_proj.weight", prefix), Some(1))?,
            gate_proj: self.load_sharded_weight_to_cpu(&format!("{}.mlp.gate_proj.weight", prefix), Some(0))?,
            up_proj: self.load_sharded_weight_to_cpu(&format!("{}.mlp.up_proj.weight", prefix), Some(0))?,
            down_proj: self.load_sharded_weight_to_cpu(&format!("{}.mlp.down_proj.weight", prefix), Some(1))?,
            input_layernorm: self.load_sharded_weight_to_cpu(&format!("{}.input_layernorm.weight", prefix), None)?,
            post_attn_layernorm: self.load_sharded_weight_to_cpu(&format!("{}.post_attention_layernorm.weight", prefix), None)?,
            q_proj_bias,
            k_proj_bias,
            v_proj_bias,
        })
    }

    /// Load a small weight (like layer norms) as FP32
    fn load_sharded_weight_f32(
        &self,
        name: &str,
        shard_dim: Option<usize>,
    ) -> Result<Vec<CudaTensor>> {
        let (shape, dtype, data) = self.get_weight_data(name)?;
        let num_gpus = self.tp_config.num_gpus;
        let mut shards = Vec::with_capacity(num_gpus);

        match shard_dim {
            Some(dim) => {
                for gpu_id in 0..num_gpus {
                    let shard_data =
                        slice_tensor_data(&data, &shape, dtype.size_bytes(), dim, gpu_id, num_gpus);
                    let mut shard_shape = shape.clone();
                    shard_shape[dim] /= num_gpus;

                    let device = self.gpu_ctx.device(gpu_id);
                    let tensor = CudaTensor::from_bf16_bytes(device, shard_shape, &shard_data)
                        .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                    shards.push(tensor);
                }
            }
            None => {
                for gpu_id in 0..num_gpus {
                    let device = self.gpu_ctx.device(gpu_id);
                    let tensor = CudaTensor::from_bf16_bytes(device, shape.clone(), &data)
                        .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                    shards.push(tensor);
                }
            }
        }

        Ok(shards)
    }

    /// Load a single layer with sharding
    /// Weight matrices stored as BF16 on GPU (half memory), layer norms as FP32
    pub fn load_layer(&self, layer_idx: usize) -> Result<ShardedLayerWeights> {
        let prefix = format!("model.layers.{}", layer_idx);
        let num_gpus = self.tp_config.num_gpus;

        println!(
            "  Loading layer {} across {} GPUs...",
            layer_idx, num_gpus
        );

        // All projections: Load as BF16 directly on GPU (saves 50% memory)
        // BF16→FP32 conversion happens during matmul on GPU

        // Q projection: shard output dimension (dim 0)
        let q_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.self_attn.q_proj.weight", prefix),
            Some(0),
        )?;

        // K projection: shard output dimension for memory efficiency
        let k_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.self_attn.k_proj.weight", prefix),
            Some(0),
        )?;

        // V projection: shard output dimension for memory efficiency
        let v_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.self_attn.v_proj.weight", prefix),
            Some(0),
        )?;

        // O projection: shard input dimension (dim 1)
        let o_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.self_attn.o_proj.weight", prefix),
            Some(1),
        )?;

        // FFN gate: shard output
        let gate_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.mlp.gate_proj.weight", prefix),
            Some(0),
        )?;

        // FFN up: shard output
        let up_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.mlp.up_proj.weight", prefix),
            Some(0),
        )?;

        // FFN down: shard input
        let down_proj = self.load_sharded_weight_bf16_native(
            &format!("{}.mlp.down_proj.weight", prefix),
            Some(1),
        )?;

        // Layer norms: replicate - FP32 for precision
        let input_layernorm = self.load_sharded_weight_f32(
            &format!("{}.input_layernorm.weight", prefix),
            None,
        )?;

        let post_attn_layernorm = self.load_sharded_weight_f32(
            &format!("{}.post_attention_layernorm.weight", prefix),
            None,
        )?;

        // QKV biases: optional (Qwen has them, Llama doesn't) - FP32 for precision
        // Sharded same as Q/K/V projections (dim 0)
        let q_proj_bias = self.load_sharded_weight_f32(
            &format!("{}.self_attn.q_proj.bias", prefix),
            Some(0),
        ).ok();
        let k_proj_bias = self.load_sharded_weight_f32(
            &format!("{}.self_attn.k_proj.bias", prefix),
            Some(0),
        ).ok();
        let v_proj_bias = self.load_sharded_weight_f32(
            &format!("{}.self_attn.v_proj.bias", prefix),
            Some(0),
        ).ok();

        if q_proj_bias.is_some() {
            println!("    Layer {} has QKV biases (Qwen model)", layer_idx);
        }

        Ok(ShardedLayerWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            up_proj,
            down_proj,
            input_layernorm,
            post_attn_layernorm,
            q_proj_bias,
            k_proj_bias,
            v_proj_bias,
        })
    }

    /// Load a weight without quantization (BF16 -> FP32 on GPU)
    fn load_sharded_weight_bf16(
        &self,
        name: &str,
        shard_dim: Option<usize>,
    ) -> Result<Vec<CudaTensor>> {
        let (shape, dtype, data) = self.get_weight_data(name)?;
        let num_gpus = self.tp_config.num_gpus;
        let mut shards = Vec::with_capacity(num_gpus);

        match shard_dim {
            Some(dim) => {
                for gpu_id in 0..num_gpus {
                    let shard_data =
                        slice_tensor_data(&data, &shape, dtype.size_bytes(), dim, gpu_id, num_gpus);
                    let mut shard_shape = shape.clone();
                    shard_shape[dim] /= num_gpus;

                    let device = self.gpu_ctx.device(gpu_id);
                    // Convert BF16 to FP32 and upload to GPU
                    let tensor = CudaTensor::from_bf16_bytes(device, shard_shape, &shard_data)
                        .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                    shards.push(tensor);
                }
            }
            None => {
                for gpu_id in 0..num_gpus {
                    let device = self.gpu_ctx.device(gpu_id);
                    // Convert BF16 to FP32 and upload to GPU
                    let tensor = CudaTensor::from_bf16_bytes(device, shape.clone(), &data)
                        .map_err(|e| ModelError::InvalidFormat(e.to_string()))?;
                    shards.push(tensor);
                }
            }
        }

        Ok(shards)
    }

    /// Load all model weights with sharding
    ///
    /// If a valid cache exists, loads from cache (fast, ~30-60s).
    /// Otherwise, loads from safetensors with quantization and saves to cache (slow, ~20-30min).
    pub fn load_all_weights(&self) -> Result<ShardedModelWeights> {
        use std::time::Instant;

        // Check if cache exists and is valid
        if self.weight_cache.exists() {
            println!("Found valid weight cache, loading from cache...");
            return self.load_from_cache();
        }

        // No cache, do full quantization and save to cache
        println!("No cache found, quantizing weights (this will take ~20-30 min)...");
        println!("After first load, subsequent loads will be fast (~30-60s)");
        println!();

        let weights = self.load_all_weights_fresh()?;

        // Save to cache for next time
        if matches!(self.quantization, QuantizationMode::Int8PerChannel | QuantizationMode::Int8PerTensor) {
            println!();
            println!("Saving quantized weights to cache...");
            if let Err(e) = self.save_to_cache(&weights) {
                println!("Warning: Failed to save cache: {}", e);
                println!("Next load will require re-quantization.");
            }
        }

        Ok(weights)
    }

    /// Load all weights fresh (no cache) - ALL LAYERS DIRECTLY TO GPU
    /// For H200 with 141GB VRAM, we load all 80 layers (~140GB) to GPU for max performance.
    /// NO CPU offloading - eliminates 80 CPU→GPU transfers per token.
    fn load_all_weights_fresh(&self) -> Result<ShardedModelWeights> {
        use std::time::Instant;

        let num_layers = self.config.num_layers;

        println!("Loading {:?} model with {} layers across {} GPUs...",
            self.architecture, num_layers, self.tp_config.num_gpus);
        println!("Mode: FULL GPU (all layers on GPU - no CPU offloading)");

        // Print memory estimate
        let mem_per_gpu = self.sharding_plan.memory_per_gpu_gb();
        println!("Estimated memory per GPU: {:?} GB", mem_per_gpu);

        // Load embeddings (sharded along vocab dimension) - BF16 on GPU
        println!("Loading embeddings (BF16 on GPU)...");
        let start = Instant::now();
        let embed_tokens = self.load_sharded_weight_bf16_native("model.embed_tokens.weight", Some(0))?;
        println!("  Embeddings loaded in {:.2}s", start.elapsed().as_secs_f64());

        // Load ALL layers DIRECTLY TO GPU (no CPU offloading!)
        // This is the key optimization - eliminates 80 CPU→GPU transfers per token
        // Weights stored as BF16 (half memory: ~70GB vs ~140GB for FP32)
        println!("Loading {} layers DIRECTLY TO GPU (BF16 weights, FP32 norms)...", num_layers);
        let mut layers = Vec::with_capacity(num_layers);
        let layer_start = Instant::now();
        for i in 0..num_layers {
            let start = Instant::now();
            layers.push(self.load_layer(i)?);

            if i % 10 == 0 || i == num_layers - 1 {
                println!("    Layer {}/{} loaded to GPU in {:.2}s", i + 1, num_layers, start.elapsed().as_secs_f64());

                #[cfg(feature = "cuda")]
                {
                    let (used, total) = self.gpu_ctx.device(0).memory_info();
                    println!("      GPU 0 memory: {:.1}/{:.1} GB ({:.1}%)",
                        used as f64 / 1e9, total as f64 / 1e9,
                        (used as f64 / total as f64) * 100.0);
                }
            }
        }
        println!("  All {} layers loaded to GPU in {:.2}s", num_layers, layer_start.elapsed().as_secs_f64());

        // Load final norm (replicated) - FP32 for precision
        println!("Loading final norm (FP32 on GPU)...");
        let norm = self.load_sharded_weight_f32("model.norm.weight", None)?;

        // Load LM head (sharded along vocab dimension) - BF16 on GPU
        // Some models (like Qwen 2.5 small variants) have tied embeddings,
        // meaning lm_head.weight doesn't exist and shares model.embed_tokens.weight
        println!("Loading LM head (BF16 on GPU)...");
        let start = Instant::now();
        let lm_head = match self.load_sharded_weight_bf16_native("lm_head.weight", Some(0)) {
            Ok(weights) => weights,
            Err(ModelError::MissingWeight(_)) => {
                println!("  lm_head.weight not found, using tied embeddings (model.embed_tokens.weight)");
                // Clone embed_tokens for lm_head (they share weights in tied embedding models)
                let mut cloned = Vec::with_capacity(embed_tokens.len());
                for (gpu_id, tensor) in embed_tokens.iter().enumerate() {
                    let device = self.gpu_ctx.device(gpu_id);
                    cloned.push(tensor.clone_on_device(device)
                        .map_err(|e| ModelError::Gpu(e.to_string()))?);
                }
                cloned
            }
            Err(e) => return Err(e),
        };
        println!("  LM head loaded in {:.2}s", start.elapsed().as_secs_f64());

        #[cfg(feature = "cuda")]
        {
            let (used, total) = self.gpu_ctx.device(0).memory_info();
            println!("GPU 0 memory after loading: {:.1}/{:.1} GB ({:.1}%)",
                used as f64 / 1e9, total as f64 / 1e9,
                (used as f64 / total as f64) * 100.0);
        }

        println!("Model loading complete!");
        println!("  - ALL {} layers on GPU (no CPU offloading)", num_layers);
        println!("  - Embeddings and LM head on GPU");
        println!("  - Maximum inference performance enabled");

        Ok(ShardedModelWeights {
            embed_tokens,
            cpu_layers: Vec::new(), // No CPU layers - everything on GPU!
            layers,                  // All layers on GPU
            norm,
            lm_head,
        })
    }

    /// Save weights to cache for fast subsequent loads
    fn save_to_cache(&self, weights: &ShardedModelWeights) -> Result<()> {
        use std::time::Instant;
        let start = Instant::now();

        let mut cache_builder = CacheBuilder::new(
            &self.model_dir,
            self.architecture,
            self.quantization,
            self.tp_config.num_gpus,
            &self.config,
        )?;

        // Save embeddings (BF16)
        println!("  Caching embeddings...");
        let embed_tensor = self.cuda_tensors_bf16_to_cached(&weights.embed_tokens)?;
        cache_builder.save_tensor("embed_tokens", &embed_tensor)?;

        // Save layers (either GPU layers or CPU-offloaded layers)
        if !weights.layers.is_empty() {
            // GPU layers mode
            for (i, layer) in weights.layers.iter().enumerate() {
                println!("  Caching layer {}/{}...", i + 1, weights.layers.len());
                self.save_layer_to_cache(&mut cache_builder, i, layer)?;
            }
        } else if !weights.cpu_layers.is_empty() {
            // CPU offloading mode - cache the CPU layers
            for (i, cpu_layer) in weights.cpu_layers.iter().enumerate() {
                if i % 10 == 0 || i == weights.cpu_layers.len() - 1 {
                    println!("  Caching layer {}/{}...", i + 1, weights.cpu_layers.len());
                }
                self.save_cpu_layer_to_cache(&mut cache_builder, i, cpu_layer)?;
            }
        }

        // Save norm (FP32 -> BF16 for cache)
        println!("  Caching norm...");
        let norm_tensor = self.cuda_tensors_to_cached(&weights.norm, DType::BF16)?;
        cache_builder.save_tensor("norm", &norm_tensor)?;

        // Save lm_head (BF16)
        println!("  Caching lm_head...");
        let lm_head_tensor = self.cuda_tensors_bf16_to_cached(&weights.lm_head)?;
        cache_builder.save_tensor("lm_head", &lm_head_tensor)?;

        cache_builder.finalize()?;
        println!("Cache saved in {:.2}s", start.elapsed().as_secs_f64());

        Ok(())
    }

    /// Save a layer's weights to cache
    fn save_layer_to_cache(
        &self,
        cache_builder: &mut CacheBuilder,
        layer_idx: usize,
        layer: &ShardedLayerWeights,
    ) -> Result<()> {
        let prefix = format!("layer_{}", layer_idx);

        // Projections are BF16 on GPU, save directly as BF16
        cache_builder.save_tensor(&format!("{}_q_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.q_proj)?)?;
        cache_builder.save_tensor(&format!("{}_k_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.k_proj)?)?;
        cache_builder.save_tensor(&format!("{}_v_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.v_proj)?)?;
        cache_builder.save_tensor(&format!("{}_o_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.o_proj)?)?;
        cache_builder.save_tensor(&format!("{}_gate_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.gate_proj)?)?;
        cache_builder.save_tensor(&format!("{}_up_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.up_proj)?)?;
        cache_builder.save_tensor(&format!("{}_down_proj", prefix), &self.cuda_tensors_bf16_to_cached(&layer.down_proj)?)?;

        // Layer norms are FP32 -> BF16 for cache
        cache_builder.save_tensor(&format!("{}_input_layernorm", prefix), &self.cuda_tensors_to_cached(&layer.input_layernorm, DType::BF16)?)?;
        cache_builder.save_tensor(&format!("{}_post_attn_layernorm", prefix), &self.cuda_tensors_to_cached(&layer.post_attn_layernorm, DType::BF16)?)?;

        // QKV biases (optional - Qwen has them, Llama doesn't) - save as BF16
        if let Some(ref bias) = layer.q_proj_bias {
            cache_builder.save_tensor(&format!("{}_q_proj_bias", prefix), &self.cuda_tensors_to_cached(bias, DType::BF16)?)?;
        }
        if let Some(ref bias) = layer.k_proj_bias {
            cache_builder.save_tensor(&format!("{}_k_proj_bias", prefix), &self.cuda_tensors_to_cached(bias, DType::BF16)?)?;
        }
        if let Some(ref bias) = layer.v_proj_bias {
            cache_builder.save_tensor(&format!("{}_v_proj_bias", prefix), &self.cuda_tensors_to_cached(bias, DType::BF16)?)?;
        }

        Ok(())
    }

    /// Save a CPU layer's weights to cache (for CPU offloading mode)
    fn save_cpu_layer_to_cache(
        &self,
        cache_builder: &mut CacheBuilder,
        layer_idx: usize,
        layer: &CpuLayerWeights,
    ) -> Result<()> {
        let prefix = format!("layer_{}", layer_idx);

        // Helper to convert CPU layer format to CachedTensor
        let cpu_to_cached = |data: &[(Vec<usize>, Vec<u8>)]| -> CachedTensor {
            let shape = data.first().map(|(s, _)| s.clone()).unwrap_or_default();
            let shards = data.iter().map(|(_, d)| d.clone()).collect();
            CachedTensor {
                shape,
                dtype: DType::BF16,
                shards,
            }
        };

        // Weight matrices are already BF16 bytes
        cache_builder.save_tensor(&format!("{}_q_proj", prefix), &cpu_to_cached(&layer.q_proj))?;
        cache_builder.save_tensor(&format!("{}_k_proj", prefix), &cpu_to_cached(&layer.k_proj))?;
        cache_builder.save_tensor(&format!("{}_v_proj", prefix), &cpu_to_cached(&layer.v_proj))?;
        cache_builder.save_tensor(&format!("{}_o_proj", prefix), &cpu_to_cached(&layer.o_proj))?;
        cache_builder.save_tensor(&format!("{}_gate_proj", prefix), &cpu_to_cached(&layer.gate_proj))?;
        cache_builder.save_tensor(&format!("{}_up_proj", prefix), &cpu_to_cached(&layer.up_proj))?;
        cache_builder.save_tensor(&format!("{}_down_proj", prefix), &cpu_to_cached(&layer.down_proj))?;

        // Layer norms
        cache_builder.save_tensor(&format!("{}_input_layernorm", prefix), &cpu_to_cached(&layer.input_layernorm))?;
        cache_builder.save_tensor(&format!("{}_post_attn_layernorm", prefix), &cpu_to_cached(&layer.post_attn_layernorm))?;

        // QKV biases (optional - Qwen has them, Llama doesn't)
        if let Some(ref bias) = layer.q_proj_bias {
            cache_builder.save_tensor(&format!("{}_q_proj_bias", prefix), &cpu_to_cached(bias))?;
        }
        if let Some(ref bias) = layer.k_proj_bias {
            cache_builder.save_tensor(&format!("{}_k_proj_bias", prefix), &cpu_to_cached(bias))?;
        }
        if let Some(ref bias) = layer.v_proj_bias {
            cache_builder.save_tensor(&format!("{}_v_proj_bias", prefix), &cpu_to_cached(bias))?;
        }

        Ok(())
    }

    /// Convert CudaTensors (FP32) to CachedTensor format
    fn cuda_tensors_to_cached(&self, tensors: &[CudaTensor], _dtype: DType) -> Result<CachedTensor> {
        let mut shards = Vec::with_capacity(tensors.len());
        let mut shape = Vec::new();

        for (i, tensor) in tensors.iter().enumerate() {
            let device = self.gpu_ctx.device(tensor.device_id);
            // Download FP32 from GPU and convert to BF16 for cache storage
            let f32_data = tensor.to_f32_host(device)
                .map_err(|e| ModelError::InvalidFormat(format!("Failed to download tensor from GPU {}: {}", i, e)))?;
            let bf16_data: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
                .collect();
            if shape.is_empty() {
                shape = tensor.shape.clone();
            }
            shards.push(bf16_data);
        }

        Ok(CachedTensor {
            shape,
            dtype: DType::BF16,  // Always store as BF16 in cache
            shards,
        })
    }

    /// Convert CudaTensorBF16 to CachedTensor format (already BF16)
    fn cuda_tensors_bf16_to_cached(&self, tensors: &[CudaTensorBF16]) -> Result<CachedTensor> {
        let mut shards = Vec::with_capacity(tensors.len());
        let mut shape = Vec::new();

        for (i, tensor) in tensors.iter().enumerate() {
            let device = self.gpu_ctx.device(tensor.device_id);
            // Download BF16 bytes directly
            let bf16_data = tensor.to_host(device)
                .map_err(|e| ModelError::InvalidFormat(format!("Failed to download BF16 tensor from GPU {}: {}", i, e)))?;
            if shape.is_empty() {
                shape = tensor.shape.clone();
            }
            shards.push(bf16_data);
        }

        Ok(CachedTensor {
            shape,
            dtype: DType::BF16,
            shards,
        })
    }

    /// Load all weights from cache - ALL LAYERS DIRECTLY TO GPU
    /// For H200 with 141GB VRAM, we load all 80 layers (~140GB) to GPU for max performance.
    fn load_from_cache(&self) -> Result<ShardedModelWeights> {
        use std::time::Instant;
        let start = Instant::now();

        let num_layers = self.config.num_layers;
        let num_gpus = self.tp_config.num_gpus;

        println!("Loading {} layers across {} GPUs from cache (FULL GPU mode)...", num_layers, num_gpus);

        // Load embeddings as BF16
        println!("  Loading embeddings from cache (BF16)...");
        let embed_tensor = self.weight_cache.load_tensor("embed_tokens")?;
        let embed_tokens = self.cached_to_cuda_tensors_bf16(&embed_tensor)?;

        // Load ALL layers DIRECTLY TO GPU (no CPU offloading!)
        // Weights stored as BF16 (half memory)
        println!("  Loading {} layers DIRECTLY TO GPU (BF16 weights)...", num_layers);
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i % 10 == 0 || i == num_layers - 1 {
                println!("    Layer {}/{}...", i + 1, num_layers);

                #[cfg(feature = "cuda")]
                {
                    let (used, total) = self.gpu_ctx.device(0).memory_info();
                    println!("      GPU 0 memory: {:.1}/{:.1} GB ({:.1}%)",
                        used as f64 / 1e9, total as f64 / 1e9,
                        (used as f64 / total as f64) * 100.0);
                }
            }
            layers.push(self.load_layer_from_cache_to_gpu(i)?);
        }

        // Load norm as FP32
        println!("  Loading norm from cache (FP32)...");
        let norm_tensor = self.weight_cache.load_tensor("norm")?;
        let norm = self.cached_to_cuda_tensors(&norm_tensor)?;

        // Load lm_head as BF16
        println!("  Loading lm_head from cache (BF16)...");
        let lm_head_tensor = self.weight_cache.load_tensor("lm_head")?;
        let lm_head = self.cached_to_cuda_tensors_bf16(&lm_head_tensor)?;

        let elapsed = start.elapsed();
        println!();
        println!("Cache load complete in {:.2}s ({:.2} GB/s)",
            elapsed.as_secs_f64(),
            self.total_size_gb() / elapsed.as_secs_f64());

        #[cfg(feature = "cuda")]
        {
            let (used, total) = self.gpu_ctx.device(0).memory_info();
            println!("GPU 0 memory after loading: {:.1}/{:.1} GB ({:.1}%)",
                used as f64 / 1e9, total as f64 / 1e9,
                (used as f64 / total as f64) * 100.0);
        }

        println!("  - ALL {} layers on GPU (no CPU offloading)", num_layers);
        println!("  - Maximum inference performance enabled");

        Ok(ShardedModelWeights {
            embed_tokens,
            cpu_layers: Vec::new(), // No CPU layers - everything on GPU!
            layers,                  // All layers on GPU
            norm,
            lm_head,
        })
    }

    /// Load a layer from cache directly to GPU (for full GPU mode)
    /// Projections loaded as BF16 (half memory), layer norms as FP32 (for precision)
    fn load_layer_from_cache_to_gpu(&self, layer_idx: usize) -> Result<ShardedLayerWeights> {
        let prefix = format!("layer_{}", layer_idx);

        // Try to load QKV biases from cache (optional - Qwen has them, Llama doesn't)
        let q_proj_bias = self.weight_cache.load_tensor(&format!("{}_q_proj_bias", prefix))
            .ok()
            .map(|cached| self.cached_to_cuda_tensors(&cached))
            .transpose()?;
        let k_proj_bias = self.weight_cache.load_tensor(&format!("{}_k_proj_bias", prefix))
            .ok()
            .map(|cached| self.cached_to_cuda_tensors(&cached))
            .transpose()?;
        let v_proj_bias = self.weight_cache.load_tensor(&format!("{}_v_proj_bias", prefix))
            .ok()
            .map(|cached| self.cached_to_cuda_tensors(&cached))
            .transpose()?;

        Ok(ShardedLayerWeights {
            // Projections as BF16 (saves 50% memory)
            q_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_q_proj", prefix))?)?,
            k_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_k_proj", prefix))?)?,
            v_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_v_proj", prefix))?)?,
            o_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_o_proj", prefix))?)?,
            gate_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_gate_proj", prefix))?)?,
            up_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_up_proj", prefix))?)?,
            down_proj: self.cached_to_cuda_tensors_bf16(&self.weight_cache.load_tensor(&format!("{}_down_proj", prefix))?)?,
            // Layer norms as FP32 (for precision)
            input_layernorm: self.cached_to_cuda_tensors(&self.weight_cache.load_tensor(&format!("{}_input_layernorm", prefix))?)?,
            post_attn_layernorm: self.cached_to_cuda_tensors(&self.weight_cache.load_tensor(&format!("{}_post_attn_layernorm", prefix))?)?,
            q_proj_bias,
            k_proj_bias,
            v_proj_bias,
        })
    }

    /// Load a layer from cache to CPU (for offloading)
    fn load_layer_from_cache_to_cpu(&self, layer_idx: usize) -> Result<CpuLayerWeights> {
        let prefix = format!("layer_{}", layer_idx);

        // Helper to convert CachedTensor to CPU format
        let cached_to_cpu = |cached: CachedTensor| -> Vec<(Vec<usize>, Vec<u8>)> {
            cached.shards.into_iter()
                .map(|shard_data| (cached.shape.clone(), shard_data))
                .collect()
        };

        // Try to load QKV biases from cache (optional - Qwen has them, Llama doesn't)
        let q_proj_bias = self.weight_cache.load_tensor(&format!("{}_q_proj_bias", prefix))
            .ok()
            .map(cached_to_cpu);
        let k_proj_bias = self.weight_cache.load_tensor(&format!("{}_k_proj_bias", prefix))
            .ok()
            .map(cached_to_cpu);
        let v_proj_bias = self.weight_cache.load_tensor(&format!("{}_v_proj_bias", prefix))
            .ok()
            .map(cached_to_cpu);

        Ok(CpuLayerWeights {
            q_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_q_proj", prefix))?),
            k_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_k_proj", prefix))?),
            v_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_v_proj", prefix))?),
            o_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_o_proj", prefix))?),
            gate_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_gate_proj", prefix))?),
            up_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_up_proj", prefix))?),
            down_proj: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_down_proj", prefix))?),
            input_layernorm: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_input_layernorm", prefix))?),
            post_attn_layernorm: cached_to_cpu(self.weight_cache.load_tensor(&format!("{}_post_attn_layernorm", prefix))?),
            q_proj_bias,
            k_proj_bias,
            v_proj_bias,
        })
    }

    /// Convert CachedTensor to CudaTensors (loads BF16 from cache, converts to FP32 on GPU)
    fn cached_to_cuda_tensors(&self, cached: &CachedTensor) -> Result<Vec<CudaTensor>> {
        let num_gpus = self.tp_config.num_gpus;

        if cached.shards.len() != num_gpus {
            return Err(ModelError::InvalidFormat(format!(
                "Cache has {} shards but expected {}", cached.shards.len(), num_gpus
            )));
        }

        let mut tensors = Vec::with_capacity(num_gpus);
        for (gpu_id, shard_data) in cached.shards.iter().enumerate() {
            let device = self.gpu_ctx.device(gpu_id);
            // Convert BF16 bytes to FP32 and upload to GPU
            let tensor = CudaTensor::from_bf16_bytes(device, cached.shape.clone(), shard_data)
                .map_err(|e| ModelError::InvalidFormat(format!("Failed to upload to GPU {}: {}", gpu_id, e)))?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Convert CachedTensor to CudaTensorBF16 (stays as BF16 on GPU)
    fn cached_to_cuda_tensors_bf16(&self, cached: &CachedTensor) -> Result<Vec<CudaTensorBF16>> {
        let num_gpus = self.tp_config.num_gpus;

        if cached.shards.len() != num_gpus {
            return Err(ModelError::InvalidFormat(format!(
                "Cache has {} shards but expected {}", cached.shards.len(), num_gpus
            )));
        }

        let mut tensors = Vec::with_capacity(num_gpus);
        for (gpu_id, shard_data) in cached.shards.iter().enumerate() {
            let device = self.gpu_ctx.device(gpu_id);
            // Upload BF16 bytes directly as BF16 on GPU
            let tensor = CudaTensorBF16::from_bf16_bytes(device, cached.shape.clone(), shard_data)
                .map_err(|e| ModelError::InvalidFormat(format!("Failed to upload BF16 to GPU {}: {}", gpu_id, e)))?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Get total model size in GB
    pub fn total_size_gb(&self) -> f64 {
        self.mmaps.values().map(|m| m.len()).sum::<usize>() as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharded_loader_creation() {
        // This will fail since path doesn't exist, but tests the validation
        let loader = ShardedModelLoader::new(
            "/tmp/fake-model",
            ModelArchitecture::Llama3_1_70B,
            4,
        );
        assert!(loader.is_err());
    }
}
