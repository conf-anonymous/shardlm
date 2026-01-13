//! Memory optimization strategies for fitting large models on limited GPU memory
//!
//! This module implements three key strategies for Llama 70B on 4x A10G (24GB each):
//!
//! 1. **INT8 Quantization**: Reduce weight memory by 50% (BF16 → INT8)
//! 2. **Pipeline Parallelism**: Distribute layers across GPUs (20 layers per GPU)
//! 3. **CPU Offloading**: Keep less frequently used weights on CPU
//!
//! # Memory Budget Analysis for Llama 70B
//!
//! | Component | BF16 Size | INT8 Size | Notes |
//! |-----------|-----------|-----------|-------|
//! | Embeddings | 2.0 GB | 1.0 GB | vocab × hidden |
//! | Per Layer | 1.68 GB | 0.84 GB | Q,K,V,O + FFN |
//! | 80 Layers | 134 GB | 67 GB | Main model |
//! | LM Head | 2.0 GB | 1.0 GB | tied with embed |
//! | **Total** | **~140 GB** | **~70 GB** | |
//!
//! With INT8 + 4-way tensor parallel: ~17.5 GB per GPU (fits 24GB!)

use serde::{Deserialize, Serialize};

use crate::config::ModelArchitecture;
use crate::quantization::QuantizationMode;

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Weight quantization mode
    pub quantization: QuantizationMode,
    /// Enable pipeline parallelism (layer distribution)
    pub pipeline_parallel: bool,
    /// Number of pipeline stages (usually = num_gpus)
    pub pipeline_stages: usize,
    /// Enable CPU offloading for less active layers
    pub cpu_offload: bool,
    /// Layers to keep on CPU (e.g., first and last few for offload)
    pub cpu_offload_layers: Vec<usize>,
    /// Maximum GPU memory per device (bytes)
    pub max_gpu_memory: usize,
    /// Reserve memory for activations and KV cache (bytes)
    pub activation_reserve: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            quantization: QuantizationMode::None,
            pipeline_parallel: false,
            pipeline_stages: 1,
            cpu_offload: false,
            cpu_offload_layers: Vec::new(),
            max_gpu_memory: 24 * 1024 * 1024 * 1024, // 24GB default (A10G)
            activation_reserve: 4 * 1024 * 1024 * 1024, // 4GB for activations/KV
        }
    }
}

impl MemoryConfig {
    /// Configuration for 4x A10G GPUs with aggressive optimization
    pub fn for_4x_a10g() -> Self {
        Self {
            quantization: QuantizationMode::Int8PerChannel,
            pipeline_parallel: true,
            pipeline_stages: 4,
            cpu_offload: false, // INT8 + pipeline should be enough
            cpu_offload_layers: Vec::new(),
            max_gpu_memory: 24 * 1024 * 1024 * 1024,
            activation_reserve: 4 * 1024 * 1024 * 1024,
        }
    }

    /// Configuration for 2x H100 80GB GPUs
    pub fn for_2x_h100() -> Self {
        Self {
            quantization: QuantizationMode::None, // BF16 fits
            pipeline_parallel: false,
            pipeline_stages: 1,
            cpu_offload: false,
            cpu_offload_layers: Vec::new(),
            max_gpu_memory: 80 * 1024 * 1024 * 1024,
            activation_reserve: 8 * 1024 * 1024 * 1024,
        }
    }

    /// Configuration with CPU offloading (for very constrained memory)
    pub fn with_cpu_offload(num_gpus: usize, offload_layers: Vec<usize>) -> Self {
        Self {
            quantization: QuantizationMode::Int8PerChannel,
            pipeline_parallel: true,
            pipeline_stages: num_gpus,
            cpu_offload: true,
            cpu_offload_layers: offload_layers,
            max_gpu_memory: 24 * 1024 * 1024 * 1024,
            activation_reserve: 4 * 1024 * 1024 * 1024,
        }
    }

    /// Get usable GPU memory after reserves
    pub fn usable_gpu_memory(&self) -> usize {
        self.max_gpu_memory.saturating_sub(self.activation_reserve)
    }

    /// Calculate bytes per weight element based on quantization
    pub fn bytes_per_weight(&self) -> f64 {
        match self.quantization {
            QuantizationMode::None => 2.0, // BF16
            QuantizationMode::Int8PerChannel => 1.0 + 4.0 / 1024.0, // INT8 + scale overhead (~1 byte)
            QuantizationMode::Int8PerTensor => 1.0, // INT8 (single scale)
        }
    }
}

/// Pipeline parallelism configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Layers assigned to each stage
    pub stage_layers: Vec<Vec<usize>>,
    /// GPU assignment for each stage
    pub stage_gpus: Vec<usize>,
}

impl PipelineConfig {
    /// Create a balanced pipeline configuration
    ///
    /// Distributes layers evenly across stages, with each stage on a different GPU
    pub fn balanced(num_layers: usize, num_stages: usize) -> Self {
        let layers_per_stage = num_layers / num_stages;
        let remainder = num_layers % num_stages;

        let mut stage_layers = Vec::with_capacity(num_stages);
        let mut current_layer = 0;

        for stage in 0..num_stages {
            // First 'remainder' stages get one extra layer
            let stage_size = layers_per_stage + if stage < remainder { 1 } else { 0 };
            let layers: Vec<usize> = (current_layer..current_layer + stage_size).collect();
            stage_layers.push(layers);
            current_layer += stage_size;
        }

        let stage_gpus = (0..num_stages).collect();

        Self {
            num_stages,
            stage_layers,
            stage_gpus,
        }
    }

    /// Get the stage that owns a given layer
    pub fn stage_for_layer(&self, layer_idx: usize) -> Option<usize> {
        for (stage, layers) in self.stage_layers.iter().enumerate() {
            if layers.contains(&layer_idx) {
                return Some(stage);
            }
        }
        None
    }

    /// Get the GPU that owns a given layer
    pub fn gpu_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.stage_for_layer(layer_idx).map(|stage| self.stage_gpus[stage])
    }

    /// Get layers for a specific GPU
    pub fn layers_for_gpu(&self, gpu_id: usize) -> Vec<usize> {
        for (stage, &gpu) in self.stage_gpus.iter().enumerate() {
            if gpu == gpu_id {
                return self.stage_layers[stage].clone();
            }
        }
        Vec::new()
    }
}

/// CPU offload configuration
#[derive(Debug, Clone)]
pub struct OffloadConfig {
    /// Layers to keep on CPU
    pub cpu_layers: Vec<usize>,
    /// Layers to keep on GPU
    pub gpu_layers: Vec<usize>,
    /// Prefetch distance (how many layers ahead to start loading)
    pub prefetch_distance: usize,
}

impl OffloadConfig {
    /// Create offload config that keeps middle layers on GPU
    ///
    /// First and last layers see more variance in activations, so middle
    /// layers can be offloaded with less impact on latency
    pub fn offload_edges(num_layers: usize, num_cpu_layers: usize) -> Self {
        let half_offload = num_cpu_layers / 2;
        let cpu_layers: Vec<usize> = (0..half_offload)
            .chain((num_layers - half_offload)..num_layers)
            .collect();
        let gpu_layers: Vec<usize> = (half_offload..(num_layers - half_offload)).collect();

        Self {
            cpu_layers,
            gpu_layers,
            prefetch_distance: 2,
        }
    }

    /// Create offload config for specific layers
    pub fn specific(cpu_layers: Vec<usize>, gpu_layers: Vec<usize>) -> Self {
        Self {
            cpu_layers,
            gpu_layers,
            prefetch_distance: 2,
        }
    }

    /// Check if a layer should be on CPU
    pub fn is_cpu_layer(&self, layer_idx: usize) -> bool {
        self.cpu_layers.contains(&layer_idx)
    }
}

/// Memory estimator for model configurations
pub struct MemoryEstimator {
    arch: ModelArchitecture,
    config: MemoryConfig,
}

impl MemoryEstimator {
    pub fn new(arch: ModelArchitecture, config: MemoryConfig) -> Self {
        Self { arch, config }
    }

    /// Estimate total model weight memory in bytes
    pub fn total_weight_memory(&self) -> usize {
        let hidden = self.arch.hidden_dim();
        let intermediate = self.arch.intermediate_dim();
        let vocab = self.arch.vocab_size();
        let num_layers = self.arch.num_layers();
        let num_heads = self.arch.num_heads();
        let num_kv_heads = self.arch.num_kv_heads();
        let head_dim = hidden / num_heads;

        let bytes_per = self.config.bytes_per_weight();

        // Embeddings + LM head
        let embed_size = (vocab * hidden) as f64 * bytes_per;
        let lm_head_size = (vocab * hidden) as f64 * bytes_per;

        // Per-layer sizes
        let q_proj = (hidden * hidden) as f64 * bytes_per;
        let k_proj = (num_kv_heads * head_dim * hidden) as f64 * bytes_per;
        let v_proj = (num_kv_heads * head_dim * hidden) as f64 * bytes_per;
        let o_proj = (hidden * hidden) as f64 * bytes_per;
        let gate_proj = (intermediate * hidden) as f64 * bytes_per;
        let up_proj = (intermediate * hidden) as f64 * bytes_per;
        let down_proj = (hidden * intermediate) as f64 * bytes_per;
        let layer_norms = (hidden * 2) as f64 * bytes_per; // input + post_attn

        let per_layer = q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + layer_norms;

        let total = embed_size + lm_head_size + (per_layer * num_layers as f64);
        total as usize
    }

    /// Estimate memory per GPU with tensor parallelism
    pub fn memory_per_gpu(&self, num_gpus: usize) -> usize {
        let total = self.total_weight_memory();

        if self.config.pipeline_parallel {
            // Pipeline: each GPU holds ~1/num_gpus of layers
            // Plus replicated embeddings/norms
            let layers_per_gpu = self.arch.num_layers() / num_gpus;
            let layer_memory = self.per_layer_memory() * layers_per_gpu;

            // Embeddings are sharded with tensor parallelism
            let embed_memory = self.embedding_memory() / num_gpus;

            layer_memory + embed_memory
        } else {
            // Pure tensor parallelism: all layers sharded
            total / num_gpus
        }
    }

    /// Estimate memory per layer
    pub fn per_layer_memory(&self) -> usize {
        let hidden = self.arch.hidden_dim();
        let intermediate = self.arch.intermediate_dim();
        let num_heads = self.arch.num_heads();
        let num_kv_heads = self.arch.num_kv_heads();
        let head_dim = hidden / num_heads;
        let bytes_per = self.config.bytes_per_weight();

        let q_proj = (hidden * hidden) as f64 * bytes_per;
        let k_proj = (num_kv_heads * head_dim * hidden) as f64 * bytes_per;
        let v_proj = (num_kv_heads * head_dim * hidden) as f64 * bytes_per;
        let o_proj = (hidden * hidden) as f64 * bytes_per;
        let gate_proj = (intermediate * hidden) as f64 * bytes_per;
        let up_proj = (intermediate * hidden) as f64 * bytes_per;
        let down_proj = (hidden * intermediate) as f64 * bytes_per;
        let layer_norms = (hidden * 2) as f64 * bytes_per;

        (q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + layer_norms) as usize
    }

    /// Estimate embedding memory
    pub fn embedding_memory(&self) -> usize {
        let vocab = self.arch.vocab_size();
        let hidden = self.arch.hidden_dim();
        let bytes_per = self.config.bytes_per_weight();

        // embed_tokens + lm_head (often tied, but counted separately for safety)
        (2.0 * (vocab * hidden) as f64 * bytes_per) as usize
    }

    /// Check if configuration fits in available GPU memory
    pub fn fits_in_memory(&self, num_gpus: usize) -> bool {
        let per_gpu = self.memory_per_gpu(num_gpus);
        let usable = self.config.usable_gpu_memory();
        per_gpu <= usable
    }

    /// Get a summary of memory usage
    pub fn summary(&self, num_gpus: usize) -> MemorySummary {
        MemorySummary {
            total_weight_memory_gb: self.total_weight_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            per_gpu_memory_gb: self.memory_per_gpu(num_gpus) as f64 / (1024.0 * 1024.0 * 1024.0),
            usable_gpu_memory_gb: self.config.usable_gpu_memory() as f64 / (1024.0 * 1024.0 * 1024.0),
            fits: self.fits_in_memory(num_gpus),
            quantization: self.config.quantization,
            pipeline_parallel: self.config.pipeline_parallel,
        }
    }
}

/// Memory usage summary
#[derive(Debug)]
pub struct MemorySummary {
    pub total_weight_memory_gb: f64,
    pub per_gpu_memory_gb: f64,
    pub usable_gpu_memory_gb: f64,
    pub fits: bool,
    pub quantization: QuantizationMode,
    pub pipeline_parallel: bool,
}

impl std::fmt::Display for MemorySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Memory Summary")?;
        writeln!(f, "==============")?;
        writeln!(f, "Total weight memory: {:.2} GB", self.total_weight_memory_gb)?;
        writeln!(f, "Per-GPU memory: {:.2} GB", self.per_gpu_memory_gb)?;
        writeln!(f, "Usable GPU memory: {:.2} GB", self.usable_gpu_memory_gb)?;
        writeln!(f, "Quantization: {:?}", self.quantization)?;
        writeln!(f, "Pipeline parallel: {}", self.pipeline_parallel)?;
        writeln!(f, "Fits in memory: {}", if self.fits { "YES ✓" } else { "NO ✗" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_balanced() {
        let config = PipelineConfig::balanced(80, 4);

        assert_eq!(config.num_stages, 4);
        assert_eq!(config.stage_layers.len(), 4);

        // Check all layers are assigned
        let total_layers: usize = config.stage_layers.iter().map(|l| l.len()).sum();
        assert_eq!(total_layers, 80);

        // Each stage should have 20 layers
        for stage in &config.stage_layers {
            assert_eq!(stage.len(), 20);
        }

        // Check layer lookup
        assert_eq!(config.gpu_for_layer(0), Some(0));
        assert_eq!(config.gpu_for_layer(19), Some(0));
        assert_eq!(config.gpu_for_layer(20), Some(1));
        assert_eq!(config.gpu_for_layer(79), Some(3));
    }

    #[test]
    fn test_memory_estimator_bf16() {
        let config = MemoryConfig::default(); // BF16
        let estimator = MemoryEstimator::new(ModelArchitecture::Llama3_1_70B, config);

        let total_gb = estimator.total_weight_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

        // Llama 70B should be ~140GB in BF16
        assert!(total_gb > 130.0 && total_gb < 150.0, "Total memory: {} GB", total_gb);
    }

    #[test]
    fn test_memory_estimator_int8() {
        let config = MemoryConfig::for_4x_a10g(); // INT8
        let estimator = MemoryEstimator::new(ModelArchitecture::Llama3_1_70B, config);

        let total_gb = estimator.total_weight_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

        // INT8 should be ~70GB
        assert!(total_gb > 65.0 && total_gb < 80.0, "Total memory: {} GB", total_gb);
    }

    #[test]
    fn test_memory_fits_4x_a10g_int8() {
        let config = MemoryConfig::for_4x_a10g();
        let estimator = MemoryEstimator::new(ModelArchitecture::Llama3_1_70B, config);

        let per_gpu_gb = estimator.memory_per_gpu(4) as f64 / (1024.0 * 1024.0 * 1024.0);
        let usable_gb = 20.0; // 24GB - 4GB reserve

        println!("Per GPU: {} GB, Usable: {} GB", per_gpu_gb, usable_gb);

        // With INT8 + pipeline parallel, should fit in ~17-18GB per GPU
        assert!(per_gpu_gb < usable_gb, "Per GPU {} GB exceeds usable {} GB", per_gpu_gb, usable_gb);
    }

    #[test]
    fn test_offload_config() {
        let offload = OffloadConfig::offload_edges(80, 20);

        // First 10 and last 10 layers on CPU
        assert!(offload.is_cpu_layer(0));
        assert!(offload.is_cpu_layer(9));
        assert!(!offload.is_cpu_layer(10));
        assert!(!offload.is_cpu_layer(69));
        assert!(offload.is_cpu_layer(70));
        assert!(offload.is_cpu_layer(79));

        assert_eq!(offload.cpu_layers.len(), 20);
        assert_eq!(offload.gpu_layers.len(), 60);
    }
}
