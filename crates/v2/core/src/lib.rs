//! ShardLM v2 Core - Production-grade privacy-preserving inference
//!
//! This crate provides the core infrastructure for v2 of ShardLM,
//! targeting Llama 70B models with CUDA GPU acceleration.
//!
//! # Architecture
//!
//! v2 is designed for production deployment with:
//! - **Llama 3.1/3.3 70B support**: 128K context window, GQA attention
//! - **CUDA acceleration**: Tensor cores, flash attention, optimized kernels
//! - **Multi-GPU support**: Tensor parallelism across 4+ GPUs
//! - **FP16/BF16 inference**: Half-precision for memory efficiency
//!
//! # Key Differences from v1
//!
//! | Feature | v1 | v2 |
//! |---------|-----|-----|
//! | Target Model | Qwen 2.5 1.5B | Llama 70B |
//! | Context Window | 2048 | 128K |
//! | Compute | CPU/NEON | CUDA/Tensor Cores |
//! | Precision | i32 fixed-point | FP16/BF16 |
//! | Attention | Standard MHA | Grouped Query Attention |
//!
//! # Multi-GPU Support
//!
//! With 4x A10G GPUs (92GB total), Llama 70B (~140GB BF16) requires:
//! - Tensor parallelism: Shard weights across GPUs
//! - Column parallel for QKV and FFN up projections
//! - Row parallel for output and FFN down projections
//! - All-reduce for aggregating distributed outputs

pub mod comm;
pub mod config;
pub mod error;
pub mod gpu;
pub mod kernel;
pub mod memory;
pub mod model;
pub mod parallel;
pub mod quantization;
pub mod tensor;

pub use comm::{CommOp, GpuCommunicator};
pub use config::{ModelArchitecture, ModelConfig, ServerConfig, V2Config};
pub use error::{Result, V2Error};
pub use gpu::{CudaTensor, GpuBuffer, GpuDevice, MultiGpuContext};
pub use kernel::KernelContext;
pub use memory::{MemoryConfig, MemoryEstimator, OffloadConfig, PipelineConfig};
pub use model::LlamaConfig;
pub use parallel::{slice_tensor_data, ShardSpec, ShardingPlan, TensorParallelConfig};
pub use quantization::{QuantizationMode, QuantizedTensor};
pub use tensor::DType;
