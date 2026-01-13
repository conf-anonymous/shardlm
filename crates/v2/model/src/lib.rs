//! ShardLM v2 Model - Llama 70B model loading and inference
//!
//! This crate provides:
//! - Memory-mapped model loading for large (140GB+) models
//! - Tensor parallel loading across multiple GPUs
//! - KV cache management for long context
//! - Distributed forward pass with tensor parallelism
//! - Flash Attention integration
//! - INT8 quantization for memory efficiency
//! - Pipeline parallelism for layer distribution
//! - CPU offloading for constrained memory
//! - Weight caching for fast subsequent loads

pub mod distributed;
pub mod error;
pub mod kv_cache;
pub mod loader;
pub mod optimized_loader;
pub mod sharded_loader;
pub mod tokenizer;
pub mod transformer;
pub mod weight_cache;

pub use distributed::DistributedEngine;
pub use error::{ModelError, Result};
pub use loader::ModelLoader;
pub use optimized_loader::{OptimizedModelLoader, OptimizedModelWeights};
pub use sharded_loader::{ShardedLayerWeights, ShardedModelLoader, ShardedModelWeights};
pub use tokenizer::{ChatFormat, ChatMessage, Tokenizer};
pub use transformer::Transformer;
pub use weight_cache::{CacheBuilder, CacheManifest, CachedTensor, WeightCache};
