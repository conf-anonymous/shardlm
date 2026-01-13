//! ShardLM Model Loading
//!
//! TinyLlama weight loading and export helpers.
//! Handles safetensors format and conversion to fixed-point.

mod config;
mod embedding;
mod error;
mod loader;
mod rope;
mod tokenizer;
mod transformer;
mod weights;

#[cfg(test)]
mod rope_hf_parity;

pub use config::TinyLlamaConfig;
pub use embedding::EmbeddingTable;
pub use error::{ModelError, Result};
pub use loader::SafetensorsLoader;
pub use rope::{apply_rope_batch, apply_rope_f64, apply_rope_fixed, apply_rope_qk, RopeFrequencies};
pub use tokenizer::Tokenizer;
pub use transformer::{compute_logits, KVCache, TransformerState};
pub use weights::{
    AttentionWeights, LinearWeights, MlpWeights, ModelWeights, RmsNormWeights,
    TransformerLayerWeights,
};
