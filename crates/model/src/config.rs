//! TinyLlama model configuration

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{ModelError, Result};

/// Model configuration (supports TinyLlama, Qwen2, Llama, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinyLlamaConfig {
    /// Hidden size (embedding dimension)
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Intermediate size (FFN hidden dimension)
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    /// Number of attention heads
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads (for GQA)
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,

    /// Number of transformer layers
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,

    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Maximum sequence length
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// RoPE theta
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    /// Whether to tie input/output embeddings (Qwen uses this)
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Whether attention layers have bias (Qwen has QKV bias)
    #[serde(default)]
    pub attention_bias: bool,
}

fn default_hidden_size() -> usize { 2048 }
fn default_intermediate_size() -> usize { 5632 }
fn default_num_attention_heads() -> usize { 32 }
fn default_num_key_value_heads() -> usize { 4 }
fn default_num_hidden_layers() -> usize { 22 }
fn default_vocab_size() -> usize { 32000 }
fn default_max_position_embeddings() -> usize { 2048 }
fn default_rms_norm_eps() -> f64 { 1e-5 }
fn default_rope_theta() -> f64 { 10000.0 }

impl Default for TinyLlamaConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: default_num_key_value_heads(),
            num_hidden_layers: default_num_hidden_layers(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            rms_norm_eps: default_rms_norm_eps(),
            rope_theta: default_rope_theta(),
            tie_word_embeddings: false,
            attention_bias: false,
        }
    }
}

impl TinyLlamaConfig {
    /// Load config from a JSON file (config.json from HuggingFace)
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of key-value groups for GQA
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ModelError::Config(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(ModelError::Config(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TinyLlamaConfig::default();
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.num_hidden_layers, 22);
        config.validate().unwrap();
    }

    #[test]
    fn test_head_dim() {
        let config = TinyLlamaConfig::default();
        assert_eq!(config.head_dim(), 64); // 2048 / 32 = 64
    }
}
