//! Model architecture definitions for Llama 3.x

use crate::config::ModelArchitecture;

/// Llama 3.x model parameters
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (GQA)
    pub num_kv_heads: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Intermediate (FFN) dimension
    pub intermediate_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// RoPE theta base
    pub rope_theta: f32,
    /// Head dimension
    pub head_dim: usize,
}

impl LlamaConfig {
    /// Create config from architecture
    pub fn from_architecture(arch: ModelArchitecture) -> Self {
        let hidden_dim = arch.hidden_dim();
        let num_heads = arch.num_heads();

        Self {
            hidden_dim,
            num_heads,
            num_kv_heads: arch.num_kv_heads(),
            num_layers: arch.num_layers(),
            vocab_size: arch.vocab_size(),
            intermediate_dim: arch.intermediate_dim(),
            max_seq_len: arch.max_context_len(),
            rms_norm_eps: arch.rms_norm_eps(), // Model-specific: Llama 1e-5, Qwen 1e-6
            rope_theta: arch.rope_theta(), // Model-specific: Llama 500K, Qwen 1M
            head_dim: hidden_dim / num_heads,
        }
    }
}

/// Weight tensor metadata
#[derive(Debug, Clone)]
pub struct WeightInfo {
    /// Weight name in safetensors
    pub name: String,
    /// Expected shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: WeightDtype,
}

/// Supported weight data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDtype {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain float 16
    BF16,
    /// 8-bit integer (quantized)
    I8,
    /// 4-bit integer (quantized)
    I4,
}

impl WeightDtype {
    /// Bytes per element
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 => 1,
            Self::I4 => 1, // Packed, but we round up
        }
    }
}

/// List all weight tensors for a Llama model
pub fn list_weights(config: &LlamaConfig) -> Vec<WeightInfo> {
    let mut weights = Vec::new();
    let d = config.hidden_dim;
    let h = config.num_heads;
    let kv_h = config.num_kv_heads;
    let head_dim = config.head_dim;
    let ffn = config.intermediate_dim;
    let v = config.vocab_size;

    // Embeddings
    weights.push(WeightInfo {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![v, d],
        dtype: WeightDtype::BF16,
    });

    // Layers
    for layer in 0..config.num_layers {
        let prefix = format!("model.layers.{layer}");

        // Attention
        weights.push(WeightInfo {
            name: format!("{prefix}.self_attn.q_proj.weight"),
            shape: vec![h * head_dim, d],
            dtype: WeightDtype::BF16,
        });
        weights.push(WeightInfo {
            name: format!("{prefix}.self_attn.k_proj.weight"),
            shape: vec![kv_h * head_dim, d],
            dtype: WeightDtype::BF16,
        });
        weights.push(WeightInfo {
            name: format!("{prefix}.self_attn.v_proj.weight"),
            shape: vec![kv_h * head_dim, d],
            dtype: WeightDtype::BF16,
        });
        weights.push(WeightInfo {
            name: format!("{prefix}.self_attn.o_proj.weight"),
            shape: vec![d, h * head_dim],
            dtype: WeightDtype::BF16,
        });

        // FFN (SwiGLU)
        weights.push(WeightInfo {
            name: format!("{prefix}.mlp.gate_proj.weight"),
            shape: vec![ffn, d],
            dtype: WeightDtype::BF16,
        });
        weights.push(WeightInfo {
            name: format!("{prefix}.mlp.up_proj.weight"),
            shape: vec![ffn, d],
            dtype: WeightDtype::BF16,
        });
        weights.push(WeightInfo {
            name: format!("{prefix}.mlp.down_proj.weight"),
            shape: vec![d, ffn],
            dtype: WeightDtype::BF16,
        });

        // Layer norms
        weights.push(WeightInfo {
            name: format!("{prefix}.input_layernorm.weight"),
            shape: vec![d],
            dtype: WeightDtype::BF16,
        });
        weights.push(WeightInfo {
            name: format!("{prefix}.post_attention_layernorm.weight"),
            shape: vec![d],
            dtype: WeightDtype::BF16,
        });
    }

    // Final layer norm
    weights.push(WeightInfo {
        name: "model.norm.weight".to_string(),
        shape: vec![d],
        dtype: WeightDtype::BF16,
    });

    // LM head
    weights.push(WeightInfo {
        name: "lm_head.weight".to_string(),
        shape: vec![v, d],
        dtype: WeightDtype::BF16,
    });

    weights
}

/// Calculate total model size in bytes
pub fn calculate_model_size(config: &LlamaConfig, dtype: WeightDtype) -> usize {
    let weights = list_weights(config);
    let bytes_per_element = dtype.bytes_per_element();

    weights
        .iter()
        .map(|w| w.shape.iter().product::<usize>() * bytes_per_element)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama_70b_config() {
        let config = LlamaConfig::from_architecture(ModelArchitecture::Llama3_1_70B);
        assert_eq!(config.hidden_dim, 8192);
        assert_eq!(config.num_heads, 64);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.num_layers, 80);
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn test_model_size_70b() {
        let config = LlamaConfig::from_architecture(ModelArchitecture::Llama3_1_70B);
        let size_bf16 = calculate_model_size(&config, WeightDtype::BF16);

        // ~70B parameters * 2 bytes = ~140GB
        let size_gb = size_bf16 as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(size_gb > 130.0 && size_gb < 150.0, "Expected ~140GB, got {size_gb}GB");
    }
}
