//! Model weight structures

use shardlm_fixed_point::FixedVector;

use crate::config::TinyLlamaConfig;
use crate::embedding::EmbeddingTable;
use crate::error::{ModelError, Result};

/// Weights for a linear layer (Y = XW + b)
#[derive(Debug, Clone)]
pub struct LinearWeights {
    /// Weight matrix (in_features × out_features), stored as fixed-point
    pub weight: Vec<i32>,
    /// Bias vector (out_features), stored as fixed-point (optional)
    pub bias: Option<Vec<i32>>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
    /// Fixed-point scale
    pub scale: u8,
}

impl LinearWeights {
    /// Create from f32 weight matrix (row-major) and optional bias
    pub fn from_f32(
        weight: &[f32],
        bias: Option<&[f32]>,
        in_features: usize,
        out_features: usize,
        scale: u8,
    ) -> Result<Self> {
        if weight.len() != in_features * out_features {
            return Err(ModelError::InvalidShape {
                expected: vec![in_features, out_features],
                got: vec![weight.len()],
            });
        }

        let scale_factor = (1u64 << scale) as f64;
        let weight_fixed: Vec<i32> = weight
            .iter()
            .map(|&x| (x as f64 * scale_factor).round() as i32)
            .collect();

        let bias_fixed = if let Some(b) = bias {
            if b.len() != out_features {
                return Err(ModelError::InvalidShape {
                    expected: vec![out_features],
                    got: vec![b.len()],
                });
            }
            Some(
                b.iter()
                    .map(|&x| (x as f64 * scale_factor).round() as i32)
                    .collect(),
            )
        } else {
            None
        };

        Ok(Self {
            weight: weight_fixed,
            bias: bias_fixed,
            in_features,
            out_features,
            scale,
        })
    }

    /// Create a random linear layer (for testing)
    pub fn random(in_features: usize, out_features: usize, scale: u8) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize with small random values (Kaiming-like)
        let std_dev = (2.0 / in_features as f64).sqrt();
        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|_| {
                let val: f64 = rng.gen_range(-std_dev..std_dev);
                let scale_factor = (1u64 << scale) as f64;
                (val * scale_factor).round() as i32
            })
            .collect();

        Self {
            weight,
            bias: None,
            in_features,
            out_features,
            scale,
        }
    }

    /// Get weight at (in_idx, out_idx)
    pub fn get_weight(&self, in_idx: usize, out_idx: usize) -> i32 {
        self.weight[in_idx * self.out_features + out_idx]
    }

    /// Get bias vector as FixedVector
    pub fn get_bias(&self) -> Option<FixedVector> {
        self.bias
            .as_ref()
            .map(|b| FixedVector::from_raw(b.clone(), self.scale))
    }

    /// Convert to SharedMatrix for secure computation
    pub fn to_shared_matrix(&self) -> shardlm_sharing::SharedMatrix {
        shardlm_sharing::SharedMatrix::from_raw(
            self.weight.clone(),
            self.in_features,
            self.out_features,
            self.scale,
        )
        .expect("Dimensions already validated")
    }
}

/// RMS LayerNorm weights (just the gamma/weight vector)
#[derive(Debug, Clone)]
pub struct RmsNormWeights {
    /// Weight vector (hidden_size)
    pub weight: Vec<i32>,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Fixed-point scale
    pub scale: u8,
    /// Epsilon for numerical stability
    pub eps: f64,
}

impl RmsNormWeights {
    /// Create from raw i32 weights
    pub fn from_raw(weight: Vec<i32>, scale: u8, eps: f64) -> Self {
        let hidden_size = weight.len();
        Self {
            weight,
            hidden_size,
            scale,
            eps,
        }
    }

    /// Create with all ones (identity normalization)
    pub fn ones(hidden_size: usize, scale: u8, eps: f64) -> Self {
        Self {
            weight: vec![1 << scale; hidden_size],
            hidden_size,
            scale,
            eps,
        }
    }
}

/// Attention weights for a single layer
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// Query projection: hidden_size -> num_heads * head_dim
    pub q_proj: LinearWeights,
    /// Key projection: hidden_size -> num_kv_heads * head_dim
    pub k_proj: LinearWeights,
    /// Value projection: hidden_size -> num_kv_heads * head_dim
    pub v_proj: LinearWeights,
    /// Output projection: num_heads * head_dim -> hidden_size
    pub o_proj: LinearWeights,
}

/// MLP (FFN) weights for a single layer using SwiGLU activation
#[derive(Debug, Clone)]
pub struct MlpWeights {
    /// Gate projection: hidden_size -> intermediate_size
    pub gate_proj: LinearWeights,
    /// Up projection: hidden_size -> intermediate_size
    pub up_proj: LinearWeights,
    /// Down projection: intermediate_size -> hidden_size
    pub down_proj: LinearWeights,
}

/// Weights for a single transformer layer
#[derive(Debug, Clone)]
pub struct TransformerLayerWeights {
    /// Input LayerNorm (before attention)
    pub input_layernorm: RmsNormWeights,
    /// Self-attention weights
    pub self_attn: AttentionWeights,
    /// Post-attention LayerNorm (before MLP)
    pub post_attention_layernorm: RmsNormWeights,
    /// MLP (FFN) weights
    pub mlp: MlpWeights,
}

/// Complete model weights with all transformer layers
#[derive(Debug)]
pub struct ModelWeights {
    /// Model configuration
    pub config: TinyLlamaConfig,
    /// Embedding table
    pub embeddings: EmbeddingTable,
    /// All transformer layers
    pub layers: Vec<TransformerLayerWeights>,
    /// Final RMS LayerNorm
    pub final_norm: RmsNormWeights,
    /// LM head (for logit computation) - may be tied to embeddings
    pub lm_head: LinearWeights,
    /// Fixed-point scale
    pub scale: u8,
}

impl ModelWeights {
    /// Create random model weights for testing
    pub fn random(config: TinyLlamaConfig, scale: u8) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let num_layers = config.num_hidden_layers;
        let eps = config.rms_norm_eps;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerLayerWeights {
                input_layernorm: RmsNormWeights::ones(hidden_size, scale, eps),
                self_attn: AttentionWeights {
                    q_proj: LinearWeights::random(hidden_size, num_heads * head_dim, scale),
                    k_proj: LinearWeights::random(hidden_size, num_kv_heads * head_dim, scale),
                    v_proj: LinearWeights::random(hidden_size, num_kv_heads * head_dim, scale),
                    o_proj: LinearWeights::random(num_heads * head_dim, hidden_size, scale),
                },
                post_attention_layernorm: RmsNormWeights::ones(hidden_size, scale, eps),
                mlp: MlpWeights {
                    gate_proj: LinearWeights::random(hidden_size, intermediate_size, scale),
                    up_proj: LinearWeights::random(hidden_size, intermediate_size, scale),
                    down_proj: LinearWeights::random(intermediate_size, hidden_size, scale),
                },
            });
        }

        Self {
            embeddings: EmbeddingTable::random(config.vocab_size, hidden_size, scale),
            layers,
            final_norm: RmsNormWeights::ones(hidden_size, scale, eps),
            lm_head: LinearWeights::random(hidden_size, config.vocab_size, scale),
            config,
            scale,
        }
    }

    /// Get embedding dimension
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_linear_weights_from_f32() {
        let weight: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bias: Vec<f32> = vec![0.1, 0.2, 0.3];
        let linear = LinearWeights::from_f32(&weight, Some(&bias), 2, 3, DEFAULT_SCALE).unwrap();

        assert_eq!(linear.in_features, 2);
        assert_eq!(linear.out_features, 3);
        assert!(linear.bias.is_some());
    }

    #[test]
    fn test_random_weights() {
        let config = TinyLlamaConfig::default();
        let weights = ModelWeights::random(config, DEFAULT_SCALE);

        assert_eq!(weights.hidden_size(), 2048);
        assert_eq!(weights.vocab_size(), 32000);
        assert_eq!(weights.num_layers(), 22);
    }
}
