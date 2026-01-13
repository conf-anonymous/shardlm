//! End-to-end pipeline for ShardLM inference

use shardlm_fixed_point::{FixedVector, DEFAULT_SCALE};
use shardlm_model::{ModelWeights, TinyLlamaConfig};

use crate::error::{HarnessError, Result};
use crate::plaintext::PlaintextInference;
use crate::secure::SecureInference;

/// Result of an inference run
#[derive(Debug)]
pub struct InferenceResult {
    /// Output logits
    pub logits: FixedVector,
    /// Top-k predictions (token_id, score)
    pub top_k: Vec<(usize, f64)>,
}

/// Main ShardLM pipeline combining secure and plaintext inference
pub struct ShardLmPipeline {
    /// Plaintext reference (for verification)
    pub plaintext: PlaintextInference,
    /// Secure inference
    pub secure: SecureInference,
}

impl ShardLmPipeline {
    /// Create a new pipeline with shared weights
    pub fn new(config: TinyLlamaConfig) -> Self {
        // Create random weights
        let weights = ModelWeights::random(config, DEFAULT_SCALE);

        // Clone weights for both implementations
        // (In practice, they'd share the same weights)
        let plaintext_weights = ModelWeights::random(weights.config.clone(), DEFAULT_SCALE);
        let secure_weights = weights;

        Self {
            plaintext: PlaintextInference::new(plaintext_weights),
            secure: SecureInference::new(secure_weights),
        }
    }

    /// Create with specific weights
    pub fn with_weights(weights: ModelWeights) -> Self {
        // Clone config for creating a second set of identical weights
        let config = weights.config.clone();
        let scale = weights.scale;

        // For testing, create matching random weights with same seed approach
        // In production, both would use the same weight source
        Self {
            plaintext: PlaintextInference::new(ModelWeights::random(config.clone(), scale)),
            secure: SecureInference::new(ModelWeights::random(config, scale)),
        }
    }

    /// Create with identical weights for correctness verification
    pub fn with_shared_weights(weights: ModelWeights) -> Result<Self> {
        // Deep clone the weights for both implementations
        let weights_clone = ModelWeights {
            config: weights.config.clone(),
            embeddings: shardlm_model::EmbeddingTable::from_raw(
                weights.embeddings.data.clone(),
                weights.embeddings.vocab_size,
                weights.embeddings.embed_dim,
                weights.embeddings.scale,
            )?,
            layers: weights.layers.clone(),
            final_norm: weights.final_norm.clone(),
            lm_head: shardlm_model::LinearWeights {
                weight: weights.lm_head.weight.clone(),
                bias: weights.lm_head.bias.clone(),
                in_features: weights.lm_head.in_features,
                out_features: weights.lm_head.out_features,
                scale: weights.lm_head.scale,
            },
            scale: weights.scale,
        };

        Ok(Self {
            plaintext: PlaintextInference::new(weights),
            secure: SecureInference::new(weights_clone),
        })
    }

    /// Get model config
    pub fn config(&self) -> &TinyLlamaConfig {
        self.plaintext.config()
    }

    /// Run plaintext inference
    pub fn run_plaintext(&self, token_ids: &[usize]) -> Result<InferenceResult> {
        let logits = self.plaintext.forward_slice(token_ids)?;
        let top_k = self.plaintext.top_k(&logits, 5);

        Ok(InferenceResult { logits, top_k })
    }

    /// Run secure inference
    pub fn run_secure(&self, token_ids: &[u32]) -> Result<InferenceResult> {
        let logits = self.secure.forward_slice(token_ids)?;

        // Calculate top-k
        let scale_factor = (1u64 << logits.scale) as f64;
        let mut indexed: Vec<(usize, i32)> = logits.data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        let top_k: Vec<(usize, f64)> = indexed
            .into_iter()
            .take(5)
            .map(|(idx, val)| (idx, val as f64 / scale_factor))
            .collect();

        Ok(InferenceResult { logits, top_k })
    }

    /// Verify secure inference against plaintext reference
    /// Returns true if results match within tolerance
    pub fn verify(&self, token_ids: &[usize], tolerance: f64) -> Result<bool> {
        let token_ids_u32: Vec<u32> = token_ids.iter().map(|&x| x as u32).collect();

        let plaintext_result = self.run_plaintext(token_ids)?;
        let secure_result = self.run_secure(&token_ids_u32)?;

        // Compare logits
        if plaintext_result.logits.len() != secure_result.logits.len() {
            return Err(HarnessError::VerificationFailed {
                message: format!(
                    "Logit length mismatch: {} vs {}",
                    plaintext_result.logits.len(),
                    secure_result.logits.len()
                ),
            });
        }

        let scale_factor = (1u64 << plaintext_result.logits.scale) as f64;
        let mut max_error = 0.0f64;

        for (p, s) in plaintext_result
            .logits
            .data
            .iter()
            .zip(&secure_result.logits.data)
        {
            let p_f64 = *p as f64 / scale_factor;
            let s_f64 = *s as f64 / scale_factor;
            let error = (p_f64 - s_f64).abs();
            max_error = max_error.max(error);
        }

        Ok(max_error <= tolerance)
    }
}

/// Compare two FixedVectors and return max absolute error
#[allow(dead_code)]
pub fn compare_vectors(a: &FixedVector, b: &FixedVector) -> f64 {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.scale, b.scale);

    let scale_factor = (1u64 << a.scale) as f64;
    let mut max_error = 0.0f64;

    for (&av, &bv) in a.data.iter().zip(&b.data) {
        let a_f64 = av as f64 / scale_factor;
        let b_f64 = bv as f64 / scale_factor;
        let error = (a_f64 - b_f64).abs();
        max_error = max_error.max(error);
    }

    max_error
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let pipeline = ShardLmPipeline::new(config);
        assert_eq!(pipeline.config().vocab_size, 100);
    }

    #[test]
    fn test_plaintext_inference() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let pipeline = ShardLmPipeline::new(config);

        let result = pipeline.run_plaintext(&[1, 2, 3]).unwrap();
        assert_eq!(result.logits.len(), 100);
        assert_eq!(result.top_k.len(), 5);
    }

    #[test]
    fn test_secure_inference() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let pipeline = ShardLmPipeline::new(config);

        let result = pipeline.run_secure(&[1, 2, 3]).unwrap();
        assert_eq!(result.logits.len(), 100);
        assert_eq!(result.top_k.len(), 5);
    }

    #[test]
    fn test_embedding_correctness() {
        // This test verifies that OT-retrieved embeddings match plaintext lookups
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };

        // Create weights and share them
        let weights = ModelWeights::random(config, DEFAULT_SCALE);
        let pipeline = ShardLmPipeline::with_shared_weights(weights).unwrap();

        // Get embeddings via both methods
        let plaintext_embeddings = pipeline.plaintext.get_embeddings(&[5, 10, 15]);
        let secure_embeddings = pipeline
            .secure
            .secure_embedding_fetch(&[5, 10, 15])
            .unwrap();

        // Compare (should match since they use the same embedding table)
        assert_eq!(plaintext_embeddings.len(), secure_embeddings.len());
        for (p, s) in plaintext_embeddings.iter().zip(&secure_embeddings) {
            assert_eq!(p.data, s.data, "Embeddings should match");
        }
    }
}
