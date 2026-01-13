//! Plaintext reference implementation
//!
//! This module provides a non-secure reference implementation that can be used
//! to verify the correctness of the secure computation.

use shardlm_fixed_point::{FixedVector, DEFAULT_SCALE};
use shardlm_model::{ModelWeights, TinyLlamaConfig};

use crate::error::Result;

/// Plaintext inference for reference/verification
pub struct PlaintextInference {
    /// Model weights
    weights: ModelWeights,
}

impl PlaintextInference {
    /// Create with given weights
    pub fn new(weights: ModelWeights) -> Self {
        Self { weights }
    }

    /// Create with random weights (for testing)
    pub fn random(config: TinyLlamaConfig) -> Self {
        Self {
            weights: ModelWeights::random(config, DEFAULT_SCALE),
        }
    }

    /// Get model config
    pub fn config(&self) -> &TinyLlamaConfig {
        &self.weights.config
    }

    /// Get model weights
    pub fn weights(&self) -> &ModelWeights {
        &self.weights
    }

    /// Look up embeddings for token IDs
    pub fn get_embeddings(&self, token_ids: &[usize]) -> Vec<FixedVector> {
        self.weights.embeddings.get_batch(token_ids)
    }

    /// Apply RMS normalization (simplified)
    pub fn rms_norm(&self, x: &FixedVector, weights: &[i32]) -> FixedVector {
        // Compute RMS: sqrt(mean(x^2))
        let sum_sq: i64 = x.data.iter().map(|&v| (v as i64) * (v as i64)).sum();
        let mean_sq = sum_sq / x.len() as i64;
        let rms = ((mean_sq >> self.weights.scale) as f64).sqrt();

        // Normalize and apply weights
        let scale_factor = (1u64 << self.weights.scale) as f64;
        let data: Vec<i32> = x
            .data
            .iter()
            .zip(weights)
            .map(|(&xi, &wi)| {
                let normalized = (xi as f64) / (rms * scale_factor);
                let weighted = normalized * (wi as f64);
                weighted.round() as i32
            })
            .collect();

        FixedVector::from_raw(data, x.scale)
    }

    /// Apply a linear layer: Y = XW + b
    pub fn linear(&self, x: &FixedVector, w: &[i32], b: Option<&[i32]>, out_dim: usize) -> FixedVector {
        let in_dim = x.len();
        let mut output = vec![0i64; out_dim];

        // Matrix-vector multiply
        for i in 0..in_dim {
            let xi = x.data[i] as i64;
            for j in 0..out_dim {
                output[j] += xi * (w[i * out_dim + j] as i64);
            }
        }

        // Rescale and add bias
        let mut result: Vec<i32> = output
            .iter()
            .map(|&v| (v >> self.weights.scale) as i32)
            .collect();

        if let Some(bias) = b {
            for (r, &b) in result.iter_mut().zip(bias) {
                *r = r.wrapping_add(b);
            }
        }

        FixedVector::from_raw(result, x.scale)
    }

    /// Compute logits from hidden state using LM head
    pub fn compute_logits(&self, hidden: &FixedVector) -> FixedVector {
        self.linear(
            hidden,
            &self.weights.lm_head.weight,
            self.weights.lm_head.bias.as_deref(),
            self.weights.lm_head.out_features,
        )
    }

    /// Run a simplified forward pass (embedding + one linear layer + logits)
    /// This is the "TinyLlama forward-pass slice" for v1
    pub fn forward_slice(&self, token_ids: &[usize]) -> Result<FixedVector> {
        if token_ids.is_empty() {
            return Err(crate::error::HarnessError::InvalidInput(
                "Empty token sequence".into(),
            ));
        }

        // Step 1: Get embeddings
        let embeddings = self.get_embeddings(token_ids);
        if embeddings.is_empty() {
            return Err(crate::error::HarnessError::InvalidInput(
                "Invalid token IDs".into(),
            ));
        }

        // For simplicity, use the last token's embedding
        let last_embedding = embeddings.last().unwrap().clone();

        // Get first layer's weights
        let layer0 = &self.weights.layers[0];

        // Step 2: Apply input layernorm
        let normed = self.rms_norm(&last_embedding, &layer0.input_layernorm.weight);

        // Step 3: Apply Q projection as example linear layer
        let q = self.linear(
            &normed,
            &layer0.self_attn.q_proj.weight,
            layer0.self_attn.q_proj.bias.as_deref(),
            layer0.self_attn.q_proj.out_features,
        );

        // Step 4: Project back to hidden size via o_proj
        let hidden = self.linear(
            &q,
            &layer0.self_attn.o_proj.weight,
            layer0.self_attn.o_proj.bias.as_deref(),
            layer0.self_attn.o_proj.out_features,
        );

        // Step 5: Compute logits
        let logits = self.compute_logits(&hidden);

        Ok(logits)
    }

    /// Get top-k token predictions from logits
    pub fn top_k(&self, logits: &FixedVector, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, i32)> = logits.data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));

        let scale_factor = (1u64 << logits.scale) as f64;
        indexed
            .into_iter()
            .take(k)
            .map(|(idx, val)| (idx, val as f64 / scale_factor))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plaintext_forward() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let inference = PlaintextInference::random(config);

        let token_ids = vec![1, 2, 3];
        let logits = inference.forward_slice(&token_ids).unwrap();

        assert_eq!(logits.len(), 100); // vocab_size
    }

    #[test]
    fn test_get_embeddings() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            ..Default::default()
        };
        let inference = PlaintextInference::random(config);

        let embeddings = inference.get_embeddings(&[0, 1, 2]);
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 64);
    }

    #[test]
    fn test_top_k() {
        let config = TinyLlamaConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            ..Default::default()
        };
        let inference = PlaintextInference::random(config);

        let token_ids = vec![0];
        let logits = inference.forward_slice(&token_ids).unwrap();
        let top = inference.top_k(&logits, 3);

        assert_eq!(top.len(), 3);
    }
}
