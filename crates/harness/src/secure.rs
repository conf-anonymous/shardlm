//! Secure inference implementation
//!
//! Uses secret sharing for computation where the server never sees plaintext values.

use shardlm_fixed_point::{FixedVector, DEFAULT_SCALE};
use shardlm_model::{ModelWeights, TinyLlamaConfig};
use shardlm_ot::{OtReceiver, OtSender, OtSessionConfig, SimpleOtExtension};
use shardlm_sharing::{SharedMatrix, SharedVector};

use crate::error::{HarnessError, Result};

/// Secure inference using secret sharing
pub struct SecureInference {
    /// Model weights (held by server)
    weights: ModelWeights,
}

impl SecureInference {
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

    /// Run secure embedding retrieval using OT
    /// Returns embeddings that only the client knows
    pub fn secure_embedding_fetch(&self, token_ids: &[u32]) -> Result<Vec<FixedVector>> {
        let config = OtSessionConfig {
            vocab_size: self.weights.config.vocab_size as u32,
            hidden_dim: self.weights.config.hidden_size as u16,
            max_prompt_len: 64,
            scale: self.weights.scale,
            ..Default::default()
        };

        // Create sender (server) and receiver (client)
        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Set embedding database on server
        sender.set_embedding_db(self.weights.embeddings.to_bytes());

        // Session initialization
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce)?;
        receiver.handle_session_params(&params_payload, params_header.session_id)?;

        // Base OT (simplified)
        let (base_header, base_msg) = receiver.generate_base_ot_init()?;
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce)?
        {
            receiver.handle_base_ot_response(&response_msg)?;
        }

        // Mark ready
        let (_, ready_payload) = sender.generate_session_ready([0u8; 8])?;
        receiver.handle_session_ready(&ready_payload)?;

        // Fetch embeddings
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(token_ids)?;
        let (_, fetch_response) = sender.handle_embed_fetch(
            &fetch_request,
            fetch_header.ctr,
            fetch_header.client_nonce,
        )?;

        let embedding_bytes =
            receiver.handle_embed_fetch_response(&fetch_response, fetch_header.ctr)?;

        // Convert bytes to FixedVectors
        let row_bytes = self.weights.config.hidden_size * 4;
        let embeddings: Vec<FixedVector> = embedding_bytes
            .chunks_exact(row_bytes)
            .map(|chunk| {
                let data: Vec<i32> = chunk
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                FixedVector::from_raw(data, self.weights.scale)
            })
            .collect();

        Ok(embeddings)
    }

    /// Run secure linear layer: Y = XW + b
    /// X is secret-shared, W and b are known to server
    pub fn secure_linear(
        &self,
        x: &FixedVector,
        w: &SharedMatrix,
        b: Option<&FixedVector>,
    ) -> Result<FixedVector> {
        // Client creates shares: X = X_c + X_s
        let shared_x = SharedVector::from_plaintext(x);

        // Compute Y = X * W using secret sharing
        let shared_y = w.multiply_shared(&shared_x)?;

        // Add bias if present
        let shared_result = if let Some(bias) = b {
            SharedMatrix::add_bias(&shared_y, bias)?
        } else {
            shared_y
        };

        // Client reconstructs result
        let result = shared_result.reconstruct()?;
        Ok(result)
    }

    /// Run a simplified secure forward pass
    pub fn forward_slice(&self, token_ids: &[u32]) -> Result<FixedVector> {
        if token_ids.is_empty() {
            return Err(HarnessError::InvalidInput("Empty token sequence".into()));
        }

        // Step 1: Secure embedding retrieval
        let embeddings = self.secure_embedding_fetch(token_ids)?;
        if embeddings.is_empty() {
            return Err(HarnessError::InvalidInput("No embeddings retrieved".into()));
        }

        // Use last embedding
        let last_embedding = embeddings.last().unwrap();

        // Get first layer's attention weights
        let layer0 = &self.weights.layers[0];

        // Step 2: Apply Q projection using secure linear
        let q_matrix = layer0.self_attn.q_proj.to_shared_matrix();
        let q = self.secure_linear(last_embedding, &q_matrix, layer0.self_attn.q_proj.get_bias().as_ref())?;

        // Step 3: Apply O projection
        let o_matrix = layer0.self_attn.o_proj.to_shared_matrix();
        let hidden = self.secure_linear(&q, &o_matrix, layer0.self_attn.o_proj.get_bias().as_ref())?;

        // Step 4: Compute logits
        let lm_head_matrix = self.weights.lm_head.to_shared_matrix();
        let logits = self.secure_linear(&hidden, &lm_head_matrix, self.weights.lm_head.get_bias().as_ref())?;

        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_embedding_fetch() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            ..Default::default()
        };
        let inference = SecureInference::random(config);

        let embeddings = inference.secure_embedding_fetch(&[5, 10, 15]).unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 64);
    }

    #[test]
    fn test_secure_linear() {
        let config = TinyLlamaConfig {
            vocab_size: 10,
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            ..Default::default()
        };
        let inference = SecureInference::random(config);

        // Create test input
        let x = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();

        // Create small matrix
        let w_data: Vec<i32> = (0..8 * 4).map(|_| rand::random::<i32>() % 1000).collect();
        let w = SharedMatrix::from_raw(w_data, 8, 4, DEFAULT_SCALE).unwrap();

        let result = inference.secure_linear(&x, &w, None).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_secure_forward() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            ..Default::default()
        };
        let inference = SecureInference::random(config);

        let logits = inference.forward_slice(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), 100);
    }
}
