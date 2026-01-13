//! Private transformer layer and decode loop
//!
//! This module coordinates all components for private autoregressive inference:
//! - QKV projections (secret-shared)
//! - Attention with client-side softmax (Option A)
//! - FFN with client-side SiLU activation
//! - KV cache management
//!
//! ## Privacy Guarantees
//!
//! The server never learns:
//! - Input tokens (protected by private embedding via OT)
//! - Attention patterns (client computes softmax)
//! - Intermediate activations (client computes nonlinear functions)
//! - Output tokens (client samples from logits)
//!
//! ## Decode Loop Architecture
//!
//! ```text
//! Client                          Server
//!   |                               |
//!   | ---(private embedding OT)---> |
//!   |                               |
//!   | <----(embedding shares)------ |
//!   |                               |
//!   | For each layer:               |
//!   |   |---(hidden shares)-------->|
//!   |   |                           |
//!   |   |<--(QKV products)----------|
//!   |   |                           |
//!   |   | [client: reconstruct,     |
//!   |   |  compute softmax,         |
//!   |   |  reshare weights]         |
//!   |   |                           |
//!   |   |---(weight shares)-------->|
//!   |   |                           |
//!   |   |<--(context shares)--------|
//!   |   |                           |
//!   |   | [FFN with client SiLU]    |
//!   |                               |
//!   | <----(logit shares)---------- |
//!   |                               |
//!   | [client: reconstruct logits,  |
//!   |  sample next token]           |
//! ```

use crate::error::Result;
use crate::share::Share;
use crate::matrix::SharedMatrix;
use crate::kv_cache::{KvCacheClient, KvCacheServer, reconstruct_kv};
use crate::attention::{AttentionClient, AttentionServer, compute_attention};
use crate::projection::{ProjectionClient, ProjectionServer, compute_qkv_projection, compute_output_projection};
use crate::ffn::{FfnClient, FfnServer, compute_ffn};
use crate::rmsnorm::{RmsNormClient, RmsNormConfig};

/// Client-side transformer layer
pub struct TransformerLayerClient {
    /// Input layernorm (before attention)
    input_norm: RmsNormClient,
    /// Post-attention layernorm (before FFN)
    post_attn_norm: RmsNormClient,
    /// Attention module
    attention: AttentionClient,
    /// QKV projection module
    projection: ProjectionClient,
    /// FFN module
    ffn: FfnClient,
    /// Layer index
    layer_idx: usize,
    /// Fixed-point scale
    scale: u8,
}

/// Server-side transformer layer
pub struct TransformerLayerServer {
    /// Attention module
    attention: AttentionServer,
    /// QKV projection module
    projection: ProjectionServer,
    /// FFN module
    ffn: FfnServer,
    /// Layer index
    #[allow(dead_code)]
    layer_idx: usize,
    /// Fixed-point scale
    #[allow(dead_code)]
    scale: u8,
}

/// Configuration for a transformer layer
#[derive(Clone)]
pub struct TransformerLayerConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
    /// Layer index
    pub layer_idx: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// Fixed-point scale
    pub scale: u8,
}

/// Result of a transformer layer forward pass
#[derive(Debug)]
pub struct TransformerLayerOutput {
    /// Client's output share [hidden_size]
    pub client: Share,
    /// Server's output share
    pub server: Share,
}

impl TransformerLayerClient {
    /// Create transformer layer client with unit norm weights (for testing)
    pub fn new(config: &TransformerLayerConfig) -> Self {
        let norm_config = RmsNormConfig {
            hidden_size: config.hidden_size,
            eps: config.rms_norm_eps,
            scale: config.scale,
        };

        Self {
            input_norm: RmsNormClient::with_unit_weights(&norm_config),
            post_attn_norm: RmsNormClient::with_unit_weights(&norm_config),
            attention: AttentionClient::new(
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.scale,
            ),
            projection: ProjectionClient::new(
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.scale,
            ),
            ffn: FfnClient::new(
                config.hidden_size,
                config.intermediate_size,
                config.scale,
            ),
            layer_idx: config.layer_idx,
            scale: config.scale,
        }
    }

    /// Create transformer layer client with custom norm weights
    pub fn with_norm_weights(
        config: &TransformerLayerConfig,
        input_norm_weight: Vec<f64>,
        post_attn_norm_weight: Vec<f64>,
    ) -> Result<Self> {
        let norm_config = RmsNormConfig {
            hidden_size: config.hidden_size,
            eps: config.rms_norm_eps,
            scale: config.scale,
        };

        Ok(Self {
            input_norm: RmsNormClient::new(&norm_config, input_norm_weight)?,
            post_attn_norm: RmsNormClient::new(&norm_config, post_attn_norm_weight)?,
            attention: AttentionClient::new(
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.scale,
            ),
            projection: ProjectionClient::new(
                config.hidden_size,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.scale,
            ),
            ffn: FfnClient::new(
                config.hidden_size,
                config.intermediate_size,
                config.scale,
            ),
            layer_idx: config.layer_idx,
            scale: config.scale,
        })
    }

    /// Forward pass for a single token
    ///
    /// Takes hidden state shares, returns updated hidden state shares.
    /// Both client and server KV caches are updated in sync.
    pub fn forward(
        &self,
        hidden_client: &Share,
        hidden_server: &Share,
        kv_client: &mut KvCacheClient,
        kv_server: &mut KvCacheServer,
        server: &TransformerLayerServer,
        _position: usize,
    ) -> Result<TransformerLayerOutput> {
        // Step 1: Apply input RMSNorm (client-side, reconstructs and reshares)
        let (normed_client, normed_server) = self.input_norm.forward(hidden_client, hidden_server)?;

        // Step 2: Compute QKV projections on normalized input
        let normed = reconstruct_kv(&normed_client, &normed_server)?;
        let qkv = compute_qkv_projection(&self.projection, &server.projection, &normed)?;

        // Step 3: Store K, V in BOTH caches (fix: server cache was not updated)
        kv_client.append(self.layer_idx, qkv.k_client.clone(), qkv.v_client.clone())?;
        kv_server.append(self.layer_idx, qkv.k_server.clone(), qkv.v_server.clone())?;

        // Step 4: Get cached K, V for attention
        let k_client = kv_client.get_k(self.layer_idx)
            .ok_or_else(|| crate::error::SharingError::DimensionMismatch {
                expected: 1,
                got: 0,
            })?;
        let k_server = kv_server.get_k(self.layer_idx)
            .ok_or_else(|| crate::error::SharingError::DimensionMismatch {
                expected: 1,
                got: 0,
            })?;
        let v_client = kv_client.get_v(self.layer_idx)
            .ok_or_else(|| crate::error::SharingError::DimensionMismatch {
                expected: 1,
                got: 0,
            })?;
        let v_server = kv_server.get_v(self.layer_idx)
            .ok_or_else(|| crate::error::SharingError::DimensionMismatch {
                expected: 1,
                got: 0,
            })?;

        // Step 5: Compute attention
        let attention_out = compute_attention(
            &self.attention,
            &server.attention,
            &qkv.q_client,
            &qkv.q_server,
            k_client,
            k_server,
            v_client,
            v_server,
        )?;

        // Step 6: Output projection
        let attn_output = compute_output_projection(
            &self.projection,
            &server.projection,
            &attention_out.client_share,
            &attention_out.server_share,
        )?;

        // Step 7: Residual connection (hidden + attn_output)
        let (post_attn_client, post_attn_server) = self.add_residual_shares(
            hidden_client,
            hidden_server,
            &attn_output.client,
            &attn_output.server,
        )?;

        // Step 8: Apply post-attention RMSNorm before FFN
        let (normed_ffn_client, normed_ffn_server) = self.post_attn_norm.forward(
            &post_attn_client,
            &post_attn_server,
        )?;

        // Step 9: FFN
        let ffn_out = compute_ffn(
            &self.ffn,
            &server.ffn,
            &normed_ffn_client,
            &normed_ffn_server,
        )?;

        // Step 10: Residual connection (post_attn + ffn_out)
        let (final_client, final_server) = self.add_residual_shares(
            &post_attn_client,
            &post_attn_server,
            &ffn_out.client,
            &ffn_out.server,
        )?;

        Ok(TransformerLayerOutput {
            client: final_client,
            server: final_server,
        })
    }

    /// Add two sets of shares together
    fn add_residual_shares(
        &self,
        a_client: &Share,
        a_server: &Share,
        b_client: &Share,
        b_server: &Share,
    ) -> Result<(Share, Share)> {
        // Can add shares directly: (a_c + b_c, a_s + b_s) reconstructs to a + b
        let result_client: Vec<i32> = a_client
            .data
            .iter()
            .zip(&b_client.data)
            .map(|(&a, &b)| a.wrapping_add(b))
            .collect();
        let result_server: Vec<i32> = a_server
            .data
            .iter()
            .zip(&b_server.data)
            .map(|(&a, &b)| a.wrapping_add(b))
            .collect();

        Ok((
            Share::from_raw(result_client, self.scale),
            Share::from_raw(result_server, self.scale),
        ))
    }
}

impl TransformerLayerServer {
    /// Create transformer layer server with weights
    pub fn new(
        config: &TransformerLayerConfig,
        w_q: SharedMatrix,
        w_k: SharedMatrix,
        w_v: SharedMatrix,
        w_o: SharedMatrix,
        w_gate: SharedMatrix,
        w_up: SharedMatrix,
        w_down: SharedMatrix,
    ) -> Self {
        Self {
            attention: AttentionServer::new(
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.scale,
            ),
            projection: ProjectionServer::new(
                w_q,
                w_k,
                w_v,
                w_o,
                config.num_heads,
                config.num_kv_heads,
                config.head_dim,
                config.scale,
            ),
            ffn: FfnServer::new(w_gate, w_up, w_down, config.scale),
            layer_idx: config.layer_idx,
            scale: config.scale,
        }
    }
}

/// Private decode loop for autoregressive generation
///
/// This struct manages the complete private inference pipeline:
/// - All transformer layers (client and server sides)
/// - KV caches for both parties
/// - Token generation loop
pub struct PrivateDecoder {
    /// Client-side layers
    client_layers: Vec<TransformerLayerClient>,
    /// Number of layers
    num_layers: usize,
    /// Hidden dimension
    hidden_size: usize,
    /// Vocabulary size
    vocab_size: usize,
    /// Fixed-point scale
    scale: u8,
}

impl PrivateDecoder {
    /// Create a new private decoder
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_size: usize,
        vocab_size: usize,
        rms_norm_eps: f64,
        scale: u8,
    ) -> Self {
        let client_layers: Vec<_> = (0..num_layers)
            .map(|i| {
                let config = TransformerLayerConfig {
                    hidden_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_size,
                    layer_idx: i,
                    rms_norm_eps,
                    scale,
                };
                TransformerLayerClient::new(&config)
            })
            .collect();

        Self {
            client_layers,
            num_layers,
            hidden_size,
            vocab_size,
            scale,
        }
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get a client layer
    pub fn get_client_layer(&self, idx: usize) -> Option<&TransformerLayerClient> {
        self.client_layers.get(idx)
    }

    /// Decode one step: hidden -> logits
    ///
    /// Takes the hidden state from embedding, runs through all layers,
    /// returns logit shares that client can reconstruct and sample from.
    pub fn decode_step(
        &self,
        hidden_client: &Share,
        hidden_server: &Share,
        kv_client: &mut KvCacheClient,
        kv_server: &mut KvCacheServer,
        server_layers: &[TransformerLayerServer],
        lm_head: &SharedMatrix,
        position: usize,
    ) -> Result<(Share, Share)> {
        // Run through all layers
        let mut current_client = hidden_client.clone();
        let mut current_server = hidden_server.clone();

        for (_layer_idx, (client_layer, server_layer)) in
            self.client_layers.iter().zip(server_layers).enumerate()
        {
            let output = client_layer.forward(
                &current_client,
                &current_server,
                kv_client,
                kv_server,
                server_layer,
                position,
            )?;

            current_client = output.client;
            current_server = output.server;
        }

        // Apply LM head to get logits
        let logits = self.apply_lm_head(lm_head, &current_client, &current_server)?;

        Ok(logits)
    }

    /// Apply LM head: hidden -> logits
    fn apply_lm_head(
        &self,
        lm_head: &SharedMatrix,
        hidden_client: &Share,
        hidden_server: &Share,
    ) -> Result<(Share, Share)> {
        // Compute logits = hidden @ lm_head
        // This is a linear layer, so we can use matmul on shares

        // Client contribution
        let client_contrib = self.matmul_share(lm_head, hidden_client)?;

        // Server contribution
        let server_contrib = self.matmul_share(lm_head, hidden_server)?;

        // The logits are already in secret-shared form:
        // logits = hidden @ W = (h_c + h_s) @ W = h_c @ W + h_s @ W
        Ok((client_contrib, server_contrib))
    }

    fn matmul_share(&self, weight: &SharedMatrix, input: &Share) -> Result<Share> {
        if weight.rows != input.len() {
            return Err(crate::error::SharingError::DimensionMismatch {
                expected: weight.rows,
                got: input.len(),
            });
        }

        let mut output = vec![0i64; weight.cols];
        for i in 0..weight.rows {
            let x = input.data[i] as i64;
            for j in 0..weight.cols {
                output[j] += x * weight.data[i * weight.cols + j] as i64;
            }
        }

        let result: Vec<i32> = output.iter().map(|&x| (x >> self.scale) as i32).collect();
        Ok(Share::from_raw(result, self.scale))
    }

    /// Sample next token from logit shares (client-side)
    ///
    /// Client reconstructs logits, applies temperature, samples token.
    /// Server never sees the logits or sampled token.
    pub fn sample_token(
        logit_client: &Share,
        logit_server: &Share,
        temperature: f64,
    ) -> Result<(usize, f64)> {
        // Reconstruct logits
        let logits = reconstruct_kv(logit_client, logit_server)?;
        let scale_factor = (1u64 << logits.scale) as f64;

        // Convert to f64
        let logits_f64: Vec<f64> = logits
            .data
            .iter()
            .map(|&x| x as f64 / scale_factor)
            .collect();

        // Apply temperature and softmax
        let max_logit = logits_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits: Vec<f64> = logits_f64
            .iter()
            .map(|&l| ((l - max_logit) / temperature).exp())
            .collect();
        let sum: f64 = exp_logits.iter().sum();
        let probs: Vec<f64> = exp_logits.iter().map(|&e| e / sum).collect();

        // Sample (deterministic for now: argmax)
        let (token_id, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        Ok((token_id, prob))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::{DEFAULT_SCALE, FixedVector};
    use crate::kv_cache::share_kv;

    #[allow(dead_code)]
    fn create_test_matrix(rows: usize, cols: usize, scale: u8) -> SharedMatrix {
        let mut data = vec![0i32; rows * cols];
        let one = 1i32 << (scale / 2); // Smaller values to avoid overflow
        for i in 0..rows.min(cols) {
            data[i * cols + i] = one;
        }
        SharedMatrix::from_raw(data, rows, cols, scale).unwrap()
    }

    #[test]
    fn test_private_decoder_creation() {
        let decoder = PrivateDecoder::new(
            2,      // num_layers
            8,      // hidden_size
            2,      // num_heads
            2,      // num_kv_heads
            4,      // head_dim
            16,     // intermediate_size
            100,    // vocab_size
            1e-5,   // rms_norm_eps
            DEFAULT_SCALE,
        );

        assert_eq!(decoder.num_layers(), 2);
        assert_eq!(decoder.hidden_size(), 8);
        assert_eq!(decoder.vocab_size(), 100);
    }

    #[test]
    fn test_sample_token() {
        // Create some logits
        let logits = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 0.5]).unwrap();
        let (client, server) = share_kv(&logits);

        let (token_id, prob) = PrivateDecoder::sample_token(&client, &server, 1.0).unwrap();

        // Token 2 has highest logit (3.0), so should be selected
        assert_eq!(token_id, 2);
        assert!(prob > 0.5); // Should have highest probability
    }

    #[test]
    fn test_transformer_layer_config() {
        let config = TransformerLayerConfig {
            hidden_size: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            intermediate_size: 16,
            layer_idx: 0,
            rms_norm_eps: 1e-5,
            scale: DEFAULT_SCALE,
        };

        let client = TransformerLayerClient::new(&config);
        assert_eq!(client.layer_idx, 0);
    }

    #[test]
    fn test_add_residual_shares() {
        let config = TransformerLayerConfig {
            hidden_size: 4,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 2,
            intermediate_size: 8,
            layer_idx: 0,
            rms_norm_eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let client = TransformerLayerClient::new(&config);

        let a = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = FixedVector::from_f64_slice_default(&[0.5, 0.5, 0.5, 0.5]).unwrap();

        let (a_c, a_s) = share_kv(&a);
        let (b_c, b_s) = share_kv(&b);

        let (result_c, result_s) = client.add_residual_shares(&a_c, &a_s, &b_c, &b_s).unwrap();

        let result = reconstruct_kv(&result_c, &result_s).unwrap();

        // Result should be a + b
        let expected: Vec<i32> = a.data.iter().zip(&b.data).map(|(&a, &b)| a + b).collect();
        assert_eq!(result.data, expected);
    }
}
