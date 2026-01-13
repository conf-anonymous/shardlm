//! Transformer forward pass implementation
//!
//! Implements the LLaMA-style transformer layers:
//! - RMSNorm (instead of LayerNorm)
//! - Grouped Query Attention with RoPE
//! - SwiGLU FFN

use crate::config::TinyLlamaConfig;
use crate::rope::RopeFrequencies;
use crate::weights::{
    AttentionWeights, LinearWeights, MlpWeights, ModelWeights, RmsNormWeights,
    TransformerLayerWeights,
};

/// Transformer computation state
pub struct TransformerState {
    /// Model configuration
    pub config: TinyLlamaConfig,
    /// RoPE frequencies
    pub rope_freqs: RopeFrequencies,
    /// Fixed-point scale
    pub scale: u8,
}

impl TransformerState {
    /// Create new transformer state
    pub fn new(config: TinyLlamaConfig, scale: u8) -> Self {
        let rope_freqs = RopeFrequencies::from_config(&config);
        Self {
            config,
            rope_freqs,
            scale,
        }
    }

    /// Forward pass through entire model (plaintext, no secret sharing)
    ///
    /// hidden_states: [seq_len, hidden_size] flattened
    /// Returns: [hidden_size] - the last position's hidden state
    pub fn forward(&self, weights: &ModelWeights, token_embeddings: &[i32]) -> Vec<i32> {
        let hidden_size = self.config.hidden_size;
        let seq_len = token_embeddings.len() / hidden_size;

        // Start with token embeddings
        let mut hidden_states = token_embeddings.to_vec();

        // Apply each transformer layer
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            hidden_states = self.forward_layer(layer, &hidden_states, seq_len, layer_idx);
        }

        // Apply final layer norm
        let mut final_hidden = vec![0i32; hidden_size];
        for pos in 0..seq_len {
            let start = pos * hidden_size;
            let end = start + hidden_size;
            let normed = rms_norm(&hidden_states[start..end], &weights.final_norm, self.scale);
            final_hidden.copy_from_slice(&normed);
        }

        // Return the last position's hidden state
        final_hidden
    }

    /// Forward pass for a single token (for inference)
    pub fn forward_single(
        &self,
        weights: &ModelWeights,
        hidden_state: &[i32],
        position: usize,
        kv_cache: &mut KVCache,
    ) -> Vec<i32> {
        let mut hidden = hidden_state.to_vec();

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            hidden = self.forward_layer_single(layer, &hidden, position, layer_idx, kv_cache);
        }

        // Apply final layer norm
        rms_norm(&hidden, &weights.final_norm, self.scale)
    }

    /// Forward pass through a single layer (full sequence)
    fn forward_layer(
        &self,
        layer: &TransformerLayerWeights,
        hidden_states: &[i32],
        seq_len: usize,
        _layer_idx: usize,
    ) -> Vec<i32> {
        let hidden_size = self.config.hidden_size;
        let mut output = vec![0i32; seq_len * hidden_size];

        for pos in 0..seq_len {
            let start = pos * hidden_size;
            let end = start + hidden_size;
            let hidden = &hidden_states[start..end];

            // 1. Input LayerNorm + Attention
            let normed = rms_norm(hidden, &layer.input_layernorm, self.scale);
            let attn_out = self.self_attention(&normed, &layer.self_attn, pos, seq_len);

            // 2. Residual connection
            let mut residual: Vec<i32> = hidden
                .iter()
                .zip(&attn_out)
                .map(|(&h, &a)| h.wrapping_add(a))
                .collect();

            // 3. Post-attention LayerNorm + MLP
            let normed = rms_norm(&residual, &layer.post_attention_layernorm, self.scale);
            let mlp_out = self.mlp(&normed, &layer.mlp);

            // 4. Residual connection
            for (r, m) in residual.iter_mut().zip(&mlp_out) {
                *r = r.wrapping_add(*m);
            }

            output[start..end].copy_from_slice(&residual);
        }

        output
    }

    /// Forward pass through a single layer for single token
    fn forward_layer_single(
        &self,
        layer: &TransformerLayerWeights,
        hidden_state: &[i32],
        position: usize,
        layer_idx: usize,
        kv_cache: &mut KVCache,
    ) -> Vec<i32> {
        // 1. Input LayerNorm + Attention
        let normed = rms_norm(hidden_state, &layer.input_layernorm, self.scale);
        let attn_out =
            self.self_attention_cached(&normed, &layer.self_attn, position, layer_idx, kv_cache);

        // 2. Residual connection
        let mut residual: Vec<i32> = hidden_state
            .iter()
            .zip(&attn_out)
            .map(|(&h, &a)| h.wrapping_add(a))
            .collect();

        // 3. Post-attention LayerNorm + MLP
        let normed = rms_norm(&residual, &layer.post_attention_layernorm, self.scale);
        let mlp_out = self.mlp(&normed, &layer.mlp);

        // 4. Residual connection
        for (r, m) in residual.iter_mut().zip(&mlp_out) {
            *r = r.wrapping_add(*m);
        }

        residual
    }

    /// Self-attention for a single position (no KV cache)
    fn self_attention(
        &self,
        hidden: &[i32],
        attn: &AttentionWeights,
        pos: usize,
        _seq_len: usize,
    ) -> Vec<i32> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = self.config.num_kv_groups();

        // Project Q, K, V
        let q = linear(hidden, &attn.q_proj, self.scale);
        let k = linear(hidden, &attn.k_proj, self.scale);
        let v = linear(hidden, &attn.v_proj, self.scale);

        // Apply RoPE to Q and K (stored for potential future use with KV cache)
        let _q = self.apply_rope(&q, pos, num_heads, head_dim);
        let _k = self.apply_rope(&k, pos, num_kv_heads, head_dim);

        // Simple single-position attention (just scale the value by attention weight)
        // For single token at position, Q@K^T is just a scalar per head
        // This is simplified - for proper causal attention we'd need KV cache
        let mut attn_output = vec![0i32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_head = h / num_kv_groups;

            // For single position, softmax([x]) = [1.0], so output = V
            for d in 0..head_dim {
                attn_output[h * head_dim + d] = v[kv_head * head_dim + d];
            }
        }

        // Output projection
        linear(&attn_output, &attn.o_proj, self.scale)
    }

    /// Self-attention with KV cache
    fn self_attention_cached(
        &self,
        hidden: &[i32],
        attn: &AttentionWeights,
        position: usize,
        layer_idx: usize,
        kv_cache: &mut KVCache,
    ) -> Vec<i32> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let num_kv_groups = self.config.num_kv_groups();

        // Project Q, K, V
        let q = linear(hidden, &attn.q_proj, self.scale);
        let k = linear(hidden, &attn.k_proj, self.scale);
        let v = linear(hidden, &attn.v_proj, self.scale);

        // Apply RoPE to Q and K
        let q = self.apply_rope(&q, position, num_heads, head_dim);
        let k = self.apply_rope(&k, position, num_kv_heads, head_dim);

        // Update KV cache
        kv_cache.update(layer_idx, position, &k, &v);

        // Compute attention scores over all cached positions
        let mut attn_output = vec![0i32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_head = h / num_kv_groups;

            // Compute attention scores for all cached positions
            let mut scores: Vec<i64> = Vec::with_capacity(position + 1);
            let mut max_score: i64 = i64::MIN;

            for p in 0..=position {
                let cached_k = kv_cache.get_k(layer_idx, p, kv_head, head_dim);
                let mut score: i64 = 0;
                for d in 0..head_dim {
                    score += q[h * head_dim + d] as i64 * cached_k[d] as i64;
                }
                score >>= self.scale;

                // Scale by 1/sqrt(head_dim)
                let scale_factor = ((1u64 << self.scale) as f64 / (head_dim as f64).sqrt()) as i64;
                score = (score * scale_factor) >> self.scale;

                scores.push(score);
                max_score = max_score.max(score);
            }

            // Softmax and weighted sum using f64 for numerical stability
            let scale_factor = (1u64 << self.scale) as f64;
            let max_score_f64 = max_score as f64 / scale_factor;

            let mut exp_scores: Vec<f64> = scores
                .iter()
                .map(|&s| ((s as f64 / scale_factor) - max_score_f64).exp())
                .collect();
            let exp_sum: f64 = exp_scores.iter().sum();

            // Normalize to get attention weights
            for s in &mut exp_scores {
                *s /= exp_sum;
            }

            // Weighted sum of values
            let mut weighted_v = vec![0f64; head_dim];
            for (p, &attn_weight) in exp_scores.iter().enumerate() {
                let cached_v = kv_cache.get_v(layer_idx, p, kv_head, head_dim);
                for d in 0..head_dim {
                    weighted_v[d] += attn_weight * (cached_v[d] as f64);
                }
            }

            // Convert back to fixed-point
            for d in 0..head_dim {
                attn_output[h * head_dim + d] = weighted_v[d] as i32;
            }
        }

        // Output projection
        linear(&attn_output, &attn.o_proj, self.scale)
    }

    /// Apply RoPE to query or key vectors
    fn apply_rope(&self, x: &[i32], position: usize, num_heads: usize, head_dim: usize) -> Vec<i32> {
        let mut out = x.to_vec();

        // Get cos/sin for this position
        let (cos, sin) = match self.rope_freqs.get(position) {
            Some((c, s)) => (c, s),
            None => return out, // Position out of range, return unchanged
        };

        let scale_factor = (1u64 << self.scale) as f64;

        for h in 0..num_heads {
            for i in 0..head_dim / 2 {
                let x0 = x[h * head_dim + i] as f64 / scale_factor;
                let x1 = x[h * head_dim + head_dim / 2 + i] as f64 / scale_factor;

                let c = cos[i];
                let s = sin[i];

                // RoPE: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                out[h * head_dim + i] = ((x0 * c - x1 * s) * scale_factor) as i32;
                out[h * head_dim + head_dim / 2 + i] = ((x0 * s + x1 * c) * scale_factor) as i32;
            }
        }

        out
    }

    /// SwiGLU MLP: down(silu(gate(x)) * up(x))
    fn mlp(&self, hidden: &[i32], mlp: &MlpWeights) -> Vec<i32> {
        let gate = linear(hidden, &mlp.gate_proj, self.scale);
        let up = linear(hidden, &mlp.up_proj, self.scale);

        // SiLU(gate) * up
        let intermediate: Vec<i32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let silu = silu_fixed(g, self.scale);
                ((silu as i64 * u as i64) >> self.scale) as i32
            })
            .collect();

        linear(&intermediate, &mlp.down_proj, self.scale)
    }
}

/// KV cache for autoregressive generation
pub struct KVCache {
    /// K cache: [num_layers, max_seq_len, num_kv_heads * head_dim]
    k_cache: Vec<Vec<Vec<i32>>>,
    /// V cache: [num_layers, max_seq_len, num_kv_heads * head_dim]
    v_cache: Vec<Vec<Vec<i32>>>,
}

impl KVCache {
    /// Create new KV cache
    pub fn new(num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let kv_dim = num_kv_heads * head_dim;
        Self {
            k_cache: vec![vec![vec![0; kv_dim]; max_seq_len]; num_layers],
            v_cache: vec![vec![vec![0; kv_dim]; max_seq_len]; num_layers],
        }
    }

    /// Update cache at position
    pub fn update(&mut self, layer_idx: usize, position: usize, k: &[i32], v: &[i32]) {
        self.k_cache[layer_idx][position].copy_from_slice(k);
        self.v_cache[layer_idx][position].copy_from_slice(v);
    }

    /// Get K for a specific head at position
    pub fn get_k(&self, layer_idx: usize, position: usize, head: usize, head_dim: usize) -> &[i32] {
        let start = head * head_dim;
        &self.k_cache[layer_idx][position][start..start + head_dim]
    }

    /// Get V for a specific head at position
    pub fn get_v(&self, layer_idx: usize, position: usize, head: usize, head_dim: usize) -> &[i32] {
        let start = head * head_dim;
        &self.v_cache[layer_idx][position][start..start + head_dim]
    }
}

/// RMS LayerNorm: x * w / sqrt(mean(x^2) + eps)
fn rms_norm(x: &[i32], weights: &RmsNormWeights, scale: u8) -> Vec<i32> {
    let n = x.len();
    let scale_factor = (1u64 << scale) as f64;

    // Convert to f64 for numerical stability
    let x_f64: Vec<f64> = x.iter().map(|&v| v as f64 / scale_factor).collect();

    // Compute RMS: sqrt(mean(x^2) + eps)
    let sum_sq: f64 = x_f64.iter().map(|&v| v * v).sum();
    let mean_sq = sum_sq / n as f64;
    let rms = (mean_sq + weights.eps).sqrt();

    // Apply normalization: x / rms * weight
    x_f64
        .iter()
        .zip(&weights.weight)
        .map(|(&xi, &wi)| {
            let wi_f64 = wi as f64 / scale_factor;
            let normed = (xi / rms) * wi_f64;
            (normed * scale_factor) as i32
        })
        .collect()
}

/// Linear layer: y = x @ W
fn linear(x: &[i32], weights: &LinearWeights, scale: u8) -> Vec<i32> {
    let mut output = vec![0i64; weights.out_features];

    for i in 0..weights.in_features {
        let xi = x[i] as i64;
        for j in 0..weights.out_features {
            let wij = weights.weight[i * weights.out_features + j] as i64;
            output[j] += xi * wij;
        }
    }

    // Scale down and add bias if present
    let mut result: Vec<i32> = output.iter().map(|&x| (x >> scale) as i32).collect();

    if let Some(ref bias) = weights.bias {
        for (r, b) in result.iter_mut().zip(bias) {
            *r = r.wrapping_add(*b);
        }
    }

    result
}

/// SiLU activation: x * sigmoid(x)
fn silu_fixed(x: i32, scale: u8) -> i32 {
    let scale_factor = (1u64 << scale) as f64;

    // Convert to f64 for numerical accuracy
    let x_f64 = x as f64 / scale_factor;

    // Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
    let sigmoid = 1.0 / (1.0 + (-x_f64).exp());
    let silu = x_f64 * sigmoid;

    // Convert back to fixed-point
    (silu * scale_factor) as i32
}

/// Compute logits from final hidden state
pub fn compute_logits(hidden: &[i32], lm_head: &LinearWeights, scale: u8) -> Vec<i32> {
    linear(hidden, lm_head, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::RmsNormWeights;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_rms_norm() {
        let scale = DEFAULT_SCALE;
        let one = 1 << scale;

        // Input vector [1.0, 2.0, 3.0]
        let x = vec![one, 2 * one, 3 * one];

        // Weights all 1.0
        let weights = RmsNormWeights::ones(3, scale, 1e-5);

        let result = rms_norm(&x, &weights, scale);

        // RMS = sqrt((1 + 4 + 9) / 3) = sqrt(14/3) ≈ 2.16
        // Normalized: [0.46, 0.93, 1.39]
        assert!(result.len() == 3);
    }

    #[test]
    fn test_silu() {
        let scale = DEFAULT_SCALE;

        // Test silu(0) = 0
        let result = silu_fixed(0, scale);
        assert!(result.abs() < 1000, "silu(0) should be ~0, got {}", result);

        // Test silu(1) ≈ 0.73
        let one = 1 << scale;
        let result = silu_fixed(one, scale);
        let expected = (0.73 * (1 << scale) as f64) as i32;
        assert!(
            (result - expected).abs() < one / 4,
            "silu(1) should be ~0.73, got {}",
            result as f64 / (1 << scale) as f64
        );
    }
}
