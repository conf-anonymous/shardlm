//! GPU-accelerated attention for privacy-preserving inference
//!
//! This module provides attention computation optimized for:
//! - Grouped Query Attention (GQA) used in Llama 3.x
//! - Long context windows (128K tokens)
//! - CUDA tensor cores for FP16/BF16

use crate::error::{Result, SharingError};
use rayon::prelude::*;
use shardlm_v2_core::tensor::{DType, Device};

/// GPU-accelerated attention context
pub struct GpuAttention {
    /// Number of query heads
    num_heads: usize,
    /// Number of key-value heads (for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Device to use
    device: Device,
    /// Use flash attention
    use_flash_attention: bool,
}

impl GpuAttention {
    /// Create a new GPU attention context
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        device: Device,
    ) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq_len,
            device,
            use_flash_attention: true,
        }
    }

    /// Create for Llama 70B
    pub fn for_llama_70b(device: Device) -> Self {
        Self::new(
            64,     // num_heads
            8,      // num_kv_heads (GQA ratio of 8)
            128,    // head_dim (8192 / 64)
            131072, // max_seq_len (128K)
            device,
        )
    }

    /// Get the GQA ratio (heads per KV head)
    pub fn gqa_ratio(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Compute attention scores (CPU fallback)
    ///
    /// This is a reference implementation. The actual GPU version will use
    /// FlashAttention-2 CUDA kernels.
    pub fn attention_cpu(
        &self,
        query: &[f32],      // [seq_len, num_heads, head_dim]
        key: &[f32],        // [seq_len, num_kv_heads, head_dim]
        value: &[f32],      // [seq_len, num_kv_heads, head_dim]
        seq_len: usize,
        causal: bool,
    ) -> Result<Vec<f32>> {
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let gqa_ratio = self.gqa_ratio();

        // Output: [seq_len, num_heads, head_dim]
        let output_size = seq_len * self.num_heads * self.head_dim;
        let mut output = vec![0.0f32; output_size];

        // Process heads in parallel
        let head_outputs: Vec<Vec<f32>> = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                let kv_head = head / gqa_ratio;
                let mut head_output = vec![0.0f32; seq_len * self.head_dim];

                for q_pos in 0..seq_len {
                    // Get query vector for this position
                    let q_start = (q_pos * self.num_heads + head) * self.head_dim;
                    let q = &query[q_start..q_start + self.head_dim];

                    // Compute attention scores
                    let k_end = if causal { q_pos + 1 } else { seq_len };
                    let mut scores = Vec::with_capacity(k_end);
                    let mut max_score = f32::NEG_INFINITY;

                    for k_pos in 0..k_end {
                        let k_start = (k_pos * self.num_kv_heads + kv_head) * self.head_dim;
                        let k = &key[k_start..k_start + self.head_dim];

                        // Dot product Q @ K^T
                        let score: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                        let scaled = score * scale;
                        scores.push(scaled);
                        max_score = max_score.max(scaled);
                    }

                    // Softmax
                    let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    let attention: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                    // Weighted sum of values
                    let out_start = q_pos * self.head_dim;
                    for k_pos in 0..k_end {
                        let v_start = (k_pos * self.num_kv_heads + kv_head) * self.head_dim;
                        let v = &value[v_start..v_start + self.head_dim];
                        let attn_weight = attention[k_pos];

                        for (i, &v_val) in v.iter().enumerate() {
                            head_output[out_start + i] += attn_weight * v_val;
                        }
                    }
                }

                head_output
            })
            .collect();

        // Combine head outputs
        for (head, head_out) in head_outputs.into_iter().enumerate() {
            for pos in 0..seq_len {
                let out_start = (pos * self.num_heads + head) * self.head_dim;
                let src_start = pos * self.head_dim;
                output[out_start..out_start + self.head_dim]
                    .copy_from_slice(&head_out[src_start..src_start + self.head_dim]);
            }
        }

        Ok(output)
    }

    /// Compute attention with secret-shared inputs
    ///
    /// This performs attention where Q is secret-shared (client has Q0, server has Q1)
    /// and K, V are held by the server.
    pub fn shared_attention_cpu(
        &self,
        query_share: &[f32], // Client's share of query
        key: &[f32],         // Server's key (unshared)
        value: &[f32],       // Server's value (unshared)
        seq_len: usize,
        _counter: u64,
    ) -> Result<Vec<f32>> {
        // In the privacy-preserving setting:
        // 1. Client holds Q0, server holds Q1 where Q = Q0 + Q1
        // 2. We use OT to compute softmax(Q @ K^T / sqrt(d)) without revealing Q
        // 3. Server computes attention output with the softmax weights

        // For now, this is a placeholder that shows the interface.
        // The actual implementation requires integration with the OT layer.

        // This returns the client's share of the attention output
        self.attention_cpu(query_share, key, value, seq_len, true)
    }
}

/// Flash Attention configuration
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for tiling
    pub block_size: usize,
    /// Use causal masking
    pub causal: bool,
    /// Dropout probability (0.0 for inference)
    pub dropout: f32,
    /// Softmax scale (1/sqrt(d) by default)
    pub softmax_scale: Option<f32>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size: 128,
            causal: true,
            dropout: 0.0,
            softmax_scale: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_ratio() {
        let attn = GpuAttention::for_llama_70b(Device::Cpu);
        assert_eq!(attn.gqa_ratio(), 8); // 64 heads / 8 kv_heads
    }

    #[test]
    fn test_attention_cpu() {
        let attn = GpuAttention::new(2, 2, 4, 32, Device::Cpu);

        let seq_len = 3;
        let query = vec![1.0f32; seq_len * 2 * 4]; // [3, 2, 4]
        let key = vec![1.0f32; seq_len * 2 * 4];
        let value = vec![1.0f32; seq_len * 2 * 4];

        let output = attn.attention_cpu(&query, &key, &value, seq_len, true).unwrap();
        assert_eq!(output.len(), seq_len * 2 * 4);
    }
}
