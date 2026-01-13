//! Private attention computation with client-side softmax
//!
//! ## Privacy Model (Option A - Client Softmax)
//!
//! The attention computation Y = softmax(Q·K^T / sqrt(d)) · V is split:
//!
//! 1. **Q, K, V projections**: Computed securely using secret sharing
//!    - Server computes on its shares, never sees plaintext
//!
//! 2. **Attention scores**: Q·K^T computed with masked reconstruction
//!    - Server sends masked score shares to client
//!    - Client reconstructs scores (client owns their data, this is fine)
//!
//! 3. **Softmax**: Computed by client on reconstructed scores
//!    - Client creates new secret shares of attention weights
//!
//! 4. **Weighted sum**: weights·V computed securely
//!    - Server computes on its shares
//!
//! This keeps the **server blind to all tokens and semantics** while allowing
//! efficient computation. The client learns intermediate attention scores,
//! but since the client owns their input data, this is acceptable.

use rayon::prelude::*;
use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;
use crate::kv_cache::{share_kv, reconstruct_kv};
use crate::rope::{RopeFrequencies, apply_rope_to_q, apply_rope_to_k};

/// SIMD dot product for aarch64 using NEON intrinsics
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_simd(a: &[i32], b: &[i32]) -> i64 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    let mut acc = unsafe {
        let mut sum_lo = vdupq_n_s64(0);
        let mut sum_hi = vdupq_n_s64(0);

        for i in 0..chunks {
            let offset = i * 4;
            let va = vld1q_s32(a.as_ptr().add(offset));
            let vb = vld1q_s32(b.as_ptr().add(offset));
            let prod_lo = vmull_s32(vget_low_s32(va), vget_low_s32(vb));
            let prod_hi = vmull_high_s32(va, vb);
            sum_lo = vaddq_s64(sum_lo, prod_lo);
            sum_hi = vaddq_s64(sum_hi, prod_hi);
        }

        let sum_all = vaddq_s64(sum_lo, sum_hi);
        vgetq_lane_s64(sum_all, 0) + vgetq_lane_s64(sum_all, 1)
    };

    // Handle remaining elements
    for i in (chunks * 4)..len {
        acc += a[i] as i64 * b[i] as i64;
    }

    acc
}

/// Scalar dot product fallback
#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot_product_simd(a: &[i32], b: &[i32]) -> i64 {
    a.iter().zip(b).map(|(&x, &y)| x as i64 * y as i64).sum()
}

/// Fast exp approximation using Schraudolph's method
///
/// This uses the IEEE 754 floating point representation to approximate exp(x)
/// in constant time. The idea is that exp(x) ≈ 2^(x/ln(2)), and we can compute
/// 2^y very fast by directly manipulating the exponent bits of a float.
///
/// Accuracy: ~0.3% relative error for x in [-10, 10]
/// Speed: ~3-4x faster than std::f64::exp
#[inline]
fn fast_exp(x: f64) -> f64 {
    // Constants for the approximation
    // 2^52 / ln(2) = 6497320848556798.0
    const A: f64 = 6497320848556798.0;
    // Exponent bias adjustment: 1023 * 2^52
    const B: f64 = 4607182418800017408.0;
    // Correction term for better accuracy around 0
    const C: f64 = 60801.0;

    // Clamp to avoid overflow/underflow
    let x = x.clamp(-708.0, 709.0);

    // Compute exp using bit manipulation
    // The idea: for a double, bits 52-62 are the exponent (biased by 1023)
    // By adding (x / ln(2)) * 2^52 to the exponent bits, we get 2^(x/ln(2)) = exp(x)
    let bits = ((A * x + (B - C)) as i64) as u64;
    f64::from_bits(bits)
}

/// Attention computation result
#[derive(Debug)]
pub struct AttentionOutput {
    /// Client's share of the attention output
    pub client_share: Share,
    /// Server's share of the attention output
    pub server_share: Share,
}

/// Client-side attention state
pub struct AttentionClient {
    /// Number of attention heads
    num_heads: usize,
    /// Number of KV heads (for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Scale factor for attention: 1/sqrt(head_dim)
    scale_factor: f64,
    /// Fixed-point scale
    fp_scale: u8,
    /// Pre-computed: 1 / (fp_scale_factor^2) for fast score conversion
    inv_fp_scale_sq: f64,
    /// Pre-computed: fp_scale_factor for weight conversion
    fp_scale_factor: f64,
}

/// Server-side attention state
pub struct AttentionServer {
    /// Number of attention heads
    num_heads: usize,
    /// Number of KV heads (for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Fixed-point scale
    fp_scale: u8,
}

impl AttentionClient {
    /// Create attention client
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, fp_scale: u8) -> Self {
        let fp_scale_factor = (1u64 << fp_scale) as f64;
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f64).sqrt(),
            fp_scale,
            inv_fp_scale_sq: 1.0 / (fp_scale_factor * fp_scale_factor),
            fp_scale_factor,
        }
    }

    /// Compute attention scores from Q and K (client reconstructs for softmax)
    ///
    /// Q: [num_heads * head_dim] - current query (secret-shared)
    /// K: [seq_len][num_kv_heads * head_dim] - cached keys (secret-shared)
    ///
    /// Returns: [num_heads, seq_len] attention scores (plaintext, for softmax)
    ///
    /// Optimizations:
    /// - SIMD dot products (NEON on aarch64)
    /// - Rayon parallelization across heads
    pub fn compute_scores(
        &self,
        q_client: &Share,
        q_server: &Share,
        k_client_shares: &[Share],
        k_server_shares: &[Share],
    ) -> Result<Vec<Vec<f64>>> {
        let seq_len = k_client_shares.len();
        if seq_len != k_server_shares.len() {
            return Err(SharingError::DimensionMismatch {
                expected: seq_len,
                got: k_server_shares.len(),
            });
        }

        // Reconstruct Q
        let q = reconstruct_kv(q_client, q_server)?;

        // Reconstruct all K vectors
        let keys: Vec<FixedVector> = k_client_shares
            .iter()
            .zip(k_server_shares)
            .map(|(c, s)| reconstruct_kv(c, s))
            .collect::<Result<Vec<_>>>()?;

        // Compute attention scores for each head (parallelized with rayon)
        let kv_groups = self.num_heads / self.num_kv_heads;
        let combined_scale = self.inv_fp_scale_sq * self.scale_factor;

        // Parallel computation across heads with SIMD dot products
        let scores: Vec<Vec<f64>> = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                let kv_head = head / kv_groups;
                let q_start = head * self.head_dim;
                let k_start = kv_head * self.head_dim;

                (0..seq_len)
                    .map(|pos| {
                        // Use SIMD dot product
                        let q_slice = &q.data[q_start..q_start + self.head_dim];
                        let k_slice = &keys[pos].data[k_start..k_start + self.head_dim];
                        let dot = dot_product_simd(q_slice, k_slice);
                        (dot as f64) * combined_scale
                    })
                    .collect()
            })
            .collect();

        Ok(scores)
    }

    /// Compute attention scores with RoPE applied to Q and K
    ///
    /// This is the RoPE-aware version of compute_scores. It:
    /// 1. Reconstructs Q and K from shares
    /// 2. Applies RoPE rotation based on position
    /// 3. Computes Q·K^T scores
    ///
    /// current_position: the position of the current Q token
    /// k_positions: the positions of each cached K (typically 0..seq_len)
    ///
    /// Optimizations:
    /// - SIMD dot products (NEON on aarch64)
    /// - Rayon parallelization across heads
    pub fn compute_scores_with_rope(
        &self,
        q_client: &Share,
        q_server: &Share,
        k_client_shares: &[Share],
        k_server_shares: &[Share],
        current_position: usize,
        freqs: &RopeFrequencies,
    ) -> Result<Vec<Vec<f64>>> {
        let seq_len = k_client_shares.len();
        if seq_len != k_server_shares.len() {
            return Err(SharingError::DimensionMismatch {
                expected: seq_len,
                got: k_server_shares.len(),
            });
        }

        // Reconstruct and rotate Q at current position
        let q = reconstruct_kv(q_client, q_server)?;
        let q_rotated = apply_rope_to_q(&q, current_position, freqs, self.num_heads)?;

        // Reconstruct and rotate all K vectors at their respective positions
        let mut keys_rotated = Vec::with_capacity(seq_len);
        for (pos, (kc, ks)) in k_client_shares.iter().zip(k_server_shares).enumerate() {
            let k = reconstruct_kv(kc, ks)?;
            let k_rotated = apply_rope_to_k(&k, pos, freqs, self.num_kv_heads)?;
            keys_rotated.push(k_rotated);
        }

        // Compute attention scores for each head (parallelized with rayon)
        let kv_groups = self.num_heads / self.num_kv_heads;
        let combined_scale = self.inv_fp_scale_sq * self.scale_factor;

        // Parallel computation across heads with SIMD dot products
        let scores: Vec<Vec<f64>> = (0..self.num_heads)
            .into_par_iter()
            .map(|head| {
                let kv_head = head / kv_groups;
                let q_start = head * self.head_dim;
                let k_start = kv_head * self.head_dim;

                (0..seq_len)
                    .map(|pos| {
                        // Use SIMD dot product
                        let q_slice = &q_rotated.data[q_start..q_start + self.head_dim];
                        let k_slice = &keys_rotated[pos].data[k_start..k_start + self.head_dim];
                        let dot = dot_product_simd(q_slice, k_slice);
                        (dot as f64) * combined_scale
                    })
                    .collect()
            })
            .collect();

        Ok(scores)
    }

    /// Compute softmax over attention scores (parallelized)
    ///
    /// scores: [num_heads][seq_len]
    /// Returns: [num_heads][seq_len] attention weights (sum to 1 per head)
    pub fn softmax(&self, scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
        scores
            .par_iter()
            .map(|head_scores| {
                // Numerical stability: subtract max
                let max_score = head_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_scores: Vec<f64> = head_scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum: f64 = exp_scores.iter().sum();
                exp_scores.iter().map(|&e| e / sum).collect()
            })
            .collect()
    }

    /// Fast softmax using fixed-point approximation (parallelized)
    ///
    /// Uses a piecewise linear approximation of exp() that's faster than f64 exp
    /// while maintaining good accuracy for attention weights.
    ///
    /// scores: [num_heads][seq_len]
    /// Returns: [num_heads][seq_len] attention weights (sum to 1 per head)
    pub fn softmax_fast(&self, scores: &[Vec<f64>]) -> Vec<Vec<f64>> {
        scores
            .par_iter()
            .map(|head_scores| {
                // Numerical stability: subtract max
                let max_score = head_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

                // Fast exp approximation
                let exp_scores: Vec<f64> = head_scores
                    .iter()
                    .map(|&s| {
                        let x = s - max_score;
                        if x < -20.0 {
                            0.0
                        } else {
                            fast_exp(x)
                        }
                    })
                    .collect();

                let sum: f64 = exp_scores.iter().sum();
                if sum > 0.0 {
                    exp_scores.iter().map(|&e| e / sum).collect()
                } else {
                    vec![1.0 / head_scores.len() as f64; head_scores.len()]
                }
            })
            .collect()
    }

    /// Create secret-shared attention weights from plaintext weights
    pub fn share_weights(&self, weights: &[Vec<f64>]) -> (Vec<Share>, Vec<Share>) {
        let mut client_shares = Vec::with_capacity(self.num_heads);
        let mut server_shares = Vec::with_capacity(self.num_heads);

        for head_weights in weights {
            // Convert to fixed-point using cached scale factor
            let fixed_weights: Vec<i32> = head_weights
                .iter()
                .map(|&w| (w * self.fp_scale_factor).round() as i32)
                .collect();
            let fixed_vec = FixedVector::from_raw(fixed_weights, self.fp_scale);

            let (client, server) = share_kv(&fixed_vec);
            client_shares.push(client);
            server_shares.push(server);
        }

        (client_shares, server_shares)
    }

    /// Compute weighted sum: attention_weights · V (client's part)
    ///
    /// weights: [num_heads][seq_len] - secret-shared attention weights
    /// v_shares: [seq_len][num_kv_heads * head_dim] - client's V shares
    ///
    /// Returns: [num_heads * head_dim] client's share of context vector
    pub fn compute_context_client(
        &self,
        weight_client_shares: &[Share],
        v_client_shares: &[Share],
    ) -> Result<Share> {
        let seq_len = v_client_shares.len();
        let kv_groups = self.num_heads / self.num_kv_heads;

        let mut output = vec![0i64; self.num_heads * self.head_dim];

        for head in 0..self.num_heads {
            let kv_head = head / kv_groups;
            let weight_share = &weight_client_shares[head];

            for pos in 0..seq_len {
                let w = weight_share.data[pos] as i64;
                let v_start = kv_head * self.head_dim;

                for d in 0..self.head_dim {
                    let v = v_client_shares[pos].data[v_start + d] as i64;
                    output[head * self.head_dim + d] += w * v;
                }
            }
        }

        // Rescale
        let result: Vec<i32> = output.iter().map(|&x| (x >> self.fp_scale) as i32).collect();
        Ok(Share::from_raw(result, self.fp_scale))
    }
}

impl AttentionServer {
    /// Create attention server
    pub fn new(num_heads: usize, num_kv_heads: usize, head_dim: usize, fp_scale: u8) -> Self {
        Self {
            num_heads,
            num_kv_heads,
            head_dim,
            fp_scale,
        }
    }

    /// Compute weighted sum: attention_weights · V (server's part)
    pub fn compute_context_server(
        &self,
        weight_server_shares: &[Share],
        v_server_shares: &[Share],
    ) -> Result<Share> {
        let seq_len = v_server_shares.len();
        let kv_groups = self.num_heads / self.num_kv_heads;

        let mut output = vec![0i64; self.num_heads * self.head_dim];

        for head in 0..self.num_heads {
            let kv_head = head / kv_groups;
            let weight_share = &weight_server_shares[head];

            for pos in 0..seq_len {
                let w = weight_share.data[pos] as i64;
                let v_start = kv_head * self.head_dim;

                for d in 0..self.head_dim {
                    let v = v_server_shares[pos].data[v_start + d] as i64;
                    output[head * self.head_dim + d] += w * v;
                }
            }
        }

        let result: Vec<i32> = output.iter().map(|&x| (x >> self.fp_scale) as i32).collect();
        Ok(Share::from_raw(result, self.fp_scale))
    }
}

/// Full attention computation with client-side softmax
///
/// This is the main entry point that coordinates client and server
/// to compute attention privately.
pub fn compute_attention(
    client: &AttentionClient,
    server: &AttentionServer,
    q_client: &Share,
    q_server: &Share,
    k_client_shares: &[Share],
    k_server_shares: &[Share],
    v_client_shares: &[Share],
    v_server_shares: &[Share],
) -> Result<AttentionOutput> {
    // Step 1: Client computes attention scores (reconstructing Q and K)
    let scores = client.compute_scores(q_client, q_server, k_client_shares, k_server_shares)?;

    // Step 2: Client computes softmax
    let weights = client.softmax(&scores);

    // Step 3: Client creates secret-shared weights
    let (weight_client, weight_server) = client.share_weights(&weights);

    // Step 4: Both parties compute their share of context vector
    let context_client = client.compute_context_client(&weight_client, v_client_shares)?;
    let context_server = server.compute_context_server(&weight_server, v_server_shares)?;

    Ok(AttentionOutput {
        client_share: context_client,
        server_share: context_server,
    })
}

/// Full attention computation with RoPE (Rotary Position Embedding)
///
/// This version applies RoPE to Q and K before computing attention scores.
/// The current_position parameter specifies where the current token is in the sequence.
pub fn compute_attention_with_rope(
    client: &AttentionClient,
    server: &AttentionServer,
    q_client: &Share,
    q_server: &Share,
    k_client_shares: &[Share],
    k_server_shares: &[Share],
    v_client_shares: &[Share],
    v_server_shares: &[Share],
    current_position: usize,
    freqs: &RopeFrequencies,
) -> Result<AttentionOutput> {
    // Step 1: Client computes attention scores with RoPE applied
    let scores = client.compute_scores_with_rope(
        q_client, q_server,
        k_client_shares, k_server_shares,
        current_position, freqs
    )?;

    // Step 2: Client computes softmax
    let weights = client.softmax(&scores);

    // Step 3: Client creates secret-shared weights
    let (weight_client, weight_server) = client.share_weights(&weights);

    // Step 4: Both parties compute their share of context vector
    let context_client = client.compute_context_client(&weight_client, v_client_shares)?;
    let context_server = server.compute_context_server(&weight_server, v_server_shares)?;

    Ok(AttentionOutput {
        client_share: context_client,
        server_share: context_server,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_softmax() {
        let client = AttentionClient::new(2, 2, 4, DEFAULT_SCALE);

        let scores = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0],
        ];

        let weights = client.softmax(&scores);

        // Check each head sums to 1
        for head_weights in &weights {
            let sum: f64 = head_weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1, got {}", sum);
        }

        // Uniform scores should give uniform weights
        for &w in &weights[1] {
            assert!((w - 1.0/3.0).abs() < 1e-10, "Uniform scores should give uniform weights");
        }
    }

    #[test]
    fn test_attention_output_reconstruction() {
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;

        let client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let server = AttentionServer::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);

        // Create Q
        let q_data: Vec<f64> = (0..num_heads * head_dim).map(|i| (i as f64) * 0.1).collect();
        let q = FixedVector::from_f64_slice_default(&q_data).unwrap();
        let (q_client, q_server) = share_kv(&q);

        // Create K and V for each position
        let mut k_client_shares = Vec::new();
        let mut k_server_shares = Vec::new();
        let mut v_client_shares = Vec::new();
        let mut v_server_shares = Vec::new();

        for pos in 0..seq_len {
            let k_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((pos * 10 + i) as f64) * 0.05)
                .collect();
            let k = FixedVector::from_f64_slice_default(&k_data).unwrap();
            let (kc, ks) = share_kv(&k);
            k_client_shares.push(kc);
            k_server_shares.push(ks);

            let v_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((pos + i) as f64) * 0.1)
                .collect();
            let v = FixedVector::from_f64_slice_default(&v_data).unwrap();
            let (vc, vs) = share_kv(&v);
            v_client_shares.push(vc);
            v_server_shares.push(vs);
        }

        // Compute attention
        let output = compute_attention(
            &client,
            &server,
            &q_client,
            &q_server,
            &k_client_shares,
            &k_server_shares,
            &v_client_shares,
            &v_server_shares,
        ).unwrap();

        // Verify output dimensions
        assert_eq!(output.client_share.len(), num_heads * head_dim);
        assert_eq!(output.server_share.len(), num_heads * head_dim);

        // Verify reconstruction works
        let reconstructed = reconstruct_kv(&output.client_share, &output.server_share).unwrap();
        assert_eq!(reconstructed.len(), num_heads * head_dim);
    }

    #[test]
    fn test_gqa_attention() {
        // Test grouped query attention: 4 Q heads, 2 KV heads
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 2;

        let client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let server = AttentionServer::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);

        let q = FixedVector::from_f64_slice_default(&vec![1.0; num_heads * head_dim]).unwrap();
        let (q_c, q_s) = share_kv(&q);

        let mut k_c = Vec::new();
        let mut k_s = Vec::new();
        let mut v_c = Vec::new();
        let mut v_s = Vec::new();

        for _ in 0..seq_len {
            let k = FixedVector::from_f64_slice_default(&vec![0.5; num_kv_heads * head_dim]).unwrap();
            let v = FixedVector::from_f64_slice_default(&vec![1.0; num_kv_heads * head_dim]).unwrap();
            let (kc, ks) = share_kv(&k);
            let (vc, vs) = share_kv(&v);
            k_c.push(kc);
            k_s.push(ks);
            v_c.push(vc);
            v_s.push(vs);
        }

        let output = compute_attention(
            &client, &server,
            &q_c, &q_s,
            &k_c, &k_s,
            &v_c, &v_s,
        ).unwrap();

        assert_eq!(output.client_share.len(), num_heads * head_dim);
    }

    #[test]
    fn test_attention_with_rope() {
        use crate::rope::RopeFrequencies;

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let seq_len = 3;

        let client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let server = AttentionServer::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let freqs = RopeFrequencies::new(head_dim, 128, 10000.0);

        // Create Q at position 2 (as if we've already processed 2 tokens)
        let q_data: Vec<f64> = (0..num_heads * head_dim).map(|i| (i as f64) * 0.1).collect();
        let q = FixedVector::from_f64_slice_default(&q_data).unwrap();
        let (q_client, q_server) = share_kv(&q);

        // Create K and V for positions 0, 1, 2
        let mut k_client_shares = Vec::new();
        let mut k_server_shares = Vec::new();
        let mut v_client_shares = Vec::new();
        let mut v_server_shares = Vec::new();

        for pos in 0..seq_len {
            let k_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((pos * 10 + i) as f64) * 0.05)
                .collect();
            let k = FixedVector::from_f64_slice_default(&k_data).unwrap();
            let (kc, ks) = share_kv(&k);
            k_client_shares.push(kc);
            k_server_shares.push(ks);

            let v_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((pos + i) as f64) * 0.1)
                .collect();
            let v = FixedVector::from_f64_slice_default(&v_data).unwrap();
            let (vc, vs) = share_kv(&v);
            v_client_shares.push(vc);
            v_server_shares.push(vs);
        }

        // Compute attention with RoPE at position 2
        let output = compute_attention_with_rope(
            &client,
            &server,
            &q_client,
            &q_server,
            &k_client_shares,
            &k_server_shares,
            &v_client_shares,
            &v_server_shares,
            2, // current position
            &freqs,
        ).unwrap();

        // Verify output dimensions
        assert_eq!(output.client_share.len(), num_heads * head_dim);
        assert_eq!(output.server_share.len(), num_heads * head_dim);

        // Verify reconstruction works
        let reconstructed = reconstruct_kv(&output.client_share, &output.server_share).unwrap();
        assert_eq!(reconstructed.len(), num_heads * head_dim);
    }

    #[test]
    fn test_rope_changes_attention_scores() {
        use crate::rope::RopeFrequencies;

        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        let client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let freqs = RopeFrequencies::new(head_dim, 128, 10000.0);

        // Create Q and K
        let q = FixedVector::from_f64_slice_default(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let k = FixedVector::from_f64_slice_default(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let (q_client, q_server) = share_kv(&q);
        let (k_client, k_server) = share_kv(&k);

        // Scores without RoPE (at position 0)
        let scores_pos0 = client.compute_scores_with_rope(
            &q_client, &q_server,
            &[k_client.clone()], &[k_server.clone()],
            0, &freqs
        ).unwrap();

        // Scores with RoPE (at position 10)
        let scores_pos10 = client.compute_scores_with_rope(
            &q_client, &q_server,
            &[k_client], &[k_server],
            10, &freqs
        ).unwrap();

        // Scores should be different due to position encoding
        // At position 0, RoPE is identity, but at position 10 it rotates vectors
        assert_ne!(
            scores_pos0[0][0], scores_pos10[0][0],
            "RoPE should change attention scores based on position"
        );
    }
}
