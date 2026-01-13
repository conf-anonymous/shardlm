//! Rotary Position Embedding (RoPE) for private attention
//!
//! RoPE encodes position information by rotating pairs of features.
//! Since RoPE requires plaintext values (cos/sin multiplication), it's computed
//! client-side after reconstructing Q and K from secret shares.
//!
//! ## Integration with Private Attention
//!
//! 1. Client reconstructs Q and K from shares (already done for softmax)
//! 2. Client applies RoPE rotation to Q and K
//! 3. Client computes attention scores with rotated vectors
//!
//! The server never sees the Q/K values or the rotation, maintaining privacy.

use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;
use crate::kv_cache::{share_kv, reconstruct_kv};

/// Precomputed RoPE frequencies (cos and sin tables)
#[derive(Debug, Clone)]
pub struct RopeFrequencies {
    /// Cosine values: [max_seq_len][head_dim/2]
    cos: Vec<Vec<f64>>,
    /// Sine values: [max_seq_len][head_dim/2]
    sin: Vec<Vec<f64>>,
    /// Head dimension
    head_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl RopeFrequencies {
    /// Create RoPE frequencies with specified parameters
    ///
    /// head_dim: dimension per attention head
    /// max_seq_len: maximum sequence length to precompute
    /// theta: base frequency (typically 10000.0)
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in 0..d/2
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect();

        // Precompute cos and sin for each position
        let mut cos = Vec::with_capacity(max_seq_len);
        let mut sin = Vec::with_capacity(max_seq_len);

        for pos in 0..max_seq_len {
            let pos_f64 = pos as f64;
            let cos_row: Vec<f64> = inv_freq.iter().map(|&f| (pos_f64 * f).cos()).collect();
            let sin_row: Vec<f64> = inv_freq.iter().map(|&f| (pos_f64 * f).sin()).collect();
            cos.push(cos_row);
            sin.push(sin_row);
        }

        Self {
            cos,
            sin,
            head_dim,
            max_seq_len,
        }
    }

    /// Get cos/sin values for a specific position
    pub fn get(&self, position: usize) -> Option<(&[f64], &[f64])> {
        if position < self.max_seq_len {
            Some((&self.cos[position], &self.sin[position]))
        } else {
            None
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get max sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Apply RoPE rotation to a single vector
///
/// x: input vector of shape [head_dim]
/// cos, sin: precomputed values of shape [head_dim/2]
///
/// Returns rotated vector. This follows HuggingFace's rotate_half pattern:
/// - First half: x[i] * cos[i] - x[i + half] * sin[i]
/// - Second half: x[i] * sin[i] + x[i + half] * cos[i]
pub fn apply_rope(x: &FixedVector, cos: &[f64], sin: &[f64]) -> Result<FixedVector> {
    let half_dim = x.len() / 2;

    if x.len() % 2 != 0 {
        return Err(SharingError::DimensionMismatch {
            expected: x.len() / 2 * 2,
            got: x.len(),
        });
    }

    if cos.len() != half_dim || sin.len() != half_dim {
        return Err(SharingError::DimensionMismatch {
            expected: half_dim,
            got: cos.len(),
        });
    }

    let scale_factor = (1u64 << x.scale) as f64;
    let mut output = vec![0i32; x.len()];

    for i in 0..half_dim {
        let x0 = x.data[i] as f64 / scale_factor;
        let x1 = x.data[i + half_dim] as f64 / scale_factor;
        let c = cos[i];
        let s = sin[i];

        // Apply rotation matrix
        let y0 = x0 * c - x1 * s;
        let y1 = x0 * s + x1 * c;

        output[i] = (y0 * scale_factor).round() as i32;
        output[i + half_dim] = (y1 * scale_factor).round() as i32;
    }

    Ok(FixedVector::from_raw(output, x.scale))
}

/// Apply RoPE to a multi-head Q vector
///
/// q: [num_heads * head_dim] query vector
/// position: sequence position for this query
/// freqs: precomputed RoPE frequencies
/// num_heads: number of attention heads
///
/// Returns rotated Q vector with RoPE applied to each head independently.
pub fn apply_rope_to_q(
    q: &FixedVector,
    position: usize,
    freqs: &RopeFrequencies,
    num_heads: usize,
) -> Result<FixedVector> {
    let head_dim = freqs.head_dim();

    if q.len() != num_heads * head_dim {
        return Err(SharingError::DimensionMismatch {
            expected: num_heads * head_dim,
            got: q.len(),
        });
    }

    let (cos, sin) = freqs.get(position).ok_or_else(|| {
        SharingError::DimensionMismatch {
            expected: freqs.max_seq_len(),
            got: position + 1,
        }
    })?;

    let scale_factor = (1u64 << q.scale) as f64;
    let mut output = vec![0i32; q.len()];
    let half_dim = head_dim / 2;

    for head in 0..num_heads {
        let offset = head * head_dim;

        for i in 0..half_dim {
            let x0 = q.data[offset + i] as f64 / scale_factor;
            let x1 = q.data[offset + i + half_dim] as f64 / scale_factor;
            let c = cos[i];
            let s = sin[i];

            let y0 = x0 * c - x1 * s;
            let y1 = x0 * s + x1 * c;

            output[offset + i] = (y0 * scale_factor).round() as i32;
            output[offset + i + half_dim] = (y1 * scale_factor).round() as i32;
        }
    }

    Ok(FixedVector::from_raw(output, q.scale))
}

/// Apply RoPE to a multi-head K vector
///
/// k: [num_kv_heads * head_dim] key vector
/// position: sequence position for this key
/// freqs: precomputed RoPE frequencies
/// num_kv_heads: number of KV heads (may differ from Q heads in GQA)
pub fn apply_rope_to_k(
    k: &FixedVector,
    position: usize,
    freqs: &RopeFrequencies,
    num_kv_heads: usize,
) -> Result<FixedVector> {
    let head_dim = freqs.head_dim();

    if k.len() != num_kv_heads * head_dim {
        return Err(SharingError::DimensionMismatch {
            expected: num_kv_heads * head_dim,
            got: k.len(),
        });
    }

    let (cos, sin) = freqs.get(position).ok_or_else(|| {
        SharingError::DimensionMismatch {
            expected: freqs.max_seq_len(),
            got: position + 1,
        }
    })?;

    let scale_factor = (1u64 << k.scale) as f64;
    let mut output = vec![0i32; k.len()];
    let half_dim = head_dim / 2;

    for head in 0..num_kv_heads {
        let offset = head * head_dim;

        for i in 0..half_dim {
            let x0 = k.data[offset + i] as f64 / scale_factor;
            let x1 = k.data[offset + i + half_dim] as f64 / scale_factor;
            let c = cos[i];
            let s = sin[i];

            let y0 = x0 * c - x1 * s;
            let y1 = x0 * s + x1 * c;

            output[offset + i] = (y0 * scale_factor).round() as i32;
            output[offset + i + half_dim] = (y1 * scale_factor).round() as i32;
        }
    }

    Ok(FixedVector::from_raw(output, k.scale))
}

/// Apply RoPE to secret-shared Q, returning rotated shares
///
/// This reconstructs Q, applies RoPE, and re-shares the result.
/// The server never sees the rotated values.
pub fn apply_rope_to_q_shared(
    q_client: &Share,
    q_server: &Share,
    position: usize,
    freqs: &RopeFrequencies,
    num_heads: usize,
) -> Result<(Share, Share)> {
    let q = reconstruct_kv(q_client, q_server)?;
    let q_rotated = apply_rope_to_q(&q, position, freqs, num_heads)?;
    Ok(share_kv(&q_rotated))
}

/// Apply RoPE to secret-shared K, returning rotated shares
pub fn apply_rope_to_k_shared(
    k_client: &Share,
    k_server: &Share,
    position: usize,
    freqs: &RopeFrequencies,
    num_kv_heads: usize,
) -> Result<(Share, Share)> {
    let k = reconstruct_kv(k_client, k_server)?;
    let k_rotated = apply_rope_to_k(&k, position, freqs, num_kv_heads)?;
    Ok(share_kv(&k_rotated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_rope_frequencies() {
        let freqs = RopeFrequencies::new(64, 128, 10000.0);

        assert_eq!(freqs.head_dim(), 64);
        assert_eq!(freqs.max_seq_len(), 128);

        // Position 0 should have cos=1, sin=0
        let (cos, sin) = freqs.get(0).unwrap();
        for &c in cos {
            assert!((c - 1.0).abs() < 1e-10);
        }
        for &s in sin {
            assert!(s.abs() < 1e-10);
        }
    }

    #[test]
    fn test_apply_rope_identity_at_pos_0() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let x = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let (cos, sin) = freqs.get(0).unwrap();
        let y = apply_rope(&x, cos, sin).unwrap();

        // At position 0, output should equal input
        assert_eq!(x.data, y.data);
    }

    #[test]
    fn test_apply_rope_to_q() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let num_heads = 2;

        // Q with 2 heads, each with head_dim=4
        let q = FixedVector::from_f64_slice_default(&[
            1.0, 0.0, 0.0, 0.0,  // Head 0
            0.0, 1.0, 0.0, 0.0,  // Head 1
        ]).unwrap();

        // At position 0, should be identity
        let q_rot = apply_rope_to_q(&q, 0, &freqs, num_heads).unwrap();
        assert_eq!(q.data, q_rot.data);

        // At position 5, should be different
        let q_rot5 = apply_rope_to_q(&q, 5, &freqs, num_heads).unwrap();
        assert_ne!(q.data, q_rot5.data);
    }

    #[test]
    fn test_apply_rope_shared() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let num_heads = 2;

        let q = FixedVector::from_f64_slice_default(&[
            1.0, 0.5, -0.3, 0.8,
            0.2, -0.4, 0.6, 0.1,
        ]).unwrap();
        let (q_client, q_server) = share_kv(&q);

        // Apply RoPE through shares
        let (rot_client, rot_server) = apply_rope_to_q_shared(
            &q_client, &q_server, 5, &freqs, num_heads
        ).unwrap();

        // Reconstruct and compare with direct application
        let q_rot_from_shares = reconstruct_kv(&rot_client, &rot_server).unwrap();
        let q_rot_direct = apply_rope_to_q(&q, 5, &freqs, num_heads).unwrap();

        assert_eq!(q_rot_from_shares.data, q_rot_direct.data);
    }

    #[test]
    fn test_rope_preserves_magnitude() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);

        let x = FixedVector::from_f64_slice_default(&[3.0, 4.0, 0.0, 0.0]).unwrap();
        let (cos, sin) = freqs.get(10).unwrap();
        let y = apply_rope(&x, cos, sin).unwrap();

        let scale = (1u64 << DEFAULT_SCALE) as f64;
        let x_f64: Vec<f64> = x.data.iter().map(|&v| v as f64 / scale).collect();
        let y_f64: Vec<f64> = y.data.iter().map(|&v| v as f64 / scale).collect();

        // First pair magnitude
        let mag_x = (x_f64[0].powi(2) + x_f64[2].powi(2)).sqrt();
        let mag_y = (y_f64[0].powi(2) + y_f64[2].powi(2)).sqrt();

        assert!((mag_x - mag_y).abs() < 0.01);
    }
}
