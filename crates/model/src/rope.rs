//! Rotary Position Embedding (RoPE)
//!
//! Implements the rotary position embedding from the RoFormer paper:
//! "RoFormer: Enhanced Transformer with Rotary Position Embedding"
//! https://arxiv.org/abs/2104.09864
//!
//! RoPE encodes position information by rotating pairs of features.
//! For position m and dimension d with freq θ_d:
//!   RoPE(x, m)[2i:2i+2] = R(m·θ_i) · x[2i:2i+2]
//!
//! where R(θ) is a 2D rotation matrix:
//!   R(θ) = [[cos(θ), -sin(θ)],
//!           [sin(θ),  cos(θ)]]

use shardlm_fixed_point::FixedVector;

use crate::error::{ModelError, Result};
use crate::config::TinyLlamaConfig;

/// Precomputed RoPE frequencies (cos and sin tables)
#[derive(Debug, Clone)]
pub struct RopeFrequencies {
    /// Cosine values: [max_seq_len, head_dim/2]
    pub cos: Vec<Vec<f64>>,
    /// Sine values: [max_seq_len, head_dim/2]
    pub sin: Vec<Vec<f64>>,
    /// Head dimension
    pub head_dim: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Base frequency (theta)
    pub theta: f64,
}

impl RopeFrequencies {
    /// Create RoPE frequencies from config
    pub fn from_config(config: &TinyLlamaConfig) -> Self {
        Self::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
        )
    }

    /// Create RoPE frequencies with specified parameters
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        // Compute inverse frequencies: 1 / (theta^(2i/d)) for i in 0..d/2
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect();

        // Compute cos and sin for each position
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
            theta,
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
}

/// Apply RoPE to a single head's Q or K vector (f64 version)
///
/// x: input vector of shape [head_dim]
/// cos, sin: precomputed values of shape [head_dim/2]
///
/// Returns: rotated vector of shape [head_dim]
pub fn apply_rope_f64(x: &[f64], cos: &[f64], sin: &[f64]) -> Vec<f64> {
    let half_dim = x.len() / 2;
    let mut output = vec![0.0; x.len()];

    for i in 0..half_dim {
        let x0 = x[i];
        let x1 = x[i + half_dim];
        let c = cos[i];
        let s = sin[i];

        // Apply rotation: [cos -sin] [x0]
        //                 [sin  cos] [x1]
        output[i] = x0 * c - x1 * s;
        output[i + half_dim] = x0 * s + x1 * c;
    }

    output
}

/// Apply RoPE to a single head's Q or K vector (fixed-point version)
///
/// x: input vector of shape [head_dim]
/// cos, sin: precomputed values of shape [head_dim/2]
/// scale: fixed-point scale factor
///
/// Returns: rotated vector in fixed-point
pub fn apply_rope_fixed(
    x: &FixedVector,
    cos: &[f64],
    sin: &[f64],
) -> Result<FixedVector> {
    if x.len() % 2 != 0 {
        return Err(ModelError::Rope(format!(
            "RoPE requires even head_dim, got {}",
            x.len()
        )));
    }

    let half_dim = x.len() / 2;
    if cos.len() != half_dim || sin.len() != half_dim {
        return Err(ModelError::Rope(format!(
            "cos/sin length ({}/{}) must equal head_dim/2 ({})",
            cos.len(),
            sin.len(),
            half_dim
        )));
    }

    let scale_factor = (1u64 << x.scale) as f64;
    let mut output = vec![0i32; x.len()];

    for i in 0..half_dim {
        // Convert fixed-point to f64
        let x0 = x.data[i] as f64 / scale_factor;
        let x1 = x.data[i + half_dim] as f64 / scale_factor;
        let c = cos[i];
        let s = sin[i];

        // Apply rotation
        let y0 = x0 * c - x1 * s;
        let y1 = x0 * s + x1 * c;

        // Convert back to fixed-point
        output[i] = (y0 * scale_factor).round() as i32;
        output[i + half_dim] = (y1 * scale_factor).round() as i32;
    }

    Ok(FixedVector::from_raw(output, x.scale))
}

/// Apply RoPE to a batch of vectors at different positions
///
/// vectors: [seq_len] vectors, each of shape [head_dim]
/// start_position: the position of the first vector (for KV cache)
/// freqs: precomputed RoPE frequencies
pub fn apply_rope_batch(
    vectors: &[FixedVector],
    start_position: usize,
    freqs: &RopeFrequencies,
) -> Result<Vec<FixedVector>> {
    let mut outputs = Vec::with_capacity(vectors.len());

    for (i, vec) in vectors.iter().enumerate() {
        let pos = start_position + i;
        let (cos, sin) = freqs.get(pos).ok_or_else(|| {
            ModelError::Rope(format!(
                "Position {} exceeds max_seq_len {}",
                pos, freqs.max_seq_len
            ))
        })?;

        outputs.push(apply_rope_fixed(vec, cos, sin)?);
    }

    Ok(outputs)
}

/// Apply RoPE to Q and K projections for a full sequence
///
/// This is the main entry point for integrating RoPE into attention.
///
/// q_vectors: [seq_len] Q vectors, each of shape [head_dim]
/// k_vectors: [seq_len] K vectors, each of shape [head_dim]
/// start_position: position offset (for incremental decoding)
/// config: model configuration
pub fn apply_rope_qk(
    q_vectors: &[FixedVector],
    k_vectors: &[FixedVector],
    start_position: usize,
    config: &TinyLlamaConfig,
) -> Result<(Vec<FixedVector>, Vec<FixedVector>)> {
    let freqs = RopeFrequencies::from_config(config);

    let q_rotated = apply_rope_batch(q_vectors, start_position, &freqs)?;
    let k_rotated = apply_rope_batch(k_vectors, start_position, &freqs)?;

    Ok((q_rotated, k_rotated))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_frequencies_creation() {
        let freqs = RopeFrequencies::new(64, 128, 10000.0);

        assert_eq!(freqs.head_dim, 64);
        assert_eq!(freqs.max_seq_len, 128);
        assert_eq!(freqs.cos.len(), 128);
        assert_eq!(freqs.sin.len(), 128);
        assert_eq!(freqs.cos[0].len(), 32); // head_dim / 2
    }

    #[test]
    fn test_rope_position_zero() {
        // At position 0, all angles are 0, so cos=1, sin=0
        let freqs = RopeFrequencies::new(64, 128, 10000.0);
        let (cos, sin) = freqs.get(0).unwrap();

        for &c in cos {
            assert!((c - 1.0).abs() < 1e-10, "cos(0) should be 1.0");
        }
        for &s in sin {
            assert!(s.abs() < 1e-10, "sin(0) should be 0.0");
        }
    }

    #[test]
    fn test_rope_f64_identity_at_position_zero() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (cos, sin) = freqs.get(0).unwrap();

        let output = apply_rope_f64(&x, cos, sin);

        // At position 0, rotation is identity
        for (i, (&out, &inp)) in output.iter().zip(&x).enumerate() {
            assert!(
                (out - inp).abs() < 1e-10,
                "Position 0 should be identity: out[{}]={}, inp[{}]={}",
                i, out, i, inp
            );
        }
    }

    #[test]
    fn test_rope_f64_rotation_property() {
        // Applying the same rotation twice should equal applying at double frequency
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let x = vec![1.0, 0.0, 0.0, 0.0];

        let (cos1, sin1) = freqs.get(1).unwrap();
        let y1 = apply_rope_f64(&x, cos1, sin1);
        let y2 = apply_rope_f64(&y1, cos1, sin1);

        let (cos2, sin2) = freqs.get(2).unwrap();
        let y_direct = apply_rope_f64(&x, cos2, sin2);

        // y2 should equal y_direct (applying rotation twice = rotation at 2x angle)
        for i in 0..4 {
            assert!(
                (y2[i] - y_direct[i]).abs() < 1e-10,
                "Double rotation should equal rotation at 2x: y2[{}]={}, y_direct[{}]={}",
                i, y2[i], i, y_direct[i]
            );
        }
    }

    #[test]
    fn test_rope_fixed_point_matches_f64() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let x_f64 = vec![1.5, -2.25, 0.75, 3.0];
        let x_fixed = FixedVector::from_f64_slice_default(&x_f64).unwrap();

        let (cos, sin) = freqs.get(5).unwrap();

        let y_f64 = apply_rope_f64(&x_f64, cos, sin);
        let y_fixed = apply_rope_fixed(&x_fixed, cos, sin).unwrap();
        let y_fixed_f64 = y_fixed.to_f64_vec();

        for i in 0..4 {
            let diff = (y_f64[i] - y_fixed_f64[i]).abs();
            assert!(
                diff < 0.001,
                "Fixed-point should match f64: y_f64[{}]={}, y_fixed[{}]={}",
                i, y_f64[i], i, y_fixed_f64[i]
            );
        }
    }

    #[test]
    fn test_rope_batch() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);

        let vectors: Vec<FixedVector> = (0..6)
            .map(|i| {
                let data = vec![i as f64 + 1.0, 0.5, -0.5, i as f64 * 0.1];
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let rotated = apply_rope_batch(&vectors, 0, &freqs).unwrap();

        assert_eq!(rotated.len(), 6);

        // First vector (position 0) should be unchanged
        for i in 0..4 {
            let diff = (rotated[0].data[i] - vectors[0].data[i]).abs();
            assert!(diff <= 1, "Position 0 should be identity");
        }
    }

    #[test]
    fn test_rope_preserves_magnitude() {
        // Rotation should preserve vector magnitude
        let freqs = RopeFrequencies::new(4, 128, 10000.0);
        let x = vec![3.0, 4.0, 0.0, 0.0];

        let (cos, sin) = freqs.get(10).unwrap();
        let y = apply_rope_f64(&x, cos, sin);

        // Check first two elements (one rotation pair)
        let mag_before = (x[0] * x[0] + x[2] * x[2]).sqrt();
        let mag_after = (y[0] * y[0] + y[2] * y[2]).sqrt();

        assert!(
            (mag_before - mag_after).abs() < 1e-10,
            "Rotation should preserve magnitude: before={}, after={}",
            mag_before, mag_after
        );
    }

    #[test]
    fn test_rope_from_config() {
        let config = TinyLlamaConfig::default();
        let freqs = RopeFrequencies::from_config(&config);

        assert_eq!(freqs.head_dim, 64); // 2048 / 32
        assert_eq!(freqs.max_seq_len, 2048);
        assert_eq!(freqs.theta, 10000.0);
    }

    #[test]
    fn test_apply_rope_qk() {
        let config = TinyLlamaConfig {
            hidden_size: 8,
            num_attention_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            ..Default::default()
        };

        // Create Q and K vectors (head_dim = 8/2 = 4)
        let q_vectors: Vec<FixedVector> = (0..3)
            .map(|_| FixedVector::from_f64_slice_default(&[1.0, 0.0, 0.0, 1.0]).unwrap())
            .collect();
        let k_vectors: Vec<FixedVector> = (0..3)
            .map(|_| FixedVector::from_f64_slice_default(&[0.5, 0.5, -0.5, 0.5]).unwrap())
            .collect();

        let (q_rotated, k_rotated) = apply_rope_qk(&q_vectors, &k_vectors, 0, &config).unwrap();

        assert_eq!(q_rotated.len(), 3);
        assert_eq!(k_rotated.len(), 3);
    }
}
