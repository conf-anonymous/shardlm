//! Client-side RMSNorm for private inference
//!
//! ## RMSNorm Formula
//!
//! RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
//!
//! Where:
//! - x is the input vector [hidden_size]
//! - weight is the learned scaling parameter [hidden_size]
//! - eps is a small constant for numerical stability (typically 1e-5)
//!
//! ## Privacy Model
//!
//! RMSNorm requires computing sqrt(mean(x²)), which is non-linear.
//! Like softmax and SiLU, this must be computed by the client:
//!
//! 1. Client reconstructs hidden state from shares
//! 2. Client computes RMSNorm on plaintext
//! 3. Client creates new secret shares of normalized output
//!
//! The server never sees the hidden state values, only random shares.

use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;
use crate::kv_cache::{share_kv, reconstruct_kv};

/// RMSNorm layer configuration
#[derive(Clone)]
pub struct RmsNormConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Epsilon for numerical stability
    pub eps: f64,
    /// Fixed-point scale
    pub scale: u8,
}

impl Default for RmsNormConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            eps: 1e-5,
            scale: shardlm_fixed_point::DEFAULT_SCALE,
        }
    }
}

/// Client-side RMSNorm
///
/// The client computes normalization since it requires non-linear operations.
pub struct RmsNormClient {
    /// Hidden dimension
    hidden_size: usize,
    /// Epsilon for numerical stability
    eps: f64,
    /// Learned weight parameters [hidden_size]
    weight: Vec<f64>,
    /// Fixed-point scale
    scale: u8,
}

impl RmsNormClient {
    /// Create a new RMSNorm layer
    pub fn new(config: &RmsNormConfig, weight: Vec<f64>) -> Result<Self> {
        if weight.len() != config.hidden_size {
            return Err(SharingError::DimensionMismatch {
                expected: config.hidden_size,
                got: weight.len(),
            });
        }

        Ok(Self {
            hidden_size: config.hidden_size,
            eps: config.eps,
            weight,
            scale: config.scale,
        })
    }

    /// Create RMSNorm with unit weights (for testing)
    pub fn with_unit_weights(config: &RmsNormConfig) -> Self {
        Self {
            hidden_size: config.hidden_size,
            eps: config.eps,
            weight: vec![1.0; config.hidden_size],
            scale: config.scale,
        }
    }

    /// Create RMSNorm from fixed-point weights
    pub fn from_fixed_weights(config: &RmsNormConfig, weight: &[i32]) -> Result<Self> {
        if weight.len() != config.hidden_size {
            return Err(SharingError::DimensionMismatch {
                expected: config.hidden_size,
                got: weight.len(),
            });
        }

        let scale_factor = (1u64 << config.scale) as f64;
        let weight_f64: Vec<f64> = weight
            .iter()
            .map(|&w| w as f64 / scale_factor)
            .collect();

        Ok(Self {
            hidden_size: config.hidden_size,
            eps: config.eps,
            weight: weight_f64,
            scale: config.scale,
        })
    }

    /// Compute RMSNorm on secret-shared input
    ///
    /// Reconstructs the input, computes normalization, and re-shares.
    pub fn forward(
        &self,
        client_share: &Share,
        server_share: &Share,
    ) -> Result<(Share, Share)> {
        // Step 1: Reconstruct input
        let input = reconstruct_kv(client_share, server_share)?;

        // Step 2: Convert to f64 for normalization
        let scale_factor = (1u64 << self.scale) as f64;
        let x: Vec<f64> = input.data.iter().map(|&v| v as f64 / scale_factor).collect();

        // Step 3: Compute RMS
        let mean_sq: f64 = x.iter().map(|&v| v * v).sum::<f64>() / self.hidden_size as f64;
        let rms = (mean_sq + self.eps).sqrt();

        // Step 4: Normalize and apply weight
        let normalized: Vec<f64> = x
            .iter()
            .zip(&self.weight)
            .map(|(&xi, &wi)| (xi / rms) * wi)
            .collect();

        // Step 5: Convert back to fixed-point
        let output_fixed: Vec<i32> = normalized
            .iter()
            .map(|&v| (v * scale_factor).round() as i32)
            .collect();
        let output = FixedVector::from_raw(output_fixed, self.scale);

        // Step 6: Create new secret shares
        Ok(share_kv(&output))
    }

    /// Compute RMSNorm on plaintext input (for testing/comparison)
    pub fn forward_plaintext(&self, input: &FixedVector) -> Result<FixedVector> {
        if input.len() != self.hidden_size {
            return Err(SharingError::DimensionMismatch {
                expected: self.hidden_size,
                got: input.len(),
            });
        }

        let scale_factor = (1u64 << self.scale) as f64;
        let x: Vec<f64> = input.data.iter().map(|&v| v as f64 / scale_factor).collect();

        let mean_sq: f64 = x.iter().map(|&v| v * v).sum::<f64>() / self.hidden_size as f64;
        let rms = (mean_sq + self.eps).sqrt();

        let normalized: Vec<f64> = x
            .iter()
            .zip(&self.weight)
            .map(|(&xi, &wi)| (xi / rms) * wi)
            .collect();

        let output_fixed: Vec<i32> = normalized
            .iter()
            .map(|&v| (v * scale_factor).round() as i32)
            .collect();

        Ok(FixedVector::from_raw(output_fixed, self.scale))
    }

    /// Get the hidden size
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Compute RMSNorm on secret-shared input (convenience function)
pub fn compute_rmsnorm(
    norm: &RmsNormClient,
    client_share: &Share,
    server_share: &Share,
) -> Result<(Share, Share)> {
    norm.forward(client_share, server_share)
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_rmsnorm_unit_weights() {
        let config = RmsNormConfig {
            hidden_size: 4,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let norm = RmsNormClient::with_unit_weights(&config);

        // Input: [1, 2, 3, 4]
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (client, server) = share_kv(&input);

        let (out_client, out_server) = norm.forward(&client, &server).unwrap();
        let output = reconstruct_kv(&out_client, &out_server).unwrap();

        // RMS = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.739
        // Normalized: [0.365, 0.730, 1.095, 1.461]
        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let out_f64: Vec<f64> = output.data.iter().map(|&v| v as f64 / scale_factor).collect();

        let expected_rms = (30.0f64 / 4.0).sqrt();
        let expected: Vec<f64> = [1.0, 2.0, 3.0, 4.0].iter().map(|&x| x / expected_rms).collect();

        for (got, exp) in out_f64.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < 0.01,
                "RMSNorm mismatch: got {}, expected {}",
                got, exp
            );
        }
    }

    #[test]
    fn test_rmsnorm_with_weights() {
        let config = RmsNormConfig {
            hidden_size: 4,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let weight = vec![2.0, 2.0, 2.0, 2.0]; // Scale output by 2
        let norm = RmsNormClient::new(&config, weight).unwrap();

        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (client, server) = share_kv(&input);

        let (out_client, out_server) = norm.forward(&client, &server).unwrap();
        let output = reconstruct_kv(&out_client, &out_server).unwrap();

        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let out_f64: Vec<f64> = output.data.iter().map(|&v| v as f64 / scale_factor).collect();

        // Should be 2x the unit weight output
        let expected_rms = (30.0f64 / 4.0).sqrt();
        let expected: Vec<f64> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&x| 2.0 * x / expected_rms)
            .collect();

        for (got, exp) in out_f64.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < 0.01,
                "RMSNorm mismatch: got {}, expected {}",
                got, exp
            );
        }
    }

    #[test]
    fn test_rmsnorm_preserves_direction() {
        let config = RmsNormConfig {
            hidden_size: 4,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let norm = RmsNormClient::with_unit_weights(&config);

        // Two inputs with same direction but different magnitudes
        let input1 = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let input2 = FixedVector::from_f64_slice_default(&[2.0, 4.0, 6.0, 8.0]).unwrap();

        let (c1, s1) = share_kv(&input1);
        let (c2, s2) = share_kv(&input2);

        let (oc1, os1) = norm.forward(&c1, &s1).unwrap();
        let (oc2, os2) = norm.forward(&c2, &s2).unwrap();

        let out1 = reconstruct_kv(&oc1, &os1).unwrap();
        let out2 = reconstruct_kv(&oc2, &os2).unwrap();

        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;

        // Outputs should be the same (normalized to same magnitude)
        for i in 0..4 {
            let v1 = out1.data[i] as f64 / scale_factor;
            let v2 = out2.data[i] as f64 / scale_factor;
            assert!(
                (v1 - v2).abs() < 0.01,
                "RMSNorm should normalize to same magnitude: {} vs {}",
                v1, v2
            );
        }
    }

    #[test]
    fn test_rmsnorm_plaintext_matches_shared() {
        let config = RmsNormConfig {
            hidden_size: 8,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let weight: Vec<f64> = (0..8).map(|i| 0.5 + i as f64 * 0.1).collect();
        let norm = RmsNormClient::new(&config, weight).unwrap();

        let input = FixedVector::from_f64_slice_default(
            &[0.5, -0.3, 1.2, 0.8, -0.1, 0.4, 0.9, -0.7]
        ).unwrap();

        // Plaintext computation
        let plaintext_out = norm.forward_plaintext(&input).unwrap();

        // Secret-shared computation
        let (client, server) = share_kv(&input);
        let (out_client, out_server) = norm.forward(&client, &server).unwrap();
        let shared_out = reconstruct_kv(&out_client, &out_server).unwrap();

        // Results should match
        assert_eq!(plaintext_out.data, shared_out.data);
    }

    #[test]
    fn test_rmsnorm_numerical_stability() {
        let config = RmsNormConfig {
            hidden_size: 4,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let norm = RmsNormClient::with_unit_weights(&config);

        // Very small values (should not cause division by zero)
        let input = FixedVector::from_f64_slice_default(&[1e-6, 1e-6, 1e-6, 1e-6]).unwrap();
        let (client, server) = share_kv(&input);

        let result = norm.forward(&client, &server);
        assert!(result.is_ok(), "RMSNorm should handle small values");
    }

    #[test]
    fn test_rmsnorm_from_fixed_weights() {
        let config = RmsNormConfig {
            hidden_size: 4,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };

        // Create weights in fixed-point format
        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let weight_fixed: Vec<i32> = vec![
            (1.0 * scale_factor) as i32,
            (1.0 * scale_factor) as i32,
            (1.0 * scale_factor) as i32,
            (1.0 * scale_factor) as i32,
        ];

        let norm = RmsNormClient::from_fixed_weights(&config, &weight_fixed).unwrap();
        assert_eq!(norm.hidden_size(), 4);

        // Should behave like unit weights
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = norm.forward_plaintext(&input).unwrap();
        assert_eq!(output.len(), 4);
    }
}
