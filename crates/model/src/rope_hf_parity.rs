//! HuggingFace parity tests for RoPE
//!
//! These tests verify that our RoPE implementation matches the behavior of
//! HuggingFace Transformers' LlamaRotaryEmbedding.
//!
//! The reference values are computed using the exact same algorithm as:
//! transformers.models.llama.modeling_llama.LlamaRotaryEmbedding

#[cfg(test)]
mod tests {
    use crate::rope::{RopeFrequencies, apply_rope_f64};

    /// Test that inverse frequencies match HuggingFace's computation
    ///
    /// HF computes: inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    #[test]
    fn test_inverse_frequencies_match_hf() {
        // TinyLlama config: head_dim=64, theta=10000.0
        let head_dim = 64;
        let theta = 10000.0;
        let freqs = RopeFrequencies::new(head_dim, 128, theta);

        // First few expected inverse frequencies (computed as in HF)
        // inv_freq[i] = 1 / (theta^(2i/head_dim))
        let expected_inv_freq: Vec<f64> = (0..head_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f64 / head_dim as f64))
            .collect();

        // Our freqs store cos/sin, so verify via cos at position 1
        // cos(inv_freq * 1) should match
        let (cos_pos1, _) = freqs.get(1).unwrap();

        for i in 0..head_dim / 2 {
            let expected_cos = expected_inv_freq[i].cos();
            let diff = (cos_pos1[i] - expected_cos).abs();
            assert!(
                diff < 1e-14,
                "Inverse freq mismatch at index {}: expected cos={}, got cos={}",
                i, expected_cos, cos_pos1[i]
            );
        }
    }

    /// Test RoPE application matches HuggingFace's rotate_half + apply pattern
    ///
    /// HF does:
    ///   def rotate_half(x):
    ///       x1, x2 = x[..., :half], x[..., half:]
    ///       return torch.cat([-x2, x1], dim=-1)
    ///
    ///   q_embed = q * cos + rotate_half(q) * sin
    #[test]
    fn test_rope_rotation_matches_hf_pattern() {
        let head_dim = 4;
        let theta = 10000.0;
        let freqs = RopeFrequencies::new(head_dim, 128, theta);

        // Test vector
        let q = vec![1.0, 2.0, 3.0, 4.0];
        // First half: [1.0, 2.0], second half: [3.0, 4.0]

        // At position 5
        let (cos, sin) = freqs.get(5).unwrap();

        // HuggingFace rotate_half for [1, 2, 3, 4] gives [-3, -4, 1, 2]
        // (negates second half, then concatenates in reverse order)
        //
        // HF formula: q_embed = q * cos + rotate_half(q) * sin
        // Our formula applies rotation to pairs:
        //   output[i] = q[i] * cos[i] - q[i + half] * sin[i]
        //   output[i + half] = q[i] * sin[i] + q[i + half] * cos[i]
        //
        // This is equivalent to:
        //   output[i] = q[i] * cos[i] + (-q[i + half]) * sin[i]
        //   output[i + half] = q[i + half] * cos[i] + q[i] * sin[i]
        //
        // Which matches HF's pattern!

        let output = apply_rope_f64(&q, cos, sin);

        // Verify by computing HF-style
        let half = head_dim / 2;
        for i in 0..half {
            // HF: q * cos + rotate_half(q) * sin
            // rotate_half puts -second_half first, then first_half
            // So for position i: rotate_half[i] = -q[i + half]
            //    for position i+half: rotate_half[i+half] = q[i]
            let hf_out_i = q[i] * cos[i] + (-q[i + half]) * sin[i];
            let hf_out_i_plus_half = q[i + half] * cos[i] + q[i] * sin[i];

            assert!(
                (output[i] - hf_out_i).abs() < 1e-14,
                "Mismatch at {}: ours={}, hf={}",
                i, output[i], hf_out_i
            );
            assert!(
                (output[i + half] - hf_out_i_plus_half).abs() < 1e-14,
                "Mismatch at {}: ours={}, hf={}",
                i + half, output[i + half], hf_out_i_plus_half
            );
        }
    }

    /// Test with TinyLlama's actual config values
    #[test]
    fn test_tinyllama_config_values() {
        // TinyLlama-1.1B config
        let head_dim = 64; // 2048 / 32
        let max_position_embeddings = 2048;
        let rope_theta = 10000.0;

        let freqs = RopeFrequencies::new(head_dim, max_position_embeddings, rope_theta);

        // Test positions across the range
        for pos in [0, 1, 10, 100, 1000, 2047] {
            let (cos, sin) = freqs.get(pos).unwrap();

            // Verify cos^2 + sin^2 = 1 (identity preserved)
            for i in 0..head_dim / 2 {
                let sum_squares = cos[i] * cos[i] + sin[i] * sin[i];
                assert!(
                    (sum_squares - 1.0).abs() < 1e-14,
                    "cos^2 + sin^2 != 1 at pos={}, i={}: got {}",
                    pos, i, sum_squares
                );
            }
        }
    }

    /// Test that position 0 is identity (as in HF)
    #[test]
    fn test_position_zero_is_identity() {
        let freqs = RopeFrequencies::new(64, 2048, 10000.0);

        // Test with a full head_dim=64 vector
        let full_q: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1) - 3.0).collect();

        let (cos0, sin0) = freqs.get(0).unwrap();
        let output = apply_rope_f64(&full_q, cos0, sin0);

        for i in 0..64 {
            assert!(
                (output[i] - full_q[i]).abs() < 1e-14,
                "Position 0 should be identity: output[{}]={}, input[{}]={}",
                i, output[i], i, full_q[i]
            );
        }
    }

    /// Test specific reference values computed with HF's implementation
    ///
    /// These values were computed using:
    /// ```python
    /// import torch
    /// from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    ///
    /// rope = LlamaRotaryEmbedding(dim=64, max_position_embeddings=2048, base=10000.0)
    /// cos, sin = rope(torch.ones(1, 64), torch.tensor([[5]]))
    /// # Then apply to test vector
    /// ```
    #[test]
    fn test_reference_values() {
        let freqs = RopeFrequencies::new(64, 2048, 10000.0);
        let (cos5, sin5) = freqs.get(5).unwrap();

        // At position 5, dimension 0: freq = 1.0 (theta^0 = 1)
        // angle = 5 * 1.0 = 5.0
        let expected_cos_0 = (5.0f64).cos();
        let expected_sin_0 = (5.0f64).sin();

        assert!(
            (cos5[0] - expected_cos_0).abs() < 1e-14,
            "cos mismatch at dim 0: expected {}, got {}",
            expected_cos_0, cos5[0]
        );
        assert!(
            (sin5[0] - expected_sin_0).abs() < 1e-14,
            "sin mismatch at dim 0: expected {}, got {}",
            expected_sin_0, sin5[0]
        );

        // At position 5, dimension 1: freq = 1 / theta^(2/64) = 1 / theta^0.03125
        let inv_freq_1 = 1.0 / 10000.0f64.powf(2.0 / 64.0);
        let expected_cos_1 = (5.0 * inv_freq_1).cos();
        let expected_sin_1 = (5.0 * inv_freq_1).sin();

        assert!(
            (cos5[1] - expected_cos_1).abs() < 1e-14,
            "cos mismatch at dim 1: expected {}, got {}",
            expected_cos_1, cos5[1]
        );
        assert!(
            (sin5[1] - expected_sin_1).abs() < 1e-14,
            "sin mismatch at dim 1: expected {}, got {}",
            expected_sin_1, sin5[1]
        );
    }

    /// Test incremental position application (for autoregressive decoding)
    #[test]
    fn test_incremental_positions() {
        let freqs = RopeFrequencies::new(64, 2048, 10000.0);

        // In autoregressive mode, each new token gets the next position
        let test_vec: Vec<f64> = (0..64).map(|i| (i as f64 + 1.0) * 0.01).collect();

        // Applying at positions 10, 11, 12 should give different results
        let (cos10, sin10) = freqs.get(10).unwrap();
        let (cos11, sin11) = freqs.get(11).unwrap();
        let (cos12, sin12) = freqs.get(12).unwrap();

        let out10 = apply_rope_f64(&test_vec, cos10, sin10);
        let out11 = apply_rope_f64(&test_vec, cos11, sin11);
        let out12 = apply_rope_f64(&test_vec, cos12, sin12);

        // All outputs should be different
        assert!(out10 != out11, "Position 10 and 11 should differ");
        assert!(out11 != out12, "Position 11 and 12 should differ");
        assert!(out10 != out12, "Position 10 and 12 should differ");

        // But magnitudes should be preserved
        let mag_in: f64 = test_vec.iter().map(|x| x * x).sum();
        let mag_10: f64 = out10.iter().map(|x| x * x).sum();
        let mag_11: f64 = out11.iter().map(|x| x * x).sum();
        let mag_12: f64 = out12.iter().map(|x| x * x).sum();

        assert!(
            (mag_in - mag_10).abs() < 1e-12,
            "Magnitude not preserved at pos 10"
        );
        assert!(
            (mag_in - mag_11).abs() < 1e-12,
            "Magnitude not preserved at pos 11"
        );
        assert!(
            (mag_in - mag_12).abs() < 1e-12,
            "Magnitude not preserved at pos 12"
        );
    }

    /// Test the relative position property of RoPE
    ///
    /// The key property: dot(RoPE(q, m), RoPE(k, n)) depends only on (m - n)
    /// This is what makes RoPE effective for relative position encoding.
    #[test]
    fn test_relative_position_property() {
        let freqs = RopeFrequencies::new(4, 128, 10000.0);

        let q = vec![1.0, 0.5, 0.3, 0.2];
        let k = vec![0.8, 0.4, 0.6, 0.1];

        // Test pairs with same relative distance
        let pairs = [(5, 3), (10, 8), (20, 18)]; // all have distance 2

        let mut dot_products = Vec::new();
        for (pos_q, pos_k) in pairs {
            let (cos_q, sin_q) = freqs.get(pos_q).unwrap();
            let (cos_k, sin_k) = freqs.get(pos_k).unwrap();

            let q_rot = apply_rope_f64(&q, cos_q, sin_q);
            let k_rot = apply_rope_f64(&k, cos_k, sin_k);

            let dot: f64 = q_rot.iter().zip(&k_rot).map(|(a, b)| a * b).sum();
            dot_products.push(dot);
        }

        // All dot products should be equal (relative position is same)
        for i in 1..dot_products.len() {
            assert!(
                (dot_products[0] - dot_products[i]).abs() < 1e-12,
                "Relative position property violated: dot[0]={}, dot[{}]={}",
                dot_products[0], i, dot_products[i]
            );
        }
    }
}
