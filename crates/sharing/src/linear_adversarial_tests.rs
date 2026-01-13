//! Adversarial security tests for secure linear layer
//!
//! These tests validate that the linear layer implementation correctly enforces
//! security properties against malicious inputs and protocol violations.

#[cfg(test)]
mod tests {
    use crate::error::SharingError;
    use crate::linear::{LinearClient, LinearServer, plaintext_linear, secure_linear};
    use crate::share::Share;
    use shardlm_fixed_point::{FixedVector, DEFAULT_SCALE};

    // ==================== Dimension Validation Tests ====================

    #[test]
    fn test_dimension_mismatch_input_rejected() {
        // Server with 4x2 matrix, input has wrong dimension (3 instead of 4)
        let weight = vec![0i32; 8]; // 4x2 matrix
        let server = LinearServer::new(weight, None, 4, 2, DEFAULT_SCALE).unwrap();

        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap(); // Length 3
        let (client, server_share) = LinearClient::new(&input);
        let request = client.generate_request();

        let result = server.handle_request(&request, &server_share);
        assert!(
            matches!(result, Err(SharingError::DimensionMismatch { expected: 4, got: 3 })),
            "Should reject dimension mismatch in client share"
        );
    }

    #[test]
    fn test_dimension_mismatch_server_share_rejected() {
        // Valid client request but mismatched server share
        let weight = vec![0i32; 8]; // 4x2 matrix
        let server = LinearServer::new(weight, None, 4, 2, DEFAULT_SCALE).unwrap();

        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let (client, _) = LinearClient::new(&input);
        let request = client.generate_request();

        // Wrong size server share
        let wrong_server_share = Share::random(3, DEFAULT_SCALE); // Length 3 instead of 4

        let result = server.handle_request(&request, &wrong_server_share);
        assert!(
            matches!(result, Err(SharingError::DimensionMismatch { expected: 4, got: 3 })),
            "Should reject dimension mismatch in server share"
        );
    }

    #[test]
    fn test_scale_mismatch_rejected() {
        // Client and server with different scales
        let weight = vec![0i32; 4]; // 2x2 matrix
        let server = LinearServer::new(weight, None, 2, 2, DEFAULT_SCALE).unwrap();

        // Create input with different scale
        let input = FixedVector::from_raw(vec![1, 2], DEFAULT_SCALE + 1); // Wrong scale
        let (client, server_share) = LinearClient::new(&input);
        let request = client.generate_request();

        let result = server.handle_request(&request, &server_share);
        assert!(
            matches!(result, Err(SharingError::ScaleMismatch { .. })),
            "Should reject scale mismatch"
        );
    }

    #[test]
    fn test_bias_dimension_mismatch_rejected() {
        // Bias with wrong dimension
        let weight = vec![0i32; 8]; // 4x2 matrix (out_features = 2)
        let bias = vec![0i32; 3]; // Wrong: should be 2

        let result = LinearServer::new(weight, Some(bias), 4, 2, DEFAULT_SCALE);
        assert!(
            matches!(result, Err(SharingError::DimensionMismatch { expected: 2, got: 3 })),
            "Should reject bias dimension mismatch"
        );
    }

    // ==================== Security Property Tests ====================

    #[test]
    fn test_server_never_sees_plaintext_input() {
        let input = FixedVector::from_f64_slice_default(&[42.0, -17.5, 3.14159, 2.71828]).unwrap();
        let (client, server_share) = LinearClient::new(&input);
        let request = client.generate_request();

        // Server receives client_share (X_c) and server_share (X_s)
        // Neither should equal the plaintext

        assert_ne!(
            request.client_share, input.data,
            "Client share X_c must not equal plaintext X"
        );
        assert_ne!(
            server_share.data, input.data,
            "Server share X_s must not equal plaintext X"
        );

        // But they must reconstruct correctly
        let reconstructed: Vec<i32> = request
            .client_share
            .iter()
            .zip(&server_share.data)
            .map(|(&c, &s)| c.wrapping_add(s))
            .collect();
        assert_eq!(
            reconstructed, input.data,
            "X_c + X_s must equal X"
        );
    }

    #[test]
    fn test_server_never_sees_plaintext_output() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = vec![1 << DEFAULT_SCALE; 16]; // 4x4 identity-ish

        let (output, client_share, server_share) =
            secure_linear(&input, &weight, None, 4, 4, DEFAULT_SCALE).unwrap();

        // Neither output share should equal the plaintext output
        assert_ne!(
            client_share, output.data,
            "Client output share Y_c must not equal plaintext Y"
        );
        assert_ne!(
            server_share, output.data,
            "Server output share Y_s must not equal plaintext Y"
        );

        // But they must reconstruct correctly
        let reconstructed: Vec<i32> = client_share
            .iter()
            .zip(&server_share)
            .map(|(&c, &s)| c.wrapping_add(s))
            .collect();
        assert_eq!(
            reconstructed, output.data,
            "Y_c + Y_s must equal Y"
        );
    }

    #[test]
    fn test_different_inputs_different_shares() {
        // Same plaintext with different random seeds should produce different shares
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();

        let (client1, server_share1) = LinearClient::new(&input);
        let (client2, server_share2) = LinearClient::new(&input);

        let request1 = client1.generate_request();
        let request2 = client2.generate_request();

        // Very high probability that shares differ (1 in 2^96 they match)
        assert_ne!(
            request1.client_share, request2.client_share,
            "Same input should produce different shares due to randomness"
        );
        assert_ne!(
            server_share1.data, server_share2.data,
            "Same input should produce different server shares"
        );
    }

    #[test]
    fn test_seeded_sharing_is_deterministic() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();

        let (client1, server_share1) = LinearClient::new_seeded(&input, 12345);
        let (client2, server_share2) = LinearClient::new_seeded(&input, 12345);

        let request1 = client1.generate_request();
        let request2 = client2.generate_request();

        assert_eq!(
            request1.client_share, request2.client_share,
            "Same seed should produce same client share"
        );
        assert_eq!(
            server_share1.data, server_share2.data,
            "Same seed should produce same server share"
        );
    }

    // ==================== Correctness Tests ====================

    #[test]
    fn test_secure_matches_plaintext_identity_matrix() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // 4x4 identity matrix
        let mut weight = vec![0i32; 16];
        for i in 0..4 {
            weight[i * 4 + i] = 1 << DEFAULT_SCALE;
        }

        let (secure_out, _, _) = secure_linear(&input, &weight, None, 4, 4, DEFAULT_SCALE).unwrap();
        let plain_out = plaintext_linear(&input, &weight, None, 4, 4, DEFAULT_SCALE).unwrap();

        for (s, p) in secure_out.data.iter().zip(&plain_out.data) {
            let diff = (*s as i64 - *p as i64).abs();
            assert!(diff <= 1, "Output mismatch: {} vs {}", s, p);
        }
    }

    #[test]
    fn test_secure_matches_plaintext_zero_matrix() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();

        // 3x2 zero matrix
        let weight = vec![0i32; 6];

        let (secure_out, _, _) = secure_linear(&input, &weight, None, 3, 2, DEFAULT_SCALE).unwrap();
        let plain_out = plaintext_linear(&input, &weight, None, 3, 2, DEFAULT_SCALE).unwrap();

        // Zero weights should give zero output
        assert!(secure_out.data.iter().all(|&x| x == 0));
        assert!(plain_out.data.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_bias_correctly_applied() {
        let input = FixedVector::from_f64_slice_default(&[0.0, 0.0]).unwrap(); // Zero input

        // 2x2 zero matrix
        let weight = vec![0i32; 4];

        // Bias of [1.0, 2.0]
        let bias = vec![1 << DEFAULT_SCALE, 2 << DEFAULT_SCALE];

        let (secure_out, _, _) =
            secure_linear(&input, &weight, Some(&bias), 2, 2, DEFAULT_SCALE).unwrap();

        // Output should equal bias (since X=0, W=0)
        let out_f64 = secure_out.to_f64_vec();
        assert!((out_f64[0] - 1.0).abs() < 0.01, "Bias[0] not applied correctly");
        assert!((out_f64[1] - 2.0).abs() < 0.01, "Bias[1] not applied correctly");
    }

    // ==================== Edge Cases ====================

    #[test]
    fn test_single_element_input() {
        let input = FixedVector::from_f64_slice_default(&[5.0]).unwrap();
        let weight = vec![2 << DEFAULT_SCALE]; // 1x1 matrix with value 2.0

        let (output, _, _) = secure_linear(&input, &weight, None, 1, 1, DEFAULT_SCALE).unwrap();

        let out_f64 = output.to_f64_vec();
        assert!(
            (out_f64[0] - 10.0).abs() < 0.1,
            "5.0 * 2.0 should be ~10.0, got {}", out_f64[0]
        );
    }

    #[test]
    fn test_negative_values() {
        let input = FixedVector::from_f64_slice_default(&[-1.0, -2.0]).unwrap();

        // Identity matrix
        let weight = vec![
            1 << DEFAULT_SCALE, 0,
            0, 1 << DEFAULT_SCALE,
        ];

        let (output, _, _) = secure_linear(&input, &weight, None, 2, 2, DEFAULT_SCALE).unwrap();

        let out_f64 = output.to_f64_vec();
        assert!((out_f64[0] - (-1.0)).abs() < 0.01);
        assert!((out_f64[1] - (-2.0)).abs() < 0.01);
    }

    #[test]
    fn test_overflow_wrapping() {
        // Use extreme values that would overflow without wrapping
        let input = FixedVector::from_raw(vec![i32::MAX, i32::MIN], DEFAULT_SCALE);

        // Identity matrix
        let weight = vec![
            1 << DEFAULT_SCALE, 0,
            0, 1 << DEFAULT_SCALE,
        ];

        // Should not panic, should wrap correctly
        let result = secure_linear(&input, &weight, None, 2, 2, DEFAULT_SCALE);
        assert!(result.is_ok(), "Should handle extreme values via wrapping");
    }

    // ==================== Protocol Ordering Tests ====================

    #[test]
    fn test_reconstruct_before_response_fails() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0]).unwrap();
        let (client, _) = LinearClient::new(&input);

        // Try to reconstruct before receiving response
        let dummy_server_share = vec![0i32; 2];
        let result = client.reconstruct(&dummy_server_share);

        assert!(
            matches!(result, Err(SharingError::NotReady)),
            "Reconstruct before response should fail"
        );
    }

    #[test]
    fn test_reconstruct_with_wrong_size_fails() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0]).unwrap();
        let weight = vec![0i32; 4]; // 2x2

        let (mut client, server_share) = LinearClient::new(&input);
        let server = LinearServer::new(weight, None, 2, 2, DEFAULT_SCALE).unwrap();

        let request = client.generate_request();
        let response = server.handle_request(&request, &server_share).unwrap();
        client.handle_response(&response).unwrap();

        // Try to reconstruct with wrong size
        let wrong_size = vec![0i32; 3]; // Should be 2
        let result = client.reconstruct(&wrong_size);

        assert!(
            matches!(result, Err(SharingError::DimensionMismatch { expected: 2, got: 3 })),
            "Reconstruct with wrong size should fail"
        );
    }
}
