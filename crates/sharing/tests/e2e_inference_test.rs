//! End-to-end secure inference test
//!
//! This test validates that:
//! 1. Secret sharing correctly splits values: X = X_c + X_s
//! 2. Linear operations on shares produce correct results
//! 3. Full forward pass matches plaintext computation
//! 4. Reconstruction produces the same output as non-shared computation

use shardlm_fixed_point::{FixedVector, DEFAULT_SCALE};
use shardlm_sharing::{SharedMatrix, SharedVector};

/// Test that shares reconstruct to the original value
#[test]
fn test_share_reconstruction_exact() {
    let values = vec![1.0, -2.5, 3.14159, 0.0, -100.0, 100.0];
    let plaintext = FixedVector::from_f64_slice_default(&values).unwrap();

    // Create shares
    let shared = SharedVector::from_plaintext(&plaintext);

    // Reconstruct
    let reconstructed = shared.reconstruct().unwrap();

    // Verify exact equality (in fixed-point representation)
    assert_eq!(
        plaintext.data, reconstructed.data,
        "Reconstruction should be exact"
    );
}

/// Test that share addition is correct: (a_c + b_c, a_s + b_s) reconstructs to a + b
#[test]
fn test_share_addition() {
    let a_vals = vec![1.0, 2.0, 3.0];
    let b_vals = vec![4.0, 5.0, 6.0];

    let a = FixedVector::from_f64_slice_default(&a_vals).unwrap();
    let b = FixedVector::from_f64_slice_default(&b_vals).unwrap();

    let shared_a = SharedVector::from_plaintext(&a);
    let shared_b = SharedVector::from_plaintext(&b);

    // Add shares
    let shared_sum = shared_a.add(&shared_b).unwrap();
    let reconstructed = shared_sum.reconstruct().unwrap();

    // Compare with plaintext addition
    let expected = a.add(&b).unwrap();

    assert_eq!(
        expected.data, reconstructed.data,
        "Shared addition should match plaintext"
    );
}

/// Test that matrix multiply on shares is correct: X @ W computed on shares matches plaintext
#[test]
fn test_secure_matrix_multiply_matches_plaintext() {
    // Create a known matrix and vector
    // W = [[1, 2, 3],
    //      [4, 5, 6]]
    // X = [1, 2]
    // Expected: Y = X @ W = [1*1 + 2*4, 1*2 + 2*5, 1*3 + 2*6] = [9, 12, 15]

    let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = SharedMatrix::from_f64(&w_data, 2, 3, DEFAULT_SCALE).unwrap();

    let x_vals = vec![1.0, 2.0];
    let x = FixedVector::from_f64_slice_default(&x_vals).unwrap();

    // Secure computation
    let shared_x = SharedVector::from_plaintext(&x);
    let shared_y = w.multiply_shared(&shared_x).unwrap();
    let y_secure = shared_y.reconstruct().unwrap();

    // Verify result
    let y_f64 = y_secure.to_f64_vec();
    assert!((y_f64[0] - 9.0).abs() < 0.01, "Expected 9, got {}", y_f64[0]);
    assert!(
        (y_f64[1] - 12.0).abs() < 0.01,
        "Expected 12, got {}",
        y_f64[1]
    );
    assert!(
        (y_f64[2] - 15.0).abs() < 0.01,
        "Expected 15, got {}",
        y_f64[2]
    );
}

/// Test that chained operations work correctly
/// Y = (X @ W1) @ W2
#[test]
fn test_chained_matrix_operations() {
    // W1: 4x3 matrix (input 4, output 3)
    // W2: 3x2 matrix (input 3, output 2)
    // X: 4-element vector
    // Y: 2-element vector

    let w1_data: Vec<f64> = (0..12).map(|i| (i as f64) * 0.1).collect();
    let w1 = SharedMatrix::from_f64(&w1_data, 4, 3, DEFAULT_SCALE).unwrap();

    let w2_data: Vec<f64> = (0..6).map(|i| (i as f64) * 0.1).collect();
    let w2 = SharedMatrix::from_f64(&w2_data, 3, 2, DEFAULT_SCALE).unwrap();

    let x_vals = vec![1.0, 2.0, 3.0, 4.0];
    let x = FixedVector::from_f64_slice_default(&x_vals).unwrap();

    // Secure chained computation
    let shared_x = SharedVector::from_plaintext(&x);
    let shared_h = w1.multiply_shared(&shared_x).unwrap();
    let shared_y = w2.multiply_shared(&shared_h).unwrap();
    let y_secure = shared_y.reconstruct().unwrap();

    // Plaintext computation for comparison
    // First multiply: H = X @ W1
    let h_expected = compute_matmul_plain(&x_vals, &w1_data, 4, 3);
    // Second multiply: Y = H @ W2
    let y_expected = compute_matmul_plain(&h_expected, &w2_data, 3, 2);

    let y_f64 = y_secure.to_f64_vec();

    for i in 0..2 {
        let diff = (y_f64[i] - y_expected[i]).abs();
        assert!(
            diff < 0.1,
            "Chained matmul mismatch at {}: expected {}, got {} (diff {})",
            i,
            y_expected[i],
            y_f64[i],
            diff
        );
    }
}

/// Test bias addition in secure computation
#[test]
fn test_secure_linear_with_bias() {
    // Y = X @ W + b
    let w_data = vec![1.0, 0.0, 0.0, 1.0]; // Identity 2x2
    let w = SharedMatrix::from_f64(&w_data, 2, 2, DEFAULT_SCALE).unwrap();

    let x_vals = vec![3.0, 4.0];
    let x = FixedVector::from_f64_slice_default(&x_vals).unwrap();

    let bias_vals = vec![10.0, 20.0];
    let bias = FixedVector::from_f64_slice_default(&bias_vals).unwrap();

    // Secure computation
    let shared_x = SharedVector::from_plaintext(&x);
    let shared_y = w.multiply_shared(&shared_x).unwrap();
    let shared_y_bias = SharedMatrix::add_bias(&shared_y, &bias).unwrap();
    let y_secure = shared_y_bias.reconstruct().unwrap();

    // Expected: [3+10, 4+20] = [13, 24]
    let y_f64 = y_secure.to_f64_vec();
    assert!(
        (y_f64[0] - 13.0).abs() < 0.01,
        "Expected 13, got {}",
        y_f64[0]
    );
    assert!(
        (y_f64[1] - 24.0).abs() < 0.01,
        "Expected 24, got {}",
        y_f64[1]
    );
}

/// Test that server share alone reveals nothing about input
#[test]
fn test_server_learns_nothing() {
    let secret_values = vec![12345.0, -9999.0, 0.001];
    let plaintext = FixedVector::from_f64_slice_default(&secret_values).unwrap();

    let shared = SharedVector::from_plaintext(&plaintext);

    // Server only sees server_share
    // This should be indistinguishable from random noise
    // We test that it's not equal to the plaintext
    assert_ne!(
        shared.server_share.data, plaintext.data,
        "Server share should not equal plaintext"
    );

    // Also verify that combining with wrong share gives garbage
    let wrong_client = SharedVector::from_plaintext(&plaintext);
    let wrong_result_data: Vec<i32> = wrong_client
        .client_share
        .data
        .iter()
        .zip(&shared.server_share.data)
        .map(|(&c, &s)| c.wrapping_add(s))
        .collect();

    assert_ne!(
        wrong_result_data, plaintext.data,
        "Wrong shares should not reconstruct to plaintext"
    );
}

/// Test large dimension computation (simulating real model sizes)
#[test]
fn test_large_dimension_computation() {
    let hidden_size = 256;
    let output_size = 64;

    // Create random-ish matrix
    let w_data: Vec<f64> = (0..hidden_size * output_size)
        .map(|i| ((i % 100) as f64 - 50.0) / 100.0)
        .collect();
    let w = SharedMatrix::from_f64(&w_data, hidden_size, output_size, DEFAULT_SCALE).unwrap();

    // Create input vector
    let x_vals: Vec<f64> = (0..hidden_size)
        .map(|i| ((i % 20) as f64 - 10.0) / 10.0)
        .collect();
    let x = FixedVector::from_f64_slice_default(&x_vals).unwrap();

    // Secure computation
    let shared_x = SharedVector::from_plaintext(&x);
    let shared_y = w.multiply_shared(&shared_x).unwrap();
    let y_secure = shared_y.reconstruct().unwrap();

    // Plaintext computation
    let y_expected = compute_matmul_plain(&x_vals, &w_data, hidden_size, output_size);

    let y_f64 = y_secure.to_f64_vec();

    // Check a few values (not all, for speed)
    for i in [0, 10, 32, 63] {
        let diff = (y_f64[i] - y_expected[i]).abs();
        assert!(
            diff < 0.5,
            "Large matmul mismatch at {}: expected {}, got {} (diff {})",
            i,
            y_expected[i],
            y_f64[i],
            diff
        );
    }
}

/// Simulate a simplified forward pass: embed -> linear1 -> linear2 -> logits
#[test]
fn test_simplified_forward_pass() {
    let embed_dim = 32;
    let hidden_dim = 16;
    let vocab_size = 10;

    // Simulated embedding (what OT returns)
    let embedding_vals: Vec<f64> = (0..embed_dim).map(|i| (i as f64) / 32.0).collect();
    let embedding = FixedVector::from_f64_slice_default(&embedding_vals).unwrap();

    // Layer 1: embed_dim -> hidden_dim
    let w1_data: Vec<f64> = (0..embed_dim * hidden_dim)
        .map(|i| ((i % 10) as f64 - 5.0) / 50.0)
        .collect();
    let w1 = SharedMatrix::from_f64(&w1_data, embed_dim, hidden_dim, DEFAULT_SCALE).unwrap();

    // Layer 2: hidden_dim -> vocab_size (lm_head)
    let w2_data: Vec<f64> = (0..hidden_dim * vocab_size)
        .map(|i| ((i % 10) as f64 - 5.0) / 50.0)
        .collect();
    let w2 = SharedMatrix::from_f64(&w2_data, hidden_dim, vocab_size, DEFAULT_SCALE).unwrap();

    // === Secure computation ===
    // Step 1: Create shares of embedding
    let shared_embed = SharedVector::from_plaintext(&embedding);

    // Step 2: Apply layer 1
    let shared_hidden = w1.multiply_shared(&shared_embed).unwrap();

    // Step 3: Apply layer 2 (lm_head)
    let shared_logits = w2.multiply_shared(&shared_hidden).unwrap();

    // Step 4: Reconstruct logits
    let logits_secure = shared_logits.reconstruct().unwrap();

    // === Plaintext computation ===
    let hidden_plain = compute_matmul_plain(&embedding_vals, &w1_data, embed_dim, hidden_dim);
    let logits_plain = compute_matmul_plain(&hidden_plain, &w2_data, hidden_dim, vocab_size);

    // === Verify ===
    let logits_f64 = logits_secure.to_f64_vec();

    for i in 0..vocab_size {
        let diff = (logits_f64[i] - logits_plain[i]).abs();
        assert!(
            diff < 0.5,
            "Forward pass mismatch at vocab {}: expected {}, got {} (diff {})",
            i,
            logits_plain[i],
            logits_f64[i],
            diff
        );
    }

    // Verify argmax gives same result
    let secure_argmax = logits_f64
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let plain_argmax = logits_plain
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    assert_eq!(
        secure_argmax, plain_argmax,
        "Argmax should match: secure={}, plain={}",
        secure_argmax, plain_argmax
    );
}

// Helper function for plaintext matmul
fn compute_matmul_plain(x: &[f64], w: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut result = vec![0.0; cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j] += x[i] * w[i * cols + j];
        }
    }
    result
}
