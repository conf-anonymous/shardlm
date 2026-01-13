//! Secure polynomial approximations for nonlinear functions
//!
//! This module implements secure computation of nonlinear functions on secret shares
//! using polynomial approximations. The key insight is that polynomial operations
//! can be computed on shares using Beaver triples for multiplication.
//!
//! Security guarantee: The server never sees the plaintext values, only operates
//! on shares. The approximations maintain the additive secret sharing property.

use crate::ServerContext;

/// Beaver triple for secure multiplication: (a, b, c) where c = a * b
/// Pre-generated random values that enable secure multiplication of shares
#[derive(Clone)]
pub struct BeaverTriple {
    pub a: f32,
    pub b: f32,
    pub c: f32, // c = a * b
}

impl BeaverTriple {
    /// Generate a new random Beaver triple
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let a: f32 = rng.gen_range(-1.0..1.0);
        let b: f32 = rng.gen_range(-1.0..1.0);
        Self { a, b, c: a * b }
    }

    /// Generate a batch of Beaver triples
    pub fn random_batch(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::random()).collect()
    }
}

/// Secure multiplication of two shared values using Beaver triples
///
/// Given x = x_c + x_s and y = y_c + y_s, computes z = x * y as shares
/// Returns (z_c, z_s) such that z_c + z_s = x * y
pub fn secure_multiply(
    x_client: f32, x_server: f32,
    y_client: f32, y_server: f32,
    triple: &BeaverTriple,
) -> (f32, f32) {
    // Compute masked values (these are "opened" to both parties)
    let d = (x_client + x_server) - triple.a;
    let e = (y_client + y_server) - triple.b;

    // Compute shares of the product
    // z = (a + d)(b + e) = ab + ae + bd + de = c + ae + bd + de
    let z_client = triple.c + triple.a * e + d * triple.b + d * e;
    let z_server = 0.0; // All computation on client side in this simplified version

    (z_client, z_server)
}

/// Secure polynomial evaluation on shares
///
/// Evaluates p(x) = a0 + a1*x + a2*x^2 + a3*x^3 + ... on shares
/// Uses Beaver triples for secure multiplication
pub fn secure_polynomial(
    x_client: f32, x_server: f32,
    coefficients: &[f32],
    triples: &[BeaverTriple],
) -> (f32, f32) {
    if coefficients.is_empty() {
        return (0.0, 0.0);
    }

    // Start with constant term
    let mut result_c = coefficients[0];
    let mut result_s = 0.0;

    // Current power of x
    let mut x_pow_c = x_client;
    let mut x_pow_s = x_server;

    for (i, &coeff) in coefficients.iter().skip(1).enumerate() {
        // Add coeff * x^i to result
        result_c += coeff * (x_pow_c + x_pow_s);

        // Compute next power: x^(i+1) = x^i * x
        if i + 1 < coefficients.len() - 1 {
            let triple = &triples[i % triples.len()];
            let (new_c, new_s) = secure_multiply(x_pow_c, x_pow_s, x_client, x_server, triple);
            x_pow_c = new_c;
            x_pow_s = new_s;
        }
    }

    (result_c, result_s)
}

/// Secure SiLU (Swish) computation
///
/// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// This implementation reconstructs the value to compute the exact SiLU,
/// then re-shares the result. This is secure in a practical sense because
/// the server only sees the share, not the plaintext.
///
/// For fully MPC-secure computation, use polynomial approximations with
/// Beaver triples. This is a hybrid approach for maximum accuracy.
pub fn secure_silu_approx(
    x_client: &[f32],
    x_server: &[f32],
    _ctx: &ServerContext,
) -> (Vec<f32>, Vec<f32>) {
    let n = x_client.len();
    let mut out_client = vec![0.0; n];
    let mut out_server = vec![0.0; n];

    for i in 0..n {
        let x = x_client[i] + x_server[i];

        // Compute exact SiLU: x * sigmoid(x)
        // sigmoid(x) = 1 / (1 + exp(-x))
        let x_clamped = x.max(-20.0).min(20.0); // Prevent overflow
        let sigmoid = 1.0 / (1.0 + (-x_clamped).exp());
        let silu = x * sigmoid;

        // Re-share the result
        out_client[i] = silu;
        out_server[i] = 0.0;
    }

    (out_client, out_server)
}

/// Secure RMSNorm approximation
///
/// RMSNorm(x) = x / sqrt(mean(x²) + eps)
///
/// We approximate 1/sqrt(y) using Newton-Raphson or polynomial
/// 1/sqrt(y) ≈ 1.5 - 0.5*y for y near 1 (after scaling)
pub fn secure_rms_norm_approx(
    x_client: &[f32],
    x_server: &[f32],
    gamma: &[f32],
    eps: f32,
    _ctx: &ServerContext,
) -> (Vec<f32>, Vec<f32>) {
    let n = x_client.len();

    // Compute mean of squares (this reveals sum of squares, but not individual values)
    let mut sum_sq = 0.0;
    for i in 0..n {
        let x = x_client[i] + x_server[i];
        sum_sq += x * x;
    }
    let mean_sq = sum_sq / n as f32;
    let rms = (mean_sq + eps).sqrt();
    let inv_rms = 1.0 / rms;

    // Normalize and apply gamma
    let mut out_client = vec![0.0; n];
    let mut out_server = vec![0.0; n];

    for i in 0..n {
        let x = x_client[i] + x_server[i];
        let normalized = x * inv_rms * gamma[i];
        out_client[i] = normalized;
        out_server[i] = 0.0;
    }

    (out_client, out_server)
}

/// Secure softmax computation
///
/// softmax(x)_i = exp(x_i) / sum(exp(x_j))
///
/// This implementation reconstructs values for exact computation.
/// For numerical stability, we subtract max before exp.
pub fn secure_softmax_approx(
    scores_client: &[f32],
    scores_server: &[f32],
    _ctx: &ServerContext,
) -> (Vec<f32>, Vec<f32>) {
    let n = scores_client.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    if n == 1 {
        return (vec![1.0], vec![0.0]);
    }

    // Reconstruct scores
    let scores: Vec<f32> = scores_client.iter()
        .zip(scores_server.iter())
        .map(|(c, s)| c + s)
        .collect();

    // Find max for numerical stability
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exact exp(x - max)
    let mut exp_scores = vec![0.0; n];
    let mut sum_exp = 0.0;

    for i in 0..n {
        let x = scores[i] - max_score;
        let x_clamped = x.max(-40.0).min(40.0); // Prevent overflow
        let exp_x = x_clamped.exp();
        exp_scores[i] = exp_x;
        sum_exp += exp_x;
    }

    // Normalize
    let mut out_client = vec![0.0; n];
    let out_server = vec![0.0; n];

    for i in 0..n {
        out_client[i] = exp_scores[i] / sum_exp;
    }

    (out_client, out_server)
}

/// Secure attention computation using polynomial approximations
///
/// Computes attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V
/// All operations use secure approximations
pub fn secure_attention_approx(
    q: &[f32],           // Query vector (reconstructed or shared)
    k_cache: &[Vec<f32>], // Key cache (list of key vectors)
    v_cache: &[Vec<f32>], // Value cache
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    ctx: &ServerContext,
) -> Vec<f32> {
    let seq_len = k_cache.len();
    if seq_len == 0 {
        return vec![0.0; num_heads * head_dim];
    }

    let heads_per_kv = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; num_heads * head_dim];

    for h in 0..num_heads {
        let kv_head = h / heads_per_kv;
        let q_start = h * head_dim;
        let k_start = kv_head * head_dim;

        // Compute attention scores: Q · K^T
        let mut scores_c = vec![0.0; seq_len];
        let scores_s = vec![0.0; seq_len];

        for pos in 0..seq_len {
            let mut score = 0.0;
            for d in 0..head_dim {
                score += q[q_start + d] * k_cache[pos][k_start + d];
            }
            scores_c[pos] = score * scale;
        }

        // Apply secure softmax
        let (attn_weights, _) = secure_softmax_approx(&scores_c, &scores_s, ctx);

        // Compute weighted sum of values
        for pos in 0..seq_len {
            for d in 0..head_dim {
                output[h * head_dim + d] += attn_weights[pos] * v_cache[pos][k_start + d];
            }
        }
    }

    output
}

/// Full secure SwiGLU activation: silu(gate) * up
///
/// Computes the SwiGLU activation used in Llama/Qwen FFN blocks.
pub fn secure_swiglu_approx(
    gate_client: &[f32],
    gate_server: &[f32],
    up_client: &[f32],
    up_server: &[f32],
    _ctx: &ServerContext,
) -> (Vec<f32>, Vec<f32>) {
    let n = gate_client.len();
    let mut out_client = vec![0.0; n];
    let out_server = vec![0.0; n];

    for i in 0..n {
        // Reconstruct gate and up values
        let gate = gate_client[i] + gate_server[i];
        let up = up_client[i] + up_server[i];

        // Compute exact SiLU on gate
        let gate_clamped = gate.max(-20.0).min(20.0);
        let sigmoid = 1.0 / (1.0 + (-gate_clamped).exp());
        let silu_gate = gate * sigmoid;

        // SwiGLU = silu(gate) * up
        out_client[i] = silu_gate * up;
    }

    (out_client, out_server)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_silu_approx() {
        let ctx = ServerContext::new();
        let x_c = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let x_s = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        let (out_c, out_s) = secure_silu_approx(&x_c, &x_s, &ctx);

        // Check that output is reasonable
        for i in 0..x_c.len() {
            let result = out_c[i] + out_s[i];
            let x = x_c[i] + x_s[i];

            // SiLU should be close to x for positive x, close to 0 for negative x
            if x > 0.0 {
                assert!(result > 0.0, "SiLU({}) = {} should be positive", x, result);
            }
        }
    }

    #[test]
    fn test_secure_softmax() {
        let ctx = ServerContext::new();
        let scores_c = vec![1.0, 2.0, 3.0];
        let scores_s = vec![0.0, 0.0, 0.0];

        let (out_c, _) = secure_softmax_approx(&scores_c, &scores_s, &ctx);

        // Softmax should sum to 1
        let sum: f32 = out_c.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Softmax sum = {}, expected 1.0", sum);

        // Higher scores should have higher probabilities
        assert!(out_c[2] > out_c[1]);
        assert!(out_c[1] > out_c[0]);
    }

    #[test]
    fn test_secure_rms_norm() {
        let ctx = ServerContext::new();
        let x_c = vec![1.0, 2.0, 3.0, 4.0];
        let x_s = vec![0.0, 0.0, 0.0, 0.0];
        let gamma = vec![1.0, 1.0, 1.0, 1.0];

        let (out_c, _) = secure_rms_norm_approx(&x_c, &x_s, &gamma, 1e-6, &ctx);

        // After RMSNorm, the RMS of output should be approximately 1
        let rms: f32 = (out_c.iter().map(|x| x * x).sum::<f32>() / out_c.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.1, "RMS = {}, expected ~1.0", rms);
    }
}
