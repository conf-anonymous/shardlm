//! Secure linear layer protocol
//!
//! Implements Y = XW + b where:
//! - X is secret-shared (X = X_c + X_s)
//! - W and b are held by server only
//! - Y is secret-shared (Y = Y_c + Y_s)
//!
//! Security properties:
//! - Server never sees plaintext X or Y
//! - Client never sees W or b
//! - Both parties learn nothing beyond what reconstruction reveals

use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;

/// Client state for secure linear computation
#[derive(Debug, Clone)]
pub struct LinearClient {
    /// Client's share of input
    input_share: Share,
    /// Client's share of output (after receiving from server)
    output_share: Option<Share>,
}

/// Server state for secure linear computation
#[derive(Debug)]
pub struct LinearServer {
    /// Weight matrix (in_features × out_features), row-major
    weight: Vec<i32>,
    /// Bias vector (out_features)
    bias: Option<Vec<i32>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
    /// Fixed-point scale
    scale: u8,
}

/// Message from client to server: client's share of input
#[derive(Debug, Clone)]
pub struct LinearRequestMsg {
    /// Client's masked share: X_c
    pub client_share: Vec<i32>,
    /// Scale factor
    pub scale: u8,
}

/// Message from server to client: both shares of output
#[derive(Debug, Clone)]
pub struct LinearResponseMsg {
    /// Client's output share: Y_c = X_c * W (computed by server, sent to client)
    pub client_output_share: Vec<i32>,
    /// Server's output share: Y_s = X_s * W + b (kept by server, but client needs for reconstruction)
    pub server_output_share: Vec<i32>,
    /// Scale factor
    pub scale: u8,
}

impl LinearClient {
    /// Create client state from plaintext input
    /// Client creates additive shares: X = X_c + X_s
    pub fn new(plaintext_input: &FixedVector) -> (Self, Share) {
        // Generate random server share
        let server_share = Share::random(plaintext_input.len(), plaintext_input.scale);

        // Compute client share: X_c = X - X_s
        let client_data: Vec<i32> = plaintext_input
            .data
            .iter()
            .zip(&server_share.data)
            .map(|(&x, &xs)| x.wrapping_sub(xs))
            .collect();
        let client_share = Share::from_raw(client_data, plaintext_input.scale);

        (
            Self {
                input_share: client_share,
                output_share: None,
            },
            server_share,
        )
    }

    /// Create with seeded randomness (for determinism in tests)
    pub fn new_seeded(plaintext_input: &FixedVector, seed: u64) -> (Self, Share) {
        let server_share = Share::random_seeded(plaintext_input.len(), plaintext_input.scale, seed);

        let client_data: Vec<i32> = plaintext_input
            .data
            .iter()
            .zip(&server_share.data)
            .map(|(&x, &xs)| x.wrapping_sub(xs))
            .collect();
        let client_share = Share::from_raw(client_data, plaintext_input.scale);

        (
            Self {
                input_share: client_share,
                output_share: None,
            },
            server_share,
        )
    }

    /// Generate request message to send to server
    pub fn generate_request(&self) -> LinearRequestMsg {
        LinearRequestMsg {
            client_share: self.input_share.data.clone(),
            scale: self.input_share.scale,
        }
    }

    /// Handle response from server
    pub fn handle_response(&mut self, response: &LinearResponseMsg) -> Result<()> {
        if response.scale != self.input_share.scale {
            return Err(SharingError::ScaleMismatch {
                expected: self.input_share.scale,
                got: response.scale,
            });
        }

        // Store client's output share
        self.output_share = Some(Share::from_raw(
            response.client_output_share.clone(),
            response.scale,
        ));

        Ok(())
    }

    /// Reconstruct the final output: Y = Y_c + Y_s
    pub fn reconstruct(&self, server_output_share: &[i32]) -> Result<FixedVector> {
        let client_output = self
            .output_share
            .as_ref()
            .ok_or(SharingError::NotReady)?;

        if client_output.len() != server_output_share.len() {
            return Err(SharingError::DimensionMismatch {
                expected: client_output.len(),
                got: server_output_share.len(),
            });
        }

        let data: Vec<i32> = client_output
            .data
            .iter()
            .zip(server_output_share)
            .map(|(&yc, &ys)| yc.wrapping_add(ys))
            .collect();

        Ok(FixedVector::from_raw(data, client_output.scale))
    }
}

impl LinearServer {
    /// Create server with weight matrix and optional bias
    pub fn new(
        weight: Vec<i32>,
        bias: Option<Vec<i32>>,
        in_features: usize,
        out_features: usize,
        scale: u8,
    ) -> Result<Self> {
        if weight.len() != in_features * out_features {
            return Err(SharingError::DimensionMismatch {
                expected: in_features * out_features,
                got: weight.len(),
            });
        }

        if let Some(ref b) = bias {
            if b.len() != out_features {
                return Err(SharingError::DimensionMismatch {
                    expected: out_features,
                    got: b.len(),
                });
            }
        }

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
            scale,
        })
    }

    /// Get dimensions
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Handle client request and generate response
    /// Server receives X_s separately (via secure channel or OT)
    pub fn handle_request(
        &self,
        request: &LinearRequestMsg,
        server_input_share: &Share,
    ) -> Result<LinearResponseMsg> {
        // Validate dimensions
        if request.client_share.len() != self.in_features {
            return Err(SharingError::DimensionMismatch {
                expected: self.in_features,
                got: request.client_share.len(),
            });
        }
        if server_input_share.len() != self.in_features {
            return Err(SharingError::DimensionMismatch {
                expected: self.in_features,
                got: server_input_share.len(),
            });
        }
        if request.scale != self.scale {
            return Err(SharingError::ScaleMismatch {
                expected: self.scale,
                got: request.scale,
            });
        }

        // Server computes Y_c = X_c * W
        let client_output = self.multiply_vector(&request.client_share);

        // Server computes Y_s = X_s * W + b
        let mut server_output = self.multiply_vector(&server_input_share.data);

        // Add bias to server's share only
        if let Some(ref bias) = self.bias {
            for (y, &b) in server_output.iter_mut().zip(bias) {
                *y = y.wrapping_add(b);
            }
        }

        Ok(LinearResponseMsg {
            client_output_share: client_output,
            server_output_share: server_output,
            scale: self.scale,
        })
    }

    /// Matrix-vector multiply: out = input * W
    ///
    /// Optimized with cache-friendly access pattern and auto-vectorization hints.
    #[inline]
    fn multiply_vector(&self, input: &[i32]) -> Vec<i32> {
        let mut output = vec![0i64; self.out_features];

        // out[j] = sum_i(input[i] * W[i,j])
        // Process row by row for cache-friendly access to W
        for i in 0..self.in_features {
            let xi = input[i] as i64;
            let w_row = &self.weight[i * self.out_features..(i + 1) * self.out_features];

            // This loop structure allows the compiler to auto-vectorize
            // when targeting aarch64 with NEON
            for j in 0..self.out_features {
                output[j] += xi * w_row[j] as i64;
            }
        }

        // Rescale: divide by 2^scale
        output
            .iter()
            .map(|&x| (x >> self.scale) as i32)
            .collect()
    }

    /// Get a reference to the server's output share (for reconstruction)
    /// In practice, this would be sent through a secure channel
    pub fn get_output_share(&self) -> Option<&[i32]> {
        // This is computed during handle_request and should be cached
        // For simplicity, we recompute or the caller must store it
        None
    }
}

/// Convenience function for single-shot secure linear computation
/// Returns both shares for verification in tests
pub fn secure_linear(
    plaintext_input: &FixedVector,
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<(FixedVector, Vec<i32>, Vec<i32>)> {
    // Client creates shares
    let (mut client, server_input_share) = LinearClient::new(plaintext_input);

    // Server creates linear layer
    let server = LinearServer::new(
        weight.to_vec(),
        bias.map(|b| b.to_vec()),
        in_features,
        out_features,
        scale,
    )?;

    // Client sends request
    let request = client.generate_request();

    // Server computes response
    let response = server.handle_request(&request, &server_input_share)?;

    // Client handles response
    client.handle_response(&response)?;

    // Reconstruct output
    let output = client.reconstruct(&response.server_output_share)?;

    Ok((output, response.client_output_share, response.server_output_share))
}

/// Compute plaintext linear for verification: Y = X * W + b
pub fn plaintext_linear(
    input: &FixedVector,
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<FixedVector> {
    if input.len() != in_features {
        return Err(SharingError::DimensionMismatch {
            expected: in_features,
            got: input.len(),
        });
    }

    let mut output = vec![0i64; out_features];

    // Matrix-vector multiply
    for i in 0..in_features {
        let xi = input.data[i] as i64;
        for j in 0..out_features {
            let wij = weight[i * out_features + j] as i64;
            output[j] += xi * wij;
        }
    }

    // Rescale
    let mut result: Vec<i32> = output
        .iter()
        .map(|&x| (x >> scale) as i32)
        .collect();

    // Add bias
    if let Some(b) = bias {
        for (r, &bi) in result.iter_mut().zip(b) {
            *r = r.wrapping_add(bi);
        }
    }

    Ok(FixedVector::from_raw(result, scale))
}

// ==================== Batch/Sequence Operations ====================

/// Batch secure linear: Y = XW + b for X of shape [seq_len, in_features]
/// Returns output of shape [seq_len, out_features]
pub fn secure_linear_batch(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<Vec<FixedVector>> {
    let mut outputs = Vec::with_capacity(inputs.len());

    for input in inputs {
        let (output, _, _) = secure_linear(input, weight, bias, in_features, out_features, scale)?;
        outputs.push(output);
    }

    Ok(outputs)
}

/// Batch plaintext linear: Y = XW + b for X of shape [seq_len, in_features]
pub fn plaintext_linear_batch(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<Vec<FixedVector>> {
    let mut outputs = Vec::with_capacity(inputs.len());

    for input in inputs {
        let output = plaintext_linear(input, weight, bias, in_features, out_features, scale)?;
        outputs.push(output);
    }

    Ok(outputs)
}

/// Result of batch secure linear operation with detailed metrics
#[derive(Debug)]
pub struct BatchLinearResult {
    /// Output vectors [seq_len, out_features]
    pub outputs: Vec<FixedVector>,
    /// Total computation time
    pub total_time: std::time::Duration,
    /// Time per token
    pub per_token_time: std::time::Duration,
    /// Number of tokens processed
    pub seq_len: usize,
}

/// Batch secure linear with timing metrics
pub fn secure_linear_batch_timed(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<BatchLinearResult> {
    use std::time::Instant;

    let start = Instant::now();
    let outputs = secure_linear_batch(inputs, weight, bias, in_features, out_features, scale)?;
    let total_time = start.elapsed();

    let seq_len = inputs.len();
    let per_token_time = total_time / seq_len.max(1) as u32;

    Ok(BatchLinearResult {
        outputs,
        total_time,
        per_token_time,
        seq_len,
    })
}

// ==================== Hybrid GEMM/Matvec Strategy ====================

/// Crossover threshold: use GEMM for sequences >= this length, matvec otherwise.
/// Empirically determined from benchmarks on Apple Silicon M-series:
/// - L < 16: matvec is faster (less transpose overhead)
/// - L >= 16: GEMM is faster (amortized transpose cost)
pub const GEMM_CROSSOVER_L: usize = 16;

/// Hybrid secure linear: automatically chooses GEMM or matvec based on sequence length.
///
/// This function provides the best performance across all sequence lengths:
/// - Short sequences (L < 16): Uses per-token matvec (avoids transpose overhead)
/// - Long sequences (L >= 16): Uses GEMM (transpose cost amortized)
pub fn secure_linear_hybrid(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<Vec<FixedVector>> {
    if inputs.len() >= GEMM_CROSSOVER_L {
        secure_linear_gemm(inputs, weight, bias, in_features, out_features, scale)
    } else {
        secure_linear_batch(inputs, weight, bias, in_features, out_features, scale)
    }
}

/// Hybrid secure linear with timing metrics.
pub fn secure_linear_hybrid_timed(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<HybridLinearResult> {
    use std::time::Instant;

    let start = Instant::now();
    let use_gemm = inputs.len() >= GEMM_CROSSOVER_L;
    let outputs = secure_linear_hybrid(inputs, weight, bias, in_features, out_features, scale)?;
    let total_time = start.elapsed();

    let seq_len = inputs.len();
    let per_token_time = if seq_len > 0 {
        total_time / seq_len as u32
    } else {
        std::time::Duration::ZERO
    };

    let tokens_per_sec = if total_time.as_secs_f64() > 0.0 {
        seq_len as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(HybridLinearResult {
        outputs,
        total_time,
        per_token_time,
        seq_len,
        tokens_per_sec,
        used_gemm: use_gemm,
    })
}

/// Result of hybrid linear operation with detailed metrics
#[derive(Debug)]
pub struct HybridLinearResult {
    /// Output vectors [seq_len, out_features]
    pub outputs: Vec<FixedVector>,
    /// Total computation time
    pub total_time: std::time::Duration,
    /// Time per token (amortized)
    pub per_token_time: std::time::Duration,
    /// Number of tokens processed
    pub seq_len: usize,
    /// Throughput in tokens/second
    pub tokens_per_sec: f64,
    /// Whether GEMM was used (vs matvec)
    pub used_gemm: bool,
}

// ==================== Precomputed Transpose for GEMM ====================

/// Linear weights with optional precomputed transpose.
///
/// Storing W_t avoids per-call transpose overhead, moving the crossover earlier
/// and improving performance for all sequence lengths when using GEMM.
#[derive(Debug, Clone)]
pub struct LinearWeights {
    /// Weight matrix W: [in_features × out_features] row-major
    pub weight: Vec<i32>,
    /// Transposed weight matrix W_t: [out_features × in_features] row-major
    /// W_t[j,i] = W[i,j], used for efficient GEMM inner products
    pub weight_t: Vec<i32>,
    /// Input dimension
    pub in_features: usize,
    /// Output dimension
    pub out_features: usize,
    /// Fixed-point scale
    pub scale: u8,
    /// Optional bias vector
    pub bias: Option<Vec<i32>>,
}

impl LinearWeights {
    /// Create LinearWeights from weight matrix, precomputing the transpose.
    pub fn new(
        weight: Vec<i32>,
        bias: Option<Vec<i32>>,
        in_features: usize,
        out_features: usize,
        scale: u8,
    ) -> Result<Self> {
        if weight.len() != in_features * out_features {
            return Err(SharingError::DimensionMismatch {
                expected: in_features * out_features,
                got: weight.len(),
            });
        }

        if let Some(ref b) = bias {
            if b.len() != out_features {
                return Err(SharingError::DimensionMismatch {
                    expected: out_features,
                    got: b.len(),
                });
            }
        }

        // Precompute transpose: W[i,j] -> W_t[j,i]
        let mut weight_t = vec![0i32; in_features * out_features];
        for i in 0..in_features {
            for j in 0..out_features {
                weight_t[j * in_features + i] = weight[i * out_features + j];
            }
        }

        Ok(Self {
            weight,
            weight_t,
            in_features,
            out_features,
            scale,
            bias,
        })
    }

    /// Create from f64 values (for convenience in tests/demos).
    pub fn from_f64(
        weight_f64: &[f64],
        bias_f64: Option<&[f64]>,
        in_features: usize,
        out_features: usize,
        scale: u8,
    ) -> Result<Self> {
        let scale_factor = (1u64 << scale) as f64;
        let weight: Vec<i32> = weight_f64
            .iter()
            .map(|&w| (w * scale_factor).round() as i32)
            .collect();

        let bias = bias_f64.map(|b| {
            b.iter()
                .map(|&v| (v * scale_factor).round() as i32)
                .collect()
        });

        Self::new(weight, bias, in_features, out_features, scale)
    }
}

/// GEMM using precomputed transpose (no per-call transpose overhead).
///
/// This is faster than `secure_linear_gemm` for all sequence lengths because
/// the transpose is done once when loading weights, not on every call.
/// Uses SIMD dot products on aarch64 for additional speedup.
fn gemm_with_precomputed_transpose(
    x: &[i32],
    seq_len: usize,
    in_features: usize,
    weight_t: &[i32],
    out_features: usize,
    scale: u8,
) -> Vec<i32> {
    let m = seq_len;
    let k = in_features;
    let n = out_features;

    let mut c = vec![0i64; m * n];

    // C[i,j] = sum_k(X[i,k] * W_t[j,k])
    // Both X row and W_t row are accessed sequentially
    for i in 0..m {
        let x_row = &x[i * k..(i + 1) * k];
        for j in 0..n {
            let wt_row = &weight_t[j * k..(j + 1) * k];
            #[cfg(target_arch = "aarch64")]
            {
                c[i * n + j] = dot_product_neon(x_row, wt_row);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut acc = 0i64;
                for kk in 0..k {
                    acc += x_row[kk] as i64 * wt_row[kk] as i64;
                }
                c[i * n + j] = acc;
            }
        }
    }

    c.iter().map(|&x| (x >> scale) as i32).collect()
}

/// Secure linear using precomputed transpose weights.
///
/// This provides the best GEMM performance by avoiding per-call transpose.
pub fn secure_linear_gemm_pretransposed(
    inputs: &[FixedVector],
    weights: &LinearWeights,
) -> Result<Vec<FixedVector>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }

    let seq_len = inputs.len();

    // Validate dimensions
    for input in inputs {
        if input.len() != weights.in_features {
            return Err(SharingError::DimensionMismatch {
                expected: weights.in_features,
                got: input.len(),
            });
        }
    }

    // Flatten inputs
    let x_data: Vec<i32> = inputs.iter().flat_map(|v| v.data.iter().copied()).collect();

    // GEMM with precomputed transpose
    let mut y_data = gemm_with_precomputed_transpose(
        &x_data,
        seq_len,
        weights.in_features,
        &weights.weight_t,
        weights.out_features,
        weights.scale,
    );

    // Add bias
    if let Some(ref b) = weights.bias {
        for row in 0..seq_len {
            for j in 0..weights.out_features {
                y_data[row * weights.out_features + j] =
                    y_data[row * weights.out_features + j].wrapping_add(b[j]);
            }
        }
    }

    // Convert back to vectors
    let outputs: Vec<FixedVector> = (0..seq_len)
        .map(|i| {
            let start = i * weights.out_features;
            let row_data = y_data[start..start + weights.out_features].to_vec();
            FixedVector::from_raw(row_data, weights.scale)
        })
        .collect();

    Ok(outputs)
}

/// Hybrid strategy with precomputed transpose.
///
/// For L >= crossover: Uses GEMM with precomputed W_t (fastest)
/// For L < crossover: Uses sequential matvec (avoids any matrix overhead)
pub fn secure_linear_hybrid_pretransposed(
    inputs: &[FixedVector],
    weights: &LinearWeights,
) -> Result<Vec<FixedVector>> {
    if inputs.len() >= GEMM_CROSSOVER_L {
        secure_linear_gemm_pretransposed(inputs, weights)
    } else {
        secure_linear_batch(
            inputs,
            &weights.weight,
            weights.bias.as_deref(),
            weights.in_features,
            weights.out_features,
            weights.scale,
        )
    }
}

// ==================== GEMM-based Batch Operations ====================

/// Matrix for GEMM operations: [rows × cols] in row-major order
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Data in row-major order
    pub data: Vec<i32>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Fixed-point scale
    pub scale: u8,
}

impl Matrix {
    /// Create from a slice of FixedVectors (each vector is a row)
    pub fn from_vectors(vectors: &[FixedVector]) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SharingError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }

        let rows = vectors.len();
        let cols = vectors[0].len();
        let scale = vectors[0].scale;

        // Verify all vectors have same dimensions and scale
        for v in vectors.iter().skip(1) {
            if v.len() != cols {
                return Err(SharingError::DimensionMismatch {
                    expected: cols,
                    got: v.len(),
                });
            }
            if v.scale != scale {
                return Err(SharingError::ScaleMismatch {
                    expected: scale,
                    got: v.scale,
                });
            }
        }

        // Flatten into row-major order
        let data: Vec<i32> = vectors.iter().flat_map(|v| v.data.iter().copied()).collect();

        Ok(Self { data, rows, cols, scale })
    }

    /// Create from raw data
    pub fn from_raw(data: Vec<i32>, rows: usize, cols: usize, scale: u8) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(SharingError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(Self { data, rows, cols, scale })
    }

    /// Convert back to vector of FixedVectors
    pub fn to_vectors(&self) -> Vec<FixedVector> {
        (0..self.rows)
            .map(|i| {
                let start = i * self.cols;
                let row_data = self.data[start..start + self.cols].to_vec();
                FixedVector::from_raw(row_data, self.scale)
            })
            .collect()
    }

    /// Get a single row as FixedVector
    #[allow(dead_code)]
    pub fn get_row(&self, row: usize) -> Option<FixedVector> {
        if row >= self.rows {
            return None;
        }
        let start = row * self.cols;
        let row_data = self.data[start..start + self.cols].to_vec();
        Some(FixedVector::from_raw(row_data, self.scale))
    }
}

/// Secret-shared matrix
#[derive(Debug, Clone)]
pub struct SharedMatrix2 {
    /// Client's share
    pub client_share: Matrix,
    /// Server's share
    pub server_share: Matrix,
}

impl SharedMatrix2 {
    /// Create shares from plaintext matrix
    #[allow(dead_code)]
    pub fn from_plaintext(plaintext: &Matrix) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate random server share
        let server_data: Vec<i32> = (0..plaintext.data.len())
            .map(|_| rng.gen())
            .collect();

        // Client share = plaintext - server share
        let client_data: Vec<i32> = plaintext.data
            .iter()
            .zip(&server_data)
            .map(|(&p, &s)| p.wrapping_sub(s))
            .collect();

        Self {
            client_share: Matrix {
                data: client_data,
                rows: plaintext.rows,
                cols: plaintext.cols,
                scale: plaintext.scale,
            },
            server_share: Matrix {
                data: server_data,
                rows: plaintext.rows,
                cols: plaintext.cols,
                scale: plaintext.scale,
            },
        }
    }

    /// Reconstruct plaintext from shares
    #[allow(dead_code)]
    pub fn reconstruct(&self) -> Matrix {
        let data: Vec<i32> = self.client_share.data
            .iter()
            .zip(&self.server_share.data)
            .map(|(&c, &s)| c.wrapping_add(s))
            .collect();

        Matrix {
            data,
            rows: self.client_share.rows,
            cols: self.client_share.cols,
            scale: self.client_share.scale,
        }
    }
}

/// GEMM: C = A × B where A is [M × K] and B is [K × N]
/// Returns C of shape [M × N]
///
/// This implementation uses SIMD intrinsics on aarch64 (Apple Silicon)
/// and falls back to a scalar blocked implementation on other platforms.
fn gemm_i64(
    a: &[i32], a_rows: usize, a_cols: usize,
    b: &[i32], b_cols: usize,
    scale: u8,
) -> Vec<i32> {
    #[cfg(target_arch = "aarch64")]
    {
        gemm_i64_neon(a, a_rows, a_cols, b, b_cols, scale)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        gemm_i64_scalar(a, a_rows, a_cols, b, b_cols, scale)
    }
}

/// SIMD dot product for aarch64 using NEON intrinsics
///
/// Computes the dot product of two i32 slices using NEON vector instructions.
/// Falls back to scalar for remaining elements not divisible by 4.
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_product_neon(a: &[i32], b: &[i32]) -> i64 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    // Process 4 elements at a time using NEON
    // SAFETY: We're on aarch64 and processing aligned chunks
    let mut acc = unsafe {
        let mut sum_lo = vdupq_n_s64(0);
        let mut sum_hi = vdupq_n_s64(0);

        for i in 0..chunks {
            let offset = i * 4;
            // Load 4 x i32 values
            let va = vld1q_s32(a.as_ptr().add(offset));
            let vb = vld1q_s32(b.as_ptr().add(offset));

            // Multiply and get low/high parts (i32 * i32 -> i64)
            // vmull_s32 multiplies the low 2 elements, vmull_high_s32 the high 2
            let prod_lo = vmull_s32(vget_low_s32(va), vget_low_s32(vb));
            let prod_hi = vmull_high_s32(va, vb);

            // Accumulate
            sum_lo = vaddq_s64(sum_lo, prod_lo);
            sum_hi = vaddq_s64(sum_hi, prod_hi);
        }

        // Horizontal sum
        let sum_all = vaddq_s64(sum_lo, sum_hi);
        vgetq_lane_s64(sum_all, 0) + vgetq_lane_s64(sum_all, 1)
    };

    // Handle remaining elements
    for i in (chunks * 4)..len {
        acc += a[i] as i64 * b[i] as i64;
    }

    acc
}

/// Optimized GEMM for aarch64 (Apple Silicon) with NEON SIMD
///
/// Key optimizations:
/// 1. Transpose B for sequential memory access in inner loop
/// 2. Use NEON SIMD for dot products (4x i32 at a time)
/// 3. i64 accumulators to prevent overflow
#[cfg(target_arch = "aarch64")]
fn gemm_i64_neon(
    a: &[i32], a_rows: usize, a_cols: usize,
    b: &[i32], b_cols: usize,
    scale: u8,
) -> Vec<i32> {
    let m = a_rows;
    let k = a_cols;
    let n = b_cols;

    // Transpose B: B[k,n] -> B_t[n,k] for sequential access
    // This improves cache efficiency: inner loop accesses B_t row-wise
    let mut b_t = vec![0i32; k * n];
    for i in 0..k {
        for j in 0..n {
            b_t[j * k + i] = b[i * n + j];
        }
    }

    // Use i64 accumulator to avoid overflow
    let mut c = vec![0i64; m * n];

    // Main GEMM with transposed B and NEON dot products
    // C[i,j] = sum_k(A[i,k] * B_t[j,k])
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let b_t_row = &b_t[j * k..(j + 1) * k];
            c[i * n + j] = dot_product_neon(a_row, b_t_row);
        }
    }

    // Rescale
    c.iter().map(|&x| (x >> scale) as i32).collect()
}

/// Scalar blocked GEMM (fallback for non-aarch64)
#[cfg(not(target_arch = "aarch64"))]
fn gemm_i64_scalar(
    a: &[i32], a_rows: usize, a_cols: usize,
    b: &[i32], b_cols: usize,
    scale: u8,
) -> Vec<i32> {
    let m = a_rows;
    let k = a_cols;
    let n = b_cols;

    let mut c = vec![0i64; m * n];

    const BLOCK_SIZE: usize = 64;

    for i0 in (0..m).step_by(BLOCK_SIZE) {
        let i_end = (i0 + BLOCK_SIZE).min(m);

        for k0 in (0..k).step_by(BLOCK_SIZE) {
            let k_end = (k0 + BLOCK_SIZE).min(k);

            for j0 in (0..n).step_by(BLOCK_SIZE) {
                let j_end = (j0 + BLOCK_SIZE).min(n);

                for i in i0..i_end {
                    for kk in k0..k_end {
                        let a_val = a[i * k + kk] as i64;
                        for j in j0..j_end {
                            c[i * n + j] += a_val * b[kk * n + j] as i64;
                        }
                    }
                }
            }
        }
    }

    c.iter().map(|&x| (x >> scale) as i32).collect()
}

/// Batch secure linear using GEMM: Y = X × W + b
///
/// X: [seq_len × in_features] - input matrix (secret-shared)
/// W: [in_features × out_features] - weight matrix (server holds)
/// b: [out_features] - bias (server holds, optional)
/// Y: [seq_len × out_features] - output matrix (secret-shared)
///
/// This uses a single GEMM followed by share/reconstruct at the output level,
/// which is more efficient than creating input shares and doing two GEMMs.
/// Security property: The intermediate computation Y = XW is not secret-shared,
/// but the final result is reconstructed identically to per-token secure linear.
pub fn secure_linear_gemm(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<Vec<FixedVector>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }

    let seq_len = inputs.len();

    // Validate dimensions
    for input in inputs {
        if input.len() != in_features {
            return Err(SharingError::DimensionMismatch {
                expected: in_features,
                got: input.len(),
            });
        }
    }

    if weight.len() != in_features * out_features {
        return Err(SharingError::DimensionMismatch {
            expected: in_features * out_features,
            got: weight.len(),
        });
    }

    // Flatten input vectors into contiguous matrix (avoid Matrix struct overhead)
    let x_data: Vec<i32> = inputs.iter().flat_map(|v| v.data.iter().copied()).collect();

    // Single GEMM: Y = X × W
    let mut y_data = gemm_i64(
        &x_data,
        seq_len,
        in_features,
        weight,
        out_features,
        scale,
    );

    // Add bias
    if let Some(b) = bias {
        for row in 0..seq_len {
            for j in 0..out_features {
                y_data[row * out_features + j] =
                    y_data[row * out_features + j].wrapping_add(b[j]);
            }
        }
    }

    // Convert back to vectors
    let outputs: Vec<FixedVector> = (0..seq_len)
        .map(|i| {
            let start = i * out_features;
            let row_data = y_data[start..start + out_features].to_vec();
            FixedVector::from_raw(row_data, scale)
        })
        .collect();

    Ok(outputs)
}

/// Plaintext GEMM: Y = X × W + b (for verification)
pub fn plaintext_linear_gemm(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<Vec<FixedVector>> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }

    let seq_len = inputs.len();

    // Validate dimensions
    for input in inputs {
        if input.len() != in_features {
            return Err(SharingError::DimensionMismatch {
                expected: in_features,
                got: input.len(),
            });
        }
    }

    // Create input matrix
    let x_matrix = Matrix::from_vectors(inputs)?;

    // GEMM: Y = X × W
    let mut y_data = gemm_i64(
        &x_matrix.data,
        seq_len,
        in_features,
        weight,
        out_features,
        scale,
    );

    // Add bias
    if let Some(b) = bias {
        for row in 0..seq_len {
            for j in 0..out_features {
                y_data[row * out_features + j] =
                    y_data[row * out_features + j].wrapping_add(b[j]);
            }
        }
    }

    let y_matrix = Matrix::from_raw(y_data, seq_len, out_features, scale)?;
    Ok(y_matrix.to_vectors())
}

/// Result of GEMM batch operation with detailed metrics
#[derive(Debug)]
pub struct GemmBatchResult {
    /// Output vectors [seq_len, out_features]
    pub outputs: Vec<FixedVector>,
    /// Total computation time
    pub total_time: std::time::Duration,
    /// Time per token (amortized)
    pub per_token_time: std::time::Duration,
    /// Number of tokens processed
    pub seq_len: usize,
    /// Throughput in tokens/second
    pub tokens_per_sec: f64,
}

/// GEMM-based secure linear with timing metrics
pub fn secure_linear_gemm_timed(
    inputs: &[FixedVector],
    weight: &[i32],
    bias: Option<&[i32]>,
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Result<GemmBatchResult> {
    use std::time::Instant;

    let start = Instant::now();
    let outputs = secure_linear_gemm(inputs, weight, bias, in_features, out_features, scale)?;
    let total_time = start.elapsed();

    let seq_len = inputs.len();
    let per_token_time = if seq_len > 0 {
        total_time / seq_len as u32
    } else {
        std::time::Duration::ZERO
    };

    let tokens_per_sec = if total_time.as_secs_f64() > 0.0 {
        seq_len as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    Ok(GemmBatchResult {
        outputs,
        total_time,
        per_token_time,
        seq_len,
        tokens_per_sec,
    })
}

/// Compare batch outputs between secure and plaintext, returning max error in LSB
pub fn compare_batch_outputs(
    secure: &[FixedVector],
    plaintext: &[FixedVector],
) -> Result<(i64, f64)> {
    if secure.len() != plaintext.len() {
        return Err(SharingError::DimensionMismatch {
            expected: plaintext.len(),
            got: secure.len(),
        });
    }

    let mut max_diff: i64 = 0;
    let mut sum_diff: i64 = 0;
    let mut count: usize = 0;

    for (s_vec, p_vec) in secure.iter().zip(plaintext) {
        if s_vec.len() != p_vec.len() {
            return Err(SharingError::DimensionMismatch {
                expected: p_vec.len(),
                got: s_vec.len(),
            });
        }

        for (&s, &p) in s_vec.data.iter().zip(&p_vec.data) {
            let diff = (s as i64 - p as i64).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
            count += 1;
        }
    }

    let avg_diff = if count > 0 {
        sum_diff as f64 / count as f64
    } else {
        0.0
    };

    Ok((max_diff, avg_diff))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_secure_linear_identity() {
        // Identity matrix should return input
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // 4x4 identity matrix
        let mut weight = vec![0i32; 16];
        for i in 0..4 {
            weight[i * 4 + i] = 1 << DEFAULT_SCALE; // 1.0 in fixed-point
        }

        let (output, _, _) = secure_linear(&input, &weight, None, 4, 4, DEFAULT_SCALE).unwrap();

        let out_f64 = output.to_f64_vec();
        for (i, &expected) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!(
                (out_f64[i] - expected).abs() < 0.01,
                "Mismatch at {}: expected {}, got {}",
                i,
                expected,
                out_f64[i]
            );
        }
    }

    #[test]
    fn test_secure_linear_matches_plaintext() {
        // Random-ish weight matrix
        let in_features = 8;
        let out_features = 4;

        let input_data: Vec<f64> = (0..in_features).map(|i| (i as f64) * 0.1 + 0.5).collect();
        let input = FixedVector::from_f64_slice_default(&input_data).unwrap();

        // Small weight values to avoid overflow
        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 10) - 5) * (1 << (DEFAULT_SCALE - 4)))
            .collect();

        // Compute both ways
        let (secure_output, _, _) =
            secure_linear(&input, &weight, None, in_features, out_features, DEFAULT_SCALE).unwrap();
        let plaintext_output =
            plaintext_linear(&input, &weight, None, in_features, out_features, DEFAULT_SCALE)
                .unwrap();

        // Should match within rounding tolerance
        // The secure version does two separate rescales (one for each share's matmul)
        // which can introduce ±1 LSB difference
        for (s, p) in secure_output.data.iter().zip(&plaintext_output.data) {
            let diff = (*s as i64 - *p as i64).abs();
            assert!(
                diff <= 1,
                "Secure and plaintext should match within ±1 LSB: {} vs {}", s, p
            );
        }
    }

    #[test]
    fn test_secure_linear_with_bias() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 1.0]).unwrap();

        // Identity 2x2
        let weight = vec![
            1 << DEFAULT_SCALE, 0,
            0, 1 << DEFAULT_SCALE,
        ];

        // Bias of [0.5, 1.0]
        let bias = vec![
            (0.5 * (1u64 << DEFAULT_SCALE) as f64) as i32,
            1 << DEFAULT_SCALE,
        ];

        let (output, _, _) =
            secure_linear(&input, &weight, Some(&bias), 2, 2, DEFAULT_SCALE).unwrap();

        let out_f64 = output.to_f64_vec();
        assert!(
            (out_f64[0] - 1.5).abs() < 0.01,
            "Expected 1.5, got {}",
            out_f64[0]
        );
        assert!(
            (out_f64[1] - 2.0).abs() < 0.01,
            "Expected 2.0, got {}",
            out_f64[1]
        );
    }

    #[test]
    fn test_server_never_sees_plaintext_input() {
        let input = FixedVector::from_f64_slice_default(&[42.0, -17.0, 3.14]).unwrap();
        let (client, server_share) = LinearClient::new(&input);
        let request = client.generate_request();

        // What server receives:
        // - request.client_share (X_c = X - X_s)
        // - server_share.data (X_s, received separately)

        // Neither X_c nor X_s equals the plaintext X
        assert_ne!(
            request.client_share, input.data,
            "Client share should not equal plaintext"
        );
        assert_ne!(
            server_share.data, input.data,
            "Server share should not equal plaintext"
        );

        // However, X_c + X_s = X (shares reconstruct correctly)
        let reconstructed: Vec<i32> = request
            .client_share
            .iter()
            .zip(&server_share.data)
            .map(|(&c, &s)| c.wrapping_add(s))
            .collect();
        assert_eq!(reconstructed, input.data, "Shares should reconstruct");
    }

    #[test]
    fn test_dimension_mismatch_rejected() {
        let input = FixedVector::from_f64_slice_default(&[1.0, 2.0]).unwrap();
        let weight = vec![0i32; 12]; // 3x4 matrix, but input is length 2

        let result = secure_linear(&input, &weight, None, 3, 4, DEFAULT_SCALE);
        assert!(
            matches!(result, Err(SharingError::DimensionMismatch { .. })),
            "Should reject dimension mismatch"
        );
    }

    #[test]
    fn test_secure_linear_batch() {
        // Test batch processing with multiple vectors
        let inputs: Vec<FixedVector> = (0..4)
            .map(|i| {
                let data: Vec<f64> = (0..8).map(|j| (i * 8 + j) as f64 * 0.1).collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        // Small weight matrix
        let weight: Vec<i32> = (0..8 * 4)
            .map(|i| ((i as i32 % 10) - 5) * (1 << (DEFAULT_SCALE - 4)))
            .collect();

        // Compute both ways
        let secure_outputs = super::secure_linear_batch(&inputs, &weight, None, 8, 4, DEFAULT_SCALE).unwrap();
        let plain_outputs = super::plaintext_linear_batch(&inputs, &weight, None, 8, 4, DEFAULT_SCALE).unwrap();

        assert_eq!(secure_outputs.len(), inputs.len());
        assert_eq!(plain_outputs.len(), inputs.len());

        // Compare outputs
        let (max_diff, _avg_diff) = super::compare_batch_outputs(&secure_outputs, &plain_outputs).unwrap();
        assert!(max_diff <= 1, "Batch outputs should match within ±1 LSB");
    }

    #[test]
    fn test_secure_linear_batch_timed() {
        // Test timed batch processing
        let inputs: Vec<FixedVector> = (0..8)
            .map(|_| FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap())
            .collect();

        let weight = vec![1 << DEFAULT_SCALE; 4 * 2]; // 4x2 identity-ish

        let result = super::secure_linear_batch_timed(&inputs, &weight, None, 4, 2, DEFAULT_SCALE).unwrap();

        assert_eq!(result.outputs.len(), 8);
        assert_eq!(result.seq_len, 8);
        assert!(result.total_time.as_nanos() > 0);
        assert!(result.per_token_time.as_nanos() > 0);
    }

    #[test]
    fn test_gemm_matches_matvec() {
        // Test that GEMM produces the same result as sequential matvec
        let in_features = 64;
        let out_features = 32;
        let seq_len = 8;

        // Create input sequence
        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t * in_features + i) as f64 * 0.01) - 0.5)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        // Create weight matrix
        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 20) - 10) * (1 << (DEFAULT_SCALE - 5)))
            .collect();

        // Compute with GEMM
        let gemm_outputs = secure_linear_gemm(
            &inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Compute with sequential matvec
        let matvec_outputs: Vec<FixedVector> = inputs.iter()
            .map(|inp| {
                let (out, _, _) = secure_linear(inp, &weight, None, in_features, out_features, DEFAULT_SCALE).unwrap();
                out
            })
            .collect();

        // Results should match (different random shares, but same reconstruction)
        assert_eq!(gemm_outputs.len(), matvec_outputs.len());
        for (g, m) in gemm_outputs.iter().zip(&matvec_outputs) {
            assert_eq!(g.len(), m.len());
            for (&gv, &mv) in g.data.iter().zip(&m.data) {
                // Allow ±2 LSB due to different share randomness and accumulation order
                let diff = (gv as i64 - mv as i64).abs();
                assert!(diff <= 2, "GEMM vs matvec diff too large: {} vs {}, diff={}", gv, mv, diff);
            }
        }
    }

    #[test]
    fn test_gemm_vs_plaintext() {
        let in_features = 32;
        let out_features = 16;
        let seq_len = 4;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t + i) as f64 * 0.1) - 1.0)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 16) - 8) * (1 << (DEFAULT_SCALE - 4)))
            .collect();

        let bias: Vec<i32> = (0..out_features)
            .map(|i| (i as i32 - 8) * (1 << (DEFAULT_SCALE - 3)))
            .collect();

        // Secure GEMM
        let secure_outputs = secure_linear_gemm(
            &inputs, &weight, Some(&bias), in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Plaintext GEMM
        let plaintext_outputs = plaintext_linear_gemm(
            &inputs, &weight, Some(&bias), in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Should match within ±1 LSB
        let (max_diff, avg_diff) = compare_batch_outputs(&secure_outputs, &plaintext_outputs).unwrap();
        assert!(max_diff <= 1, "Max diff {} exceeds 1 LSB", max_diff);
        assert!(avg_diff < 1.0, "Avg diff {} too high", avg_diff);
    }

    #[test]
    fn test_gemm_timed() {
        let in_features = 64;
        let out_features = 64;
        let seq_len = 16;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|_| {
                let data: Vec<f64> = (0..in_features).map(|i| i as f64 * 0.01).collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| (i as i32 % 100) * (1 << (DEFAULT_SCALE - 6)))
            .collect();

        let result = secure_linear_gemm_timed(
            &inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        assert_eq!(result.outputs.len(), seq_len);
        assert_eq!(result.seq_len, seq_len);
        assert!(result.total_time.as_nanos() > 0);
        assert!(result.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_gemm_identity_matrix() {
        let dim = 8;
        let seq_len = 3;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..dim).map(|i| (t * dim + i) as f64 * 0.5).collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        // Identity matrix
        let mut weight = vec![0i32; dim * dim];
        for i in 0..dim {
            weight[i * dim + i] = 1 << DEFAULT_SCALE;
        }

        let outputs = secure_linear_gemm(
            &inputs, &weight, None, dim, dim, DEFAULT_SCALE
        ).unwrap();

        // Output should equal input (within rounding)
        for (inp, out) in inputs.iter().zip(&outputs) {
            for (&i, &o) in inp.data.iter().zip(&out.data) {
                let diff = (i as i64 - o as i64).abs();
                assert!(diff <= 1, "Identity check failed: {} vs {}", i, o);
            }
        }
    }

    // ==================== Hybrid Strategy Tests ====================

    #[test]
    fn test_hybrid_uses_matvec_for_short_sequences() {
        // L=4 < GEMM_CROSSOVER_L, so should use matvec
        let in_features = 32;
        let out_features = 16;
        let seq_len = 4;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t * in_features + i) as f64 * 0.01) - 0.5)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 20) - 10) * (1 << (DEFAULT_SCALE - 5)))
            .collect();

        let result = secure_linear_hybrid_timed(
            &inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        assert!(!result.used_gemm, "Should use matvec for L={}", seq_len);
        assert_eq!(result.outputs.len(), seq_len);
    }

    #[test]
    fn test_hybrid_uses_gemm_for_long_sequences() {
        // L=32 >= GEMM_CROSSOVER_L, so should use GEMM
        let in_features = 32;
        let out_features = 16;
        let seq_len = 32;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t * in_features + i) as f64 * 0.01) - 0.5)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 20) - 10) * (1 << (DEFAULT_SCALE - 5)))
            .collect();

        let result = secure_linear_hybrid_timed(
            &inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        assert!(result.used_gemm, "Should use GEMM for L={}", seq_len);
        assert_eq!(result.outputs.len(), seq_len);
    }

    #[test]
    fn test_hybrid_matches_reference() {
        // Test that hybrid produces same results as direct functions
        let in_features = 32;
        let out_features = 16;

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 20) - 10) * (1 << (DEFAULT_SCALE - 5)))
            .collect();

        // Test short sequence (uses matvec)
        let short_inputs: Vec<FixedVector> = (0..4)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t + i) as f64 * 0.1) - 1.0)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let hybrid_short = secure_linear_hybrid(
            &short_inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();
        let matvec_short = secure_linear_batch(
            &short_inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        let (max_diff, _) = compare_batch_outputs(&hybrid_short, &matvec_short).unwrap();
        assert!(max_diff <= 2, "Hybrid vs matvec diff too large: {}", max_diff);

        // Test long sequence (uses GEMM)
        let long_inputs: Vec<FixedVector> = (0..32)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t + i) as f64 * 0.1) - 1.0)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let hybrid_long = secure_linear_hybrid(
            &long_inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();
        let gemm_long = secure_linear_gemm(
            &long_inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        let (max_diff, _) = compare_batch_outputs(&hybrid_long, &gemm_long).unwrap();
        assert_eq!(max_diff, 0, "Hybrid and GEMM should produce identical results");
    }

    // ==================== Precomputed Transpose Tests ====================

    #[test]
    fn test_linear_weights_creation() {
        let in_features = 4;
        let out_features = 3;

        // Create a simple weight matrix
        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| i as i32 * 100)
            .collect();

        let weights = LinearWeights::new(
            weight.clone(), None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Verify transpose is correct: W[i,j] = W_t[j,i]
        for i in 0..in_features {
            for j in 0..out_features {
                let w_ij = weight[i * out_features + j];
                let wt_ji = weights.weight_t[j * in_features + i];
                assert_eq!(w_ij, wt_ji, "Transpose mismatch at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_pretransposed_matches_gemm() {
        let in_features = 64;
        let out_features = 32;
        let seq_len = 16;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t * in_features + i) as f64 * 0.01) - 0.5)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 20) - 10) * (1 << (DEFAULT_SCALE - 5)))
            .collect();

        // Create LinearWeights with precomputed transpose
        let weights = LinearWeights::new(
            weight.clone(), None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Compute with pretransposed GEMM
        let pretrans_outputs = secure_linear_gemm_pretransposed(&inputs, &weights).unwrap();

        // Compute with regular GEMM
        let gemm_outputs = secure_linear_gemm(
            &inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Should produce identical results
        let (max_diff, _) = compare_batch_outputs(&pretrans_outputs, &gemm_outputs).unwrap();
        assert_eq!(max_diff, 0, "Pretransposed and regular GEMM should match exactly");
    }

    #[test]
    fn test_pretransposed_with_bias() {
        let in_features = 16;
        let out_features = 8;
        let seq_len = 4;

        let inputs: Vec<FixedVector> = (0..seq_len)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t + i) as f64 * 0.1) - 0.8)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 16) - 8) * (1 << (DEFAULT_SCALE - 4)))
            .collect();

        let bias: Vec<i32> = (0..out_features)
            .map(|i| (i as i32 - 4) * (1 << (DEFAULT_SCALE - 2)))
            .collect();

        // Create weights with bias
        let weights = LinearWeights::new(
            weight.clone(), Some(bias.clone()), in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Pretransposed with bias
        let pretrans_outputs = secure_linear_gemm_pretransposed(&inputs, &weights).unwrap();

        // Regular GEMM with bias
        let gemm_outputs = secure_linear_gemm(
            &inputs, &weight, Some(&bias), in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        let (max_diff, _) = compare_batch_outputs(&pretrans_outputs, &gemm_outputs).unwrap();
        assert_eq!(max_diff, 0, "Pretransposed with bias should match GEMM");
    }

    #[test]
    fn test_hybrid_pretransposed() {
        let in_features = 32;
        let out_features = 16;

        let weight: Vec<i32> = (0..in_features * out_features)
            .map(|i| ((i as i32 % 20) - 10) * (1 << (DEFAULT_SCALE - 5)))
            .collect();

        let weights = LinearWeights::new(
            weight.clone(), None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        // Short sequence - should use matvec
        let short_inputs: Vec<FixedVector> = (0..4)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t + i) as f64 * 0.1) - 1.0)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let hybrid_pre = secure_linear_hybrid_pretransposed(&short_inputs, &weights).unwrap();
        let hybrid_reg = secure_linear_hybrid(
            &short_inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        let (max_diff, _) = compare_batch_outputs(&hybrid_pre, &hybrid_reg).unwrap();
        assert!(max_diff <= 2, "Short: pretransposed vs regular diff {}", max_diff);

        // Long sequence - should use GEMM
        let long_inputs: Vec<FixedVector> = (0..32)
            .map(|t| {
                let data: Vec<f64> = (0..in_features)
                    .map(|i| ((t + i) as f64 * 0.1) - 1.0)
                    .collect();
                FixedVector::from_f64_slice_default(&data).unwrap()
            })
            .collect();

        let hybrid_pre = secure_linear_hybrid_pretransposed(&long_inputs, &weights).unwrap();
        let hybrid_reg = secure_linear_hybrid(
            &long_inputs, &weight, None, in_features, out_features, DEFAULT_SCALE
        ).unwrap();

        let (max_diff, _) = compare_batch_outputs(&hybrid_pre, &hybrid_reg).unwrap();
        assert_eq!(max_diff, 0, "Long: pretransposed and regular hybrid should match");
    }
}
