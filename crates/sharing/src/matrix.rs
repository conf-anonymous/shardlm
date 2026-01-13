//! Shared matrix operations for secure linear layers

use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;

/// A vector that is secret-shared between client and server
#[derive(Debug, Clone)]
pub struct SharedVector {
    /// Client's share
    pub client_share: Share,
    /// Server's share
    pub server_share: Share,
}

impl SharedVector {
    /// Create from plaintext (client has plaintext, creates shares)
    pub fn from_plaintext(plaintext: &FixedVector) -> Self {
        let server_share = Share::random(plaintext.len(), plaintext.scale);
        let client_data: Vec<i32> = plaintext
            .data
            .iter()
            .zip(&server_share.data)
            .map(|(&x, &xs)| x.wrapping_sub(xs))
            .collect();
        let client_share = Share::from_raw(client_data, plaintext.scale);

        Self {
            client_share,
            server_share,
        }
    }

    /// Create from plaintext with seed for determinism
    pub fn from_plaintext_seeded(plaintext: &FixedVector, seed: u64) -> Self {
        let server_share = Share::random_seeded(plaintext.len(), plaintext.scale, seed);
        let client_data: Vec<i32> = plaintext
            .data
            .iter()
            .zip(&server_share.data)
            .map(|(&x, &xs)| x.wrapping_sub(xs))
            .collect();
        let client_share = Share::from_raw(client_data, plaintext.scale);

        Self {
            client_share,
            server_share,
        }
    }

    /// Reconstruct plaintext from shares
    pub fn reconstruct(&self) -> Result<FixedVector> {
        if self.client_share.len() != self.server_share.len() {
            return Err(SharingError::DimensionMismatch {
                expected: self.client_share.len(),
                got: self.server_share.len(),
            });
        }

        let data: Vec<i32> = self
            .client_share
            .data
            .iter()
            .zip(&self.server_share.data)
            .map(|(&xc, &xs)| xc.wrapping_add(xs))
            .collect();

        Ok(FixedVector::from_raw(data, self.client_share.scale))
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.client_share.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.client_share.is_empty()
    }

    /// Get scale
    pub fn scale(&self) -> u8 {
        self.client_share.scale
    }

    /// Add two shared vectors (local operation, no communication)
    pub fn add(&self, other: &SharedVector) -> Result<SharedVector> {
        if self.len() != other.len() {
            return Err(SharingError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        if self.scale() != other.scale() {
            return Err(SharingError::ScaleMismatch {
                expected: self.scale(),
                got: other.scale(),
            });
        }

        let client_data: Vec<i32> = self
            .client_share
            .data
            .iter()
            .zip(&other.client_share.data)
            .map(|(&a, &b)| a.wrapping_add(b))
            .collect();
        let server_data: Vec<i32> = self
            .server_share
            .data
            .iter()
            .zip(&other.server_share.data)
            .map(|(&a, &b)| a.wrapping_add(b))
            .collect();

        Ok(SharedVector {
            client_share: Share::from_raw(client_data, self.scale()),
            server_share: Share::from_raw(server_data, self.scale()),
        })
    }
}

/// A matrix that is held in plaintext by one party (typically server)
/// Used for model weights
#[derive(Debug, Clone)]
pub struct SharedMatrix {
    /// Matrix data in row-major order
    pub data: Vec<i32>,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Scale factor
    pub scale: u8,
}

impl SharedMatrix {
    /// Create from row-major f64 data
    pub fn from_f64(data: &[f64], rows: usize, cols: usize, scale: u8) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(SharingError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }

        let fixed_data: Vec<i32> = data
            .iter()
            .map(|&v| {
                let scale_factor = (1u64 << scale) as f64;
                (v * scale_factor).round() as i32
            })
            .collect();

        Ok(Self {
            data: fixed_data,
            rows,
            cols,
            scale,
        })
    }

    /// Create from raw i32 data
    pub fn from_raw(data: Vec<i32>, rows: usize, cols: usize, scale: u8) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(SharingError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }

        Ok(Self {
            data,
            rows,
            cols,
            scale,
        })
    }

    /// Get element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> i32 {
        self.data[row * self.cols + col]
    }

    /// Get a row as a FixedVector
    pub fn row(&self, row: usize) -> FixedVector {
        let start = row * self.cols;
        let end = start + self.cols;
        FixedVector::from_raw(self.data[start..end].to_vec(), self.scale)
    }

    /// Compute Y = X * W where X is a shared vector and W is this matrix
    /// This implements the secure linear layer computation.
    ///
    /// The server computes: Y_s = X_s * W
    /// The client computes: Y_c = X_c * W
    /// Result: Y = Y_c + Y_s = (X_c + X_s) * W = X * W
    pub fn multiply_shared(&self, input: &SharedVector) -> Result<SharedVector> {
        if input.len() != self.rows {
            return Err(SharingError::DimensionMismatch {
                expected: self.rows,
                got: input.len(),
            });
        }
        if input.scale() != self.scale {
            return Err(SharingError::ScaleMismatch {
                expected: self.scale,
                got: input.scale(),
            });
        }

        // Client computes Y_c = X_c * W
        let client_output = self.multiply_share(&input.client_share)?;

        // Server computes Y_s = X_s * W
        let server_output = self.multiply_share(&input.server_share)?;

        Ok(SharedVector {
            client_share: client_output,
            server_share: server_output,
        })
    }

    /// Multiply a single share by this matrix
    fn multiply_share(&self, share: &Share) -> Result<Share> {
        let mut output = vec![0i64; self.cols];

        // Matrix-vector multiply: out[j] = sum_i(input[i] * W[i,j])
        for i in 0..self.rows {
            let input_val = share.data[i] as i64;
            for j in 0..self.cols {
                let weight = self.get(i, j) as i64;
                output[j] += input_val * weight;
            }
        }

        // Rescale: divide by 2^scale
        let data: Vec<i32> = output.iter().map(|&x| (x >> self.scale) as i32).collect();

        Ok(Share::from_raw(data, self.scale))
    }

    /// Add a bias vector to a shared vector (bias is public/known to server)
    pub fn add_bias(shared: &SharedVector, bias: &FixedVector) -> Result<SharedVector> {
        if shared.len() != bias.len() {
            return Err(SharingError::DimensionMismatch {
                expected: shared.len(),
                got: bias.len(),
            });
        }

        // Server adds bias to their share
        let server_data: Vec<i32> = shared
            .server_share
            .data
            .iter()
            .zip(&bias.data)
            .map(|(&s, &b)| s.wrapping_add(b))
            .collect();

        Ok(SharedVector {
            client_share: shared.client_share.clone(),
            server_share: Share::from_raw(server_data, shared.scale()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_shared_vector_roundtrip() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let shared = SharedVector::from_plaintext(&plaintext);
        let reconstructed = shared.reconstruct().unwrap();
        assert_eq!(plaintext, reconstructed);
    }

    #[test]
    fn test_shared_vector_add() {
        let a = FixedVector::from_f64_slice_default(&[1.0, 2.0]).unwrap();
        let b = FixedVector::from_f64_slice_default(&[3.0, 4.0]).unwrap();

        let shared_a = SharedVector::from_plaintext(&a);
        let shared_b = SharedVector::from_plaintext(&b);
        let shared_sum = shared_a.add(&shared_b).unwrap();

        let sum = shared_sum.reconstruct().unwrap();
        let expected = a.add(&b).unwrap();
        assert_eq!(sum, expected);
    }

    #[test]
    fn test_matrix_multiply_plaintext() {
        // Simple 2x2 matrix multiply
        // W = [[1, 2], [3, 4]]
        // X = [1, 1]
        // Y = X * W = [1*1 + 1*3, 1*2 + 1*4] = [4, 6]

        let w = SharedMatrix::from_f64(&[1.0, 2.0, 3.0, 4.0], 2, 2, DEFAULT_SCALE).unwrap();
        let x = FixedVector::from_f64_slice_default(&[1.0, 1.0]).unwrap();

        let shared_x = SharedVector::from_plaintext(&x);
        let shared_y = w.multiply_shared(&shared_x).unwrap();
        let y = shared_y.reconstruct().unwrap();

        let y_f64 = y.to_f64_vec();
        assert!((y_f64[0] - 4.0).abs() < 0.01, "Got {}", y_f64[0]);
        assert!((y_f64[1] - 6.0).abs() < 0.01, "Got {}", y_f64[1]);
    }

    #[test]
    fn test_matrix_multiply_with_bias() {
        let w = SharedMatrix::from_f64(&[1.0, 0.0, 0.0, 1.0], 2, 2, DEFAULT_SCALE).unwrap();
        let x = FixedVector::from_f64_slice_default(&[2.0, 3.0]).unwrap();
        let bias = FixedVector::from_f64_slice_default(&[1.0, 1.0]).unwrap();

        let shared_x = SharedVector::from_plaintext(&x);
        let shared_y = w.multiply_shared(&shared_x).unwrap();
        let shared_y_bias = SharedMatrix::add_bias(&shared_y, &bias).unwrap();
        let y = shared_y_bias.reconstruct().unwrap();

        let y_f64 = y.to_f64_vec();
        // Identity matrix, so Y = X + bias = [3, 4]
        assert!((y_f64[0] - 3.0).abs() < 0.01);
        assert!((y_f64[1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_larger_matrix() {
        // Test with larger dimensions (closer to real model)
        let rows = 128;
        let cols = 64;

        // Create identity-like sparse matrix for easy verification
        let mut data = vec![0.0; rows * cols];
        for i in 0..cols.min(rows) {
            data[i * cols + i] = 1.0;
        }

        let w = SharedMatrix::from_f64(&data, rows, cols, DEFAULT_SCALE).unwrap();
        let x: Vec<f64> = (0..rows).map(|i| i as f64).collect();
        let x_fixed = FixedVector::from_f64_slice_default(&x).unwrap();

        let shared_x = SharedVector::from_plaintext(&x_fixed);
        let shared_y = w.multiply_shared(&shared_x).unwrap();
        let y = shared_y.reconstruct().unwrap();

        let y_f64 = y.to_f64_vec();
        // First 64 elements should equal input, rest are 0
        for i in 0..cols {
            assert!((y_f64[i] - i as f64).abs() < 0.1, "Mismatch at {}", i);
        }
    }
}
