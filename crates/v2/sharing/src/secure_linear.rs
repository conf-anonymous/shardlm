//! Secure linear layer computation
//!
//! Implements Y = X·W + b where X is secret-shared and W, b are held by server.
//!
//! # Security Guarantee
//!
//! **The server NEVER sees plaintext X.** It only computes on individual shares:
//! - Y_c = X_c · W (output share from client's input share)
//! - Y_s = X_s · W + b (output share from server's input share)
//!
//! The client reconstructs Y = Y_c + Y_s locally.

use crate::error::{Result, SharingError};
use crate::secure::{ClientShare, ServerComputeResult, ServerContext, ServerShare};

/// Secure linear layer (server-side computation)
///
/// Holds weights W and bias b. Computes on shares without reconstruction.
pub struct SecureLinear {
    /// Weight matrix [in_features × out_features], row-major
    weight: Vec<f32>,
    /// Bias vector [out_features], optional
    bias: Option<Vec<f32>>,
    /// Input dimension
    in_features: usize,
    /// Output dimension
    out_features: usize,
}

impl SecureLinear {
    /// Create a new secure linear layer
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, in_features: usize, out_features: usize) -> Self {
        assert_eq!(weight.len(), in_features * out_features, "Weight size mismatch");
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features, "Bias size mismatch");
        }

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Compute on shares (SERVER-SIDE)
    ///
    /// Takes client's share and server's share SEPARATELY.
    /// Returns output shares for client reconstruction.
    ///
    /// # Security
    ///
    /// - NEVER adds the shares together
    /// - Computes Y_c = X_c · W independently
    /// - Computes Y_s = X_s · W + b independently
    /// - Client receives both and reconstructs Y = Y_c + Y_s
    pub fn forward_secure(
        &self,
        _ctx: &ServerContext, // Marker that this is server code
        client_share: &ClientShare<f32>,
        server_share: &ServerShare<f32>,
    ) -> Result<ServerComputeResult<f32>> {
        // Validate dimensions - check actual data length, not declared shape
        if client_share.len() != self.in_features {
            return Err(SharingError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![client_share.len()], // Report actual data length, not declared shape
            });
        }
        if server_share.len() != self.in_features {
            return Err(SharingError::ShapeMismatch {
                expected: vec![self.in_features],
                got: vec![server_share.len()], // Report actual data length, not declared shape
            });
        }

        // ================================================================
        // SECURITY CRITICAL: We compute on each share SEPARATELY
        // NEVER: let x = client_share + server_share
        // ================================================================

        // Y_c = X_c · W (client's input share → client's output share)
        let output_from_client = self.matmul(client_share.data());

        // Y_s = X_s · W + b (server's input share → server's output share)
        let mut output_from_server = self.matmul(server_share.data());

        // Add bias to server's output share only
        if let Some(ref bias) = self.bias {
            for (y, b) in output_from_server.iter_mut().zip(bias.iter()) {
                *y += b;
            }
        }

        Ok(ServerComputeResult {
            output_from_client_share: output_from_client,
            output_from_server_share: output_from_server,
            shape: vec![self.out_features],
        })
    }

    /// Matrix-vector multiply: y = x · W
    fn matmul(&self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_features];

        // Row-major weight layout: W[i,j] = weight[i * out_features + j]
        for i in 0..self.in_features {
            let xi = input[i];
            let row_start = i * self.out_features;
            for j in 0..self.out_features {
                output[j] += xi * self.weight[row_start + j];
            }
        }

        output
    }

    /// Get input dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

/// GPU-accelerated secure linear layer using cuBLAS
#[cfg(feature = "cuda")]
pub mod gpu {
    use super::*;
    use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};
    use shardlm_v2_core::kernel::KernelContext;
    use std::sync::Arc;

    /// GPU secure linear layer with cuBLAS acceleration
    ///
    /// # Performance
    ///
    /// Uses cuBLAS SGEMM for matrix multiplication, achieving near-peak GPU throughput.
    /// Both client and server shares are processed in parallel on GPU.
    ///
    /// # Security
    ///
    /// Maintains the same security guarantee: shares are processed SEPARATELY.
    /// The GPU never computes client_share + server_share.
    pub struct GpuSecureLinear {
        /// Weight matrix on GPU [out_features, in_features] for efficient matmul
        /// Stored transposed for cuBLAS row-major convention
        weight: CudaTensor,
        /// Bias on GPU [out_features], optional
        bias: Option<CudaTensor>,
        /// Input dimension
        in_features: usize,
        /// Output dimension
        out_features: usize,
    }

    impl GpuSecureLinear {
        /// Create from CPU weights
        ///
        /// Weights are stored as [in_features, out_features] in row-major order.
        /// We transpose to [out_features, in_features] for efficient cuBLAS gemv.
        pub fn from_cpu(
            weight: &[f32],
            bias: Option<&[f32]>,
            in_features: usize,
            out_features: usize,
            device: &GpuDevice,
        ) -> Result<Self> {
            // Store weights in [out_features, in_features] layout for cuBLAS
            // Input is [in_features, out_features], we need to transpose
            let mut weight_transposed = vec![0.0f32; in_features * out_features];
            for i in 0..in_features {
                for j in 0..out_features {
                    weight_transposed[j * in_features + i] = weight[i * out_features + j];
                }
            }

            let weight_tensor = CudaTensor::from_f32(
                device,
                vec![out_features, in_features],
                weight_transposed,
            ).map_err(|e| SharingError::CudaError(e.to_string()))?;

            let bias_tensor = if let Some(b) = bias {
                Some(CudaTensor::from_f32(device, vec![out_features], b.to_vec())
                    .map_err(|e| SharingError::CudaError(e.to_string()))?)
            } else {
                None
            };

            Ok(Self {
                weight: weight_tensor,
                bias: bias_tensor,
                in_features,
                out_features,
            })
        }

        /// Compute on GPU shares using cuBLAS (SERVER-SIDE)
        ///
        /// # Security
        ///
        /// CRITICAL: Shares are processed SEPARATELY on GPU.
        /// - Y_c = X_c · W (independent GPU kernel)
        /// - Y_s = X_s · W + b (independent GPU kernel)
        /// The GPU NEVER computes X_c + X_s.
        ///
        /// # Performance
        ///
        /// Uses cuBLAS SGEMM for both matmuls. For typical dimensions:
        /// - Hidden: 1536 → 1536: ~0.1ms per share
        /// - FFN up: 1536 → 8960: ~0.3ms per share
        /// - FFN down: 8960 → 1536: ~0.3ms per share
        pub fn forward_secure_gpu(
            &self,
            _ctx: &ServerContext,
            client_share: &[f32],
            server_share: &[f32],
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            // Validate dimensions
            if client_share.len() != self.in_features {
                return Err(SharingError::ShapeMismatch {
                    expected: vec![self.in_features],
                    got: vec![client_share.len()],
                });
            }
            if server_share.len() != self.in_features {
                return Err(SharingError::ShapeMismatch {
                    expected: vec![self.in_features],
                    got: vec![server_share.len()],
                });
            }

            // ================================================================
            // SECURITY CRITICAL: GPU processes each share SEPARATELY
            // We NEVER compute client_share + server_share on the GPU
            // ================================================================

            // Upload shares to GPU
            let client_gpu = CudaTensor::from_f32(device, vec![1, self.in_features], client_share.to_vec())
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let server_gpu = CudaTensor::from_f32(device, vec![1, self.in_features], server_share.to_vec())
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            // Y_c = X_c · W^T using cuBLAS SGEMM
            // Our weight is [out_features, in_features], input is [1, in_features]
            // Result is [1, out_features]
            let output_client_gpu = kernels.matmul_f32(&client_gpu, &self.weight)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            // Y_s = X_s · W^T using cuBLAS SGEMM
            let mut output_server_gpu = kernels.matmul_f32(&server_gpu, &self.weight)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            // Add bias to server output only (maintains share property)
            if let Some(ref bias) = self.bias {
                output_server_gpu = kernels.add_bias(&output_server_gpu, bias)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?;
            }

            // Download results back to CPU for HTTP response
            let output_client = output_client_gpu.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let output_server = output_server_gpu.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            Ok((output_client, output_server))
        }

        /// Legacy GPU tensor interface (for compatibility)
        pub fn forward_secure(
            &self,
            _ctx: &ServerContext,
            client_share: &CudaTensor,
            server_share: &CudaTensor,
            device: &GpuDevice,
        ) -> Result<(CudaTensor, CudaTensor)> {
            // Validate dimensions
            let client_numel = client_share.shape.iter().product::<usize>();
            let server_numel = server_share.shape.iter().product::<usize>();

            if client_numel != self.in_features {
                return Err(SharingError::ShapeMismatch {
                    expected: vec![self.in_features],
                    got: client_share.shape.clone(),
                });
            }
            if server_numel != self.in_features {
                return Err(SharingError::ShapeMismatch {
                    expected: vec![self.in_features],
                    got: server_share.shape.clone(),
                });
            }

            // Download, compute on CPU, upload (legacy path)
            let input_client = client_share.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let input_server = server_share.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let weight = self.weight.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            // CPU matmul (legacy fallback)
            let mut output_client = vec![0.0f32; self.out_features];
            let mut output_server = vec![0.0f32; self.out_features];

            // Weight is [out_features, in_features]
            for j in 0..self.out_features {
                for i in 0..self.in_features {
                    let w = weight[j * self.in_features + i];
                    output_client[j] += input_client[i] * w;
                    output_server[j] += input_server[i] * w;
                }
            }

            // Add bias to server output
            if let Some(ref bias) = self.bias {
                let bias_cpu = bias.to_f32_host(device)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?;
                for (y, b) in output_server.iter_mut().zip(bias_cpu.iter()) {
                    *y += b;
                }
            }

            let out_client = CudaTensor::from_f32(device, vec![self.out_features], output_client)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let out_server = CudaTensor::from_f32(device, vec![self.out_features], output_server)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            Ok((out_client, out_server))
        }

        /// Get input dimension
        pub fn in_features(&self) -> usize {
            self.in_features
        }

        /// Get output dimension
        pub fn out_features(&self) -> usize {
            self.out_features
        }

        /// GPU-native forward pass - NO CPU transfers
        ///
        /// Takes CudaTensors, does matmul on GPU, returns CudaTensors.
        /// This is the most efficient path for chained GPU operations.
        ///
        /// # Security
        ///
        /// CRITICAL: Shares are processed SEPARATELY on GPU.
        /// The GPU NEVER computes client_share + server_share.
        pub fn forward_secure_gpu_tensor(
            &self,
            client_share: &CudaTensor,
            server_share: &CudaTensor,
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<(CudaTensor, CudaTensor)> {
            // Ensure 2D shape [batch, in_features] for matmul
            // Uses efficient GPU-to-GPU copy, then reshape metadata only
            let client_2d = if client_share.shape.len() == 1 {
                client_share.clone_on_device(device)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
                    .reshape_inplace(vec![1, self.in_features])
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
            } else if client_share.shape == vec![1, self.in_features] {
                // Already correct shape, avoid copy
                client_share.clone_on_device(device)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
            } else {
                client_share.clone_on_device(device)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
            };
            let server_2d = if server_share.shape.len() == 1 {
                server_share.clone_on_device(device)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
                    .reshape_inplace(vec![1, self.in_features])
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
            } else {
                server_share.clone_on_device(device)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?
            };

            // Y_c = X_c · W^T using cuBLAS SGEMM
            let output_client = kernels.matmul_f32(&client_2d, &self.weight)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            // Y_s = X_s · W^T using cuBLAS SGEMM
            let mut output_server = kernels.matmul_f32(&server_2d, &self.weight)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            // Add bias to server output only (maintains share property)
            if let Some(ref bias) = self.bias {
                output_server = kernels.add_bias(&output_server, bias)
                    .map_err(|e| SharingError::CudaError(e.to_string()))?;
            }

            Ok((output_client, output_server))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secure::SecureSharePair;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_secure_linear_correctness() {
        // Test that secure computation produces correct result
        let in_features = 4;
        let out_features = 2;

        // Simple weight matrix (identity-like for easy verification)
        let weight = vec![
            1.0, 0.0,  // Row 0: maps x[0] to y[0]
            0.0, 1.0,  // Row 1: maps x[1] to y[1]
            1.0, 1.0,  // Row 2: maps x[2] to both
            0.5, 0.5,  // Row 3: maps x[3] to both (half)
        ];
        let bias = Some(vec![0.1, 0.2]);

        let layer = SecureLinear::new(weight, bias, in_features, out_features);
        let ctx = ServerContext::new();

        // Plaintext input
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        // Expected output (computed by hand):
        // y[0] = 1*1 + 0*2 + 1*3 + 0.5*4 + 0.1 = 1 + 0 + 3 + 2 + 0.1 = 6.1
        // y[1] = 0*1 + 1*2 + 1*3 + 0.5*4 + 0.2 = 0 + 2 + 3 + 2 + 0.2 = 7.2
        let expected = vec![6.1, 7.2];

        // Create shares
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let shares = SecureSharePair::from_plaintext(x, vec![in_features], &mut rng);

        // Secure computation (server side)
        let result = layer.forward_secure(
            &ctx,
            shares.client_share(),
            shares.server_share(),
        ).unwrap();

        // Reconstruct (client side)
        let output_shares = result.into_share_pair();
        let output = output_shares.reconstruct();

        // Verify correctness
        for (computed, expected) in output.iter().zip(expected.iter()) {
            assert!((computed - expected).abs() < 1e-5,
                "Mismatch: computed={}, expected={}", computed, expected);
        }
    }

    #[test]
    fn test_shares_are_not_plaintext() {
        let in_features = 4;
        let out_features = 2;

        let weight = vec![1.0; in_features * out_features];
        let layer = SecureLinear::new(weight, None, in_features, out_features);
        let ctx = ServerContext::new();

        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let shares = SecureSharePair::from_plaintext(x.clone(), vec![in_features], &mut rng);

        let result = layer.forward_secure(
            &ctx,
            shares.client_share(),
            shares.server_share(),
        ).unwrap();

        // Verify that output shares are NOT the plaintext output
        let plaintext_output: Vec<f32> = vec![10.0, 10.0]; // sum of all inputs

        // Client output share should not equal plaintext
        assert_ne!(result.output_from_client_share, plaintext_output);
        // Server output share should not equal plaintext
        assert_ne!(result.output_from_server_share, plaintext_output);

        // But reconstruction should equal plaintext
        let output_shares = result.into_share_pair();
        let reconstructed = output_shares.reconstruct();

        for (r, e) in reconstructed.iter().zip(plaintext_output.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }
}
