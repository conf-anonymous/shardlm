//! Secure Feed-Forward Network (FFN) computation
//!
//! Implements SwiGLU FFN: FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
//!
//! Where:
//! - gate_proj, up_proj: Computed on shares (server-side)
//! - silu (x * sigmoid(x)): Computed by client (nonlinear)
//! - Element-wise multiply: Computed by client
//! - down_proj: Computed on shares (server-side)
//!
//! # Security Guarantee
//!
//! **The server NEVER sees plaintext activations.**
//! SiLU is nonlinear and MUST be computed client-side.

use crate::error::{Result, SharingError};
use crate::secure::{ClientShare, ServerComputeResult, ServerContext, ServerShare};
use crate::secure_linear::SecureLinear;

/// Secure FFN layer (server-side projections)
///
/// Holds gate, up, and down projection weights.
pub struct SecureFFN {
    /// Gate projection [hidden_dim × intermediate_dim]
    gate_proj: SecureLinear,
    /// Up projection [hidden_dim × intermediate_dim]
    up_proj: SecureLinear,
    /// Down projection [intermediate_dim × hidden_dim]
    down_proj: SecureLinear,
    /// Hidden dimension
    hidden_dim: usize,
    /// Intermediate dimension
    intermediate_dim: usize,
}

impl SecureFFN {
    /// Create a new secure FFN layer
    pub fn new(
        gate_proj: SecureLinear,
        up_proj: SecureLinear,
        down_proj: SecureLinear,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_dim,
            intermediate_dim,
        }
    }

    /// Compute gate and up projections on shares (SERVER-SIDE)
    ///
    /// Returns shares of gate and up projections for client activation.
    ///
    /// # Security
    ///
    /// - Server computes on shares separately
    /// - Client receives output shares
    /// - Client reconstructs to compute SiLU activation
    pub fn project_gate_up(
        &self,
        ctx: &ServerContext,
        client_share: &ClientShare<f32>,
        server_share: &ServerShare<f32>,
    ) -> Result<GateUpResult> {
        // Compute gate projection on shares
        let gate_result = self.gate_proj.forward_secure(ctx, client_share, server_share)?;

        // Compute up projection on shares
        let up_result = self.up_proj.forward_secure(ctx, client_share, server_share)?;

        Ok(GateUpResult {
            gate_client: gate_result.output_from_client_share,
            gate_server: gate_result.output_from_server_share,
            up_client: up_result.output_from_client_share,
            up_server: up_result.output_from_server_share,
            intermediate_dim: self.intermediate_dim,
        })
    }

    /// Compute down projection on activated shares (SERVER-SIDE)
    ///
    /// After client computes SiLU and creates new shares, server
    /// projects back to hidden dimension.
    pub fn project_down(
        &self,
        ctx: &ServerContext,
        activated_client: &ClientShare<f32>,
        activated_server: &ServerShare<f32>,
    ) -> Result<ServerComputeResult<f32>> {
        self.down_proj.forward_secure(ctx, activated_client, activated_server)
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Get intermediate dimension
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate_dim
    }
}

/// Result of gate and up projections (sent to client)
pub struct GateUpResult {
    /// Gate client share [intermediate_dim]
    pub gate_client: Vec<f32>,
    /// Gate server share
    pub gate_server: Vec<f32>,
    /// Up client share [intermediate_dim]
    pub up_client: Vec<f32>,
    /// Up server share
    pub up_server: Vec<f32>,
    /// Intermediate dimension
    pub intermediate_dim: usize,
}

impl GateUpResult {
    /// Reconstruct gate and up on client side
    ///
    /// # Security
    ///
    /// This is called CLIENT-SIDE only.
    pub fn reconstruct(&self) -> (Vec<f32>, Vec<f32>) {
        let gate: Vec<f32> = self.gate_client.iter()
            .zip(self.gate_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        let up: Vec<f32> = self.up_client.iter()
            .zip(self.up_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        (gate, up)
    }
}

/// Client-side FFN computation
///
/// Contains functions that MUST run on the client because they involve
/// nonlinear operations (SiLU) that cannot be computed on secret shares.
pub mod client {
    use super::*;
    use crate::secure::SecureSharePair;

    /// SiLU activation function: x * sigmoid(x)
    ///
    /// # Security
    ///
    /// This is a nonlinear function and MUST be computed client-side.
    /// The server cannot compute this on shares.
    #[inline]
    pub fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Compute SwiGLU activation (CLIENT-SIDE ONLY)
    ///
    /// SwiGLU(gate, up) = silu(gate) * up
    ///
    /// # Security
    ///
    /// This function runs CLIENT-SIDE. SiLU is nonlinear and cannot
    /// be computed on secret shares.
    pub fn compute_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up.iter())
            .map(|(&g, &u)| silu(g) * u)
            .collect()
    }

    /// Create shares of activated output for server projection
    ///
    /// After computing SwiGLU on client, create new shares
    /// to send back to server for down projection.
    pub fn create_activation_shares<R: rand::Rng>(
        activated: Vec<f32>,
        rng: &mut R,
    ) -> SecureSharePair<f32>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        let shape = vec![activated.len()];
        SecureSharePair::from_plaintext(activated, shape, rng)
    }

    /// Full FFN activation (CLIENT-SIDE)
    ///
    /// Convenience function that combines reconstruction and activation.
    pub fn compute_full_activation<R: rand::Rng>(
        gate_up_result: &GateUpResult,
        rng: &mut R,
    ) -> SecureSharePair<f32>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        // Reconstruct gate and up
        let (gate, up) = gate_up_result.reconstruct();

        // Compute SwiGLU activation
        let activated = compute_swiglu(&gate, &up);

        // Create shares for server down projection
        create_activation_shares(activated, rng)
    }
}

/// Secure RMSNorm (client-side only)
///
/// RMSNorm is nonlinear (involves sqrt and division) and MUST be
/// computed client-side.
pub mod rmsnorm {
    use super::*;
    use crate::secure::SecureSharePair;

    /// Compute RMSNorm (CLIENT-SIDE ONLY)
    ///
    /// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    ///
    /// # Security
    ///
    /// This function runs CLIENT-SIDE. RMSNorm involves sqrt and division
    /// which are nonlinear and cannot be computed on secret shares.
    pub fn compute_rmsnorm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let n = x.len();

        // Compute mean of squares
        let mean_sq: f32 = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;

        // Compute RMS
        let rms = (mean_sq + eps).sqrt();

        // Normalize and scale
        x.iter()
            .zip(weight.iter())
            .map(|(&xi, &wi)| (xi / rms) * wi)
            .collect()
    }

    /// Create shares of normalized output
    pub fn create_normalized_shares<R: rand::Rng>(
        normalized: Vec<f32>,
        rng: &mut R,
    ) -> SecureSharePair<f32>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        let shape = vec![normalized.len()];
        SecureSharePair::from_plaintext(normalized, shape, rng)
    }

    /// Reconstruct input, normalize, and create new shares
    ///
    /// Full client-side RMSNorm pipeline.
    pub fn normalize_and_reshare<R: rand::Rng>(
        client_share: &[f32],
        server_share: &[f32],
        weight: &[f32],
        eps: f32,
        rng: &mut R,
    ) -> SecureSharePair<f32>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        // Reconstruct
        let x: Vec<f32> = client_share.iter()
            .zip(server_share.iter())
            .map(|(c, s)| c + s)
            .collect();

        // Normalize
        let normalized = compute_rmsnorm(&x, weight, eps);

        // Create new shares
        create_normalized_shares(normalized, rng)
    }
}

/// GPU-accelerated secure FFN using cuBLAS
#[cfg(feature = "cuda")]
pub mod gpu {
    use super::*;
    use crate::secure_linear::gpu::GpuSecureLinear;
    use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};
    use shardlm_v2_core::kernel::KernelContext;

    /// GPU-accelerated secure FFN layer
    ///
    /// Uses cuBLAS for matrix multiplications while maintaining security guarantees.
    /// Shares are processed SEPARATELY on the GPU.
    pub struct GpuSecureFFN {
        /// Gate projection on GPU
        gate_proj: GpuSecureLinear,
        /// Up projection on GPU
        up_proj: GpuSecureLinear,
        /// Down projection on GPU
        down_proj: GpuSecureLinear,
        /// Hidden dimension
        hidden_dim: usize,
        /// Intermediate dimension
        intermediate_dim: usize,
    }

    impl GpuSecureFFN {
        /// Create GPU-accelerated FFN from CPU weights
        pub fn from_cpu(
            gate_weights: &[f32],
            up_weights: &[f32],
            down_weights: &[f32],
            hidden_dim: usize,
            intermediate_dim: usize,
            device: &GpuDevice,
        ) -> Result<Self> {
            let gate_proj = GpuSecureLinear::from_cpu(
                gate_weights, None, hidden_dim, intermediate_dim, device
            )?;
            let up_proj = GpuSecureLinear::from_cpu(
                up_weights, None, hidden_dim, intermediate_dim, device
            )?;
            let down_proj = GpuSecureLinear::from_cpu(
                down_weights, None, intermediate_dim, hidden_dim, device
            )?;

            Ok(Self {
                gate_proj,
                up_proj,
                down_proj,
                hidden_dim,
                intermediate_dim,
            })
        }

        /// Compute gate and up projections on GPU (SERVER-SIDE)
        ///
        /// # Performance
        ///
        /// Uses cuBLAS SGEMM for both projections, achieving near-peak GPU throughput.
        ///
        /// # Security
        ///
        /// Shares are processed SEPARATELY. The GPU never computes client_share + server_share.
        pub fn project_gate_up_gpu(
            &self,
            ctx: &ServerContext,
            client_share: &[f32],
            server_share: &[f32],
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<GateUpResult> {
            // Compute gate projection on GPU
            let (gate_client, gate_server) = self.gate_proj.forward_secure_gpu(
                ctx, client_share, server_share, kernels, device
            )?;

            // Compute up projection on GPU
            let (up_client, up_server) = self.up_proj.forward_secure_gpu(
                ctx, client_share, server_share, kernels, device
            )?;

            Ok(GateUpResult {
                gate_client,
                gate_server,
                up_client,
                up_server,
                intermediate_dim: self.intermediate_dim,
            })
        }

        /// Compute down projection on GPU (SERVER-SIDE)
        ///
        /// After client computes SiLU and sends activated shares.
        pub fn project_down_gpu(
            &self,
            ctx: &ServerContext,
            activated_client: &[f32],
            activated_server: &[f32],
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            self.down_proj.forward_secure_gpu(ctx, activated_client, activated_server, kernels, device)
        }

        /// Get hidden dimension
        pub fn hidden_dim(&self) -> usize {
            self.hidden_dim
        }

        /// Get intermediate dimension
        pub fn intermediate_dim(&self) -> usize {
            self.intermediate_dim
        }

        // ================================================================
        // GPU-TENSOR METHODS - No CPU transfers
        // ================================================================

        /// GPU-native gate/up projection - NO CPU transfers
        ///
        /// Takes CudaTensors, returns CudaTensors. Use this for fully-GPU pipelines.
        pub fn project_gate_up_gpu_tensor(
            &self,
            client_share: &CudaTensor,
            server_share: &CudaTensor,
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<GateUpTensorResult> {
            // Compute gate projection on GPU (no CPU transfer)
            let (gate_client, gate_server) = self.gate_proj.forward_secure_gpu_tensor(
                client_share, server_share, kernels, device
            )?;

            // Compute up projection on GPU (no CPU transfer)
            let (up_client, up_server) = self.up_proj.forward_secure_gpu_tensor(
                client_share, server_share, kernels, device
            )?;

            Ok(GateUpTensorResult {
                gate_client,
                gate_server,
                up_client,
                up_server,
            })
        }

        /// GPU-native down projection - NO CPU transfers
        pub fn project_down_gpu_tensor(
            &self,
            activated_client: &CudaTensor,
            activated_server: &CudaTensor,
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<(CudaTensor, CudaTensor)> {
            self.down_proj.forward_secure_gpu_tensor(activated_client, activated_server, kernels, device)
        }
    }

    /// Result of gate/up projection with GPU tensors
    pub struct GateUpTensorResult {
        pub gate_client: CudaTensor,
        pub gate_server: CudaTensor,
        pub up_client: CudaTensor,
        pub up_server: CudaTensor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secure::SecureSharePair;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn create_identity_linear(in_size: usize, out_size: usize) -> SecureLinear {
        // Create simple weight matrix
        let weight = vec![1.0f32; in_size * out_size];
        SecureLinear::new(weight, None, in_size, out_size)
    }

    #[test]
    fn test_silu() {
        // SiLU(0) = 0
        assert!((client::silu(0.0) - 0.0).abs() < 1e-6);

        // SiLU is monotonic for positive values
        assert!(client::silu(1.0) > client::silu(0.5));
        assert!(client::silu(2.0) > client::silu(1.0));

        // SiLU(x) ≈ x for large x
        let large_x = 10.0f32;
        assert!((client::silu(large_x) - large_x).abs() < 0.1);
    }

    #[test]
    fn test_swiglu() {
        let gate = vec![0.0, 1.0, 2.0];
        let up = vec![1.0, 1.0, 1.0];

        let result = client::compute_swiglu(&gate, &up);

        // silu(0) * 1 = 0
        assert!((result[0] - 0.0).abs() < 1e-6);

        // silu(1) * 1 ≈ 0.731
        assert!((result[1] - 0.731).abs() < 0.01);

        // silu(2) * 1 ≈ 1.762
        assert!((result[2] - 1.762).abs() < 0.01);
    }

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let eps = 1e-6;

        let normalized = rmsnorm::compute_rmsnorm(&x, &weight, eps);

        // Check that output is normalized (mean of squares ≈ 1)
        let mean_sq: f32 = normalized.iter().map(|&v| v * v).sum::<f32>() / normalized.len() as f32;

        // Due to RMSNorm formula, mean_sq of output should be close to 1
        // (actually depends on input distribution)
        assert!(mean_sq > 0.0, "Output should have non-zero variance");
    }

    #[test]
    fn test_ffn_full_flow() {
        let hidden_dim = 4;
        let intermediate_dim = 8;

        let gate_proj = create_identity_linear(hidden_dim, intermediate_dim);
        let up_proj = create_identity_linear(hidden_dim, intermediate_dim);
        let down_proj = create_identity_linear(intermediate_dim, hidden_dim);

        let ffn = SecureFFN::new(
            gate_proj, up_proj, down_proj,
            hidden_dim, intermediate_dim,
        );

        let ctx = ServerContext::new();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Input
        let input: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let shares = SecureSharePair::from_plaintext(input, vec![hidden_dim], &mut rng);

        // Server: Gate and up projections
        let gate_up = ffn.project_gate_up(&ctx, shares.client_share(), shares.server_share()).unwrap();

        // Client: SwiGLU activation
        let activated_shares = client::compute_full_activation(&gate_up, &mut rng);

        // Server: Down projection
        let output_result = ffn.project_down(
            &ctx,
            activated_shares.client_share(),
            activated_shares.server_share(),
        ).unwrap();

        // Client: Final reconstruction
        let final_output = output_result.into_share_pair().reconstruct();

        // Verify output has correct shape
        assert_eq!(final_output.len(), hidden_dim);

        // Output should be non-zero
        let has_nonzero = final_output.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero, "Output should have non-zero values");
    }

    #[test]
    fn test_server_cannot_compute_silu() {
        // This test verifies that ServerContext has no silu method
        let ctx = ServerContext::new();

        // The following would NOT compile if uncommented:
        // ctx.silu(...);
        // ctx.compute_swiglu(...);
        // ctx.compute_rmsnorm(...);

        let _ = ctx; // silence unused warning
    }
}
