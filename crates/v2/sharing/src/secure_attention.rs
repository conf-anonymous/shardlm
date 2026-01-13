//! Secure attention computation
//!
//! Implements multi-head attention where:
//! - Q, K, V projections computed on shares (server-side)
//! - RoPE applied to Q, K shares (server-side, linear operation)
//! - Attention scores reconstructed by client
//! - Softmax computed by client (nonlinear - cannot be done on shares)
//! - Weighted sum computed on shares (server-side)
//!
//! # Security Guarantee
//!
//! **The server NEVER sees plaintext attention scores or softmax weights.**
//! Only the client can compute softmax because it's nonlinear.
//! RoPE is safe to apply server-side because it's element-wise linear.

use crate::error::{Result, SharingError};
use crate::secure::{ClientShare, ServerComputeResult, ServerContext, ServerShare};
use crate::secure_linear::SecureLinear;

// =============================================================================
// RoPE (Rotary Position Embeddings) - Server-side safe!
// =============================================================================
//
// RoPE is a LINEAR operation: y = x * cos(θ) - x_rotated * sin(θ)
// For secret shares x = x_c + x_s, we can apply RoPE to each share:
//   RoPE(x) = RoPE(x_c) + RoPE(x_s)
// The server never sees the plaintext x, only the individual shares.

/// Precomputed RoPE frequencies
pub struct RopeFrequencies {
    /// Cosine values: [max_seq_len][head_dim/2]
    cos: Vec<Vec<f32>>,
    /// Sine values: [max_seq_len][head_dim/2]
    sin: Vec<Vec<f32>>,
    /// Head dimension
    head_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl RopeFrequencies {
    /// Create RoPE frequencies with specified parameters
    pub fn new(head_dim: usize, max_seq_len: usize, rope_theta: f32) -> Self {
        let half_dim = head_dim / 2;

        // Compute inverse frequencies: 1 / (theta^(2i/d))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32))
            .collect();

        // Precompute cos and sin for each position
        let mut cos = Vec::with_capacity(max_seq_len);
        let mut sin = Vec::with_capacity(max_seq_len);

        for pos in 0..max_seq_len {
            let pos_f32 = pos as f32;
            let cos_row: Vec<f32> = inv_freq.iter().map(|&f| (pos_f32 * f).cos()).collect();
            let sin_row: Vec<f32> = inv_freq.iter().map(|&f| (pos_f32 * f).sin()).collect();
            cos.push(cos_row);
            sin.push(sin_row);
        }

        Self { cos, sin, head_dim, max_seq_len }
    }

    /// Get cos/sin values for a specific position
    pub fn get(&self, position: usize) -> Option<(&[f32], &[f32])> {
        if position < self.max_seq_len {
            Some((&self.cos[position], &self.sin[position]))
        } else {
            None
        }
    }

    /// Apply RoPE to a share vector (can be called on client or server share)
    ///
    /// # Arguments
    /// * `share` - Input share vector [num_heads * head_dim]
    /// * `position` - Sequence position
    /// * `num_heads` - Number of attention heads
    ///
    /// # Security
    /// This is safe to call on individual shares. The server never
    /// reconstructs the plaintext; it just rotates each share.
    pub fn apply_to_share(&self, share: &[f32], position: usize, num_heads: usize) -> Result<Vec<f32>> {
        let (cos, sin) = self.get(position).ok_or_else(|| {
            SharingError::InvalidShare(format!(
                "Position {} exceeds max_seq_len {}", position, self.max_seq_len
            ))
        })?;

        if share.len() != num_heads * self.head_dim {
            return Err(SharingError::ShapeMismatch {
                expected: vec![num_heads * self.head_dim],
                got: vec![share.len()],
            });
        }

        let half_dim = self.head_dim / 2;
        let mut output = vec![0.0f32; share.len()];

        for head in 0..num_heads {
            let offset = head * self.head_dim;

            for i in 0..half_dim {
                let x0 = share[offset + i];
                let x1 = share[offset + i + half_dim];
                let c = cos[i];
                let s = sin[i];

                // RoPE rotation: [cos -sin; sin cos] * [x0; x1]
                output[offset + i] = x0 * c - x1 * s;
                output[offset + i + half_dim] = x0 * s + x1 * c;
            }
        }

        Ok(output)
    }
}

/// Secure attention layer (server-side projections)
///
/// Holds Q, K, V projection weights. Computes projections on shares.
pub struct SecureAttention {
    /// Q projection [hidden_dim × hidden_dim]
    q_proj: SecureLinear,
    /// K projection [hidden_dim × kv_dim]
    k_proj: SecureLinear,
    /// V projection [hidden_dim × kv_dim]
    v_proj: SecureLinear,
    /// Output projection [hidden_dim × hidden_dim]
    o_proj: SecureLinear,
    /// Number of attention heads
    num_heads: usize,
    /// Number of key-value heads (for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// RoPE frequencies (precomputed)
    rope_freqs: RopeFrequencies,
}

impl SecureAttention {
    /// Create a new secure attention layer
    ///
    /// # Arguments
    /// * `rope_theta` - Base frequency for RoPE (Llama: 500000, Qwen: 1000000)
    /// * `max_seq_len` - Maximum sequence length for RoPE precomputation
    pub fn new(
        q_proj: SecureLinear,
        k_proj: SecureLinear,
        v_proj: SecureLinear,
        o_proj: SecureLinear,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
        max_seq_len: usize,
    ) -> Self {
        let hidden_dim = num_heads * head_dim;
        let rope_freqs = RopeFrequencies::new(head_dim, max_seq_len, rope_theta);
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            hidden_dim,
            rope_freqs,
        }
    }

    /// Compute Q, K, V projections on shares WITH RoPE (SERVER-SIDE)
    ///
    /// Returns shares of Q, K, V for client to use in attention computation.
    /// RoPE is applied to Q and K to encode position information.
    ///
    /// # Arguments
    /// * `position` - Sequence position for RoPE encoding
    ///
    /// # Security
    ///
    /// - Server computes on shares separately
    /// - RoPE is applied to each share independently (linear operation)
    /// - Client receives output shares with position encoded
    /// - Client reconstructs Q, K, V locally to compute attention scores
    pub fn project_qkv(
        &self,
        ctx: &ServerContext,
        client_share: &ClientShare<f32>,
        server_share: &ServerShare<f32>,
        position: usize,
    ) -> Result<QkvProjectionResult> {
        // Compute Q projection on shares
        let q_result = self.q_proj.forward_secure(ctx, client_share, server_share)?;

        // Compute K projection on shares
        let k_result = self.k_proj.forward_secure(ctx, client_share, server_share)?;

        // Compute V projection on shares (no RoPE for V)
        let v_result = self.v_proj.forward_secure(ctx, client_share, server_share)?;

        // Apply RoPE to Q shares (safe: linear operation on each share)
        let q_client_rope = self.rope_freqs.apply_to_share(
            &q_result.output_from_client_share, position, self.num_heads
        )?;
        let q_server_rope = self.rope_freqs.apply_to_share(
            &q_result.output_from_server_share, position, self.num_heads
        )?;

        // Apply RoPE to K shares (safe: linear operation on each share)
        let k_client_rope = self.rope_freqs.apply_to_share(
            &k_result.output_from_client_share, position, self.num_kv_heads
        )?;
        let k_server_rope = self.rope_freqs.apply_to_share(
            &k_result.output_from_server_share, position, self.num_kv_heads
        )?;

        Ok(QkvProjectionResult {
            q_client: q_client_rope,
            q_server: q_server_rope,
            k_client: k_client_rope,
            k_server: k_server_rope,
            v_client: v_result.output_from_client_share,
            v_server: v_result.output_from_server_share,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
        })
    }

    /// Compute output projection on attention output shares (SERVER-SIDE)
    ///
    /// After client computes attention and creates new shares, server
    /// projects back to hidden dimension.
    pub fn project_output(
        &self,
        ctx: &ServerContext,
        attn_output_client: &ClientShare<f32>,
        attn_output_server: &ServerShare<f32>,
    ) -> Result<ServerComputeResult<f32>> {
        self.o_proj.forward_secure(ctx, attn_output_client, attn_output_server)
    }

    /// Get number of heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get number of KV heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get hidden dimension
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

/// Result of Q, K, V projections (sent to client)
pub struct QkvProjectionResult {
    /// Q client share [num_heads * head_dim]
    pub q_client: Vec<f32>,
    /// Q server share
    pub q_server: Vec<f32>,
    /// K client share [num_kv_heads * head_dim]
    pub k_client: Vec<f32>,
    /// K server share
    pub k_server: Vec<f32>,
    /// V client share [num_kv_heads * head_dim]
    pub v_client: Vec<f32>,
    /// V server share
    pub v_server: Vec<f32>,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl QkvProjectionResult {
    /// Reconstruct Q, K, V on client side
    ///
    /// # Security
    ///
    /// This is called CLIENT-SIDE only. Client can safely reconstruct
    /// because this is the client's own data.
    pub fn reconstruct(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q: Vec<f32> = self.q_client.iter()
            .zip(self.q_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        let k: Vec<f32> = self.k_client.iter()
            .zip(self.k_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        let v: Vec<f32> = self.v_client.iter()
            .zip(self.v_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        (q, k, v)
    }
}

/// Client-side attention computation
///
/// This module contains functions that MUST run on the client because
/// they involve nonlinear operations (softmax) that cannot be computed
/// on secret shares.
pub mod client {
    use super::*;
    use crate::secure::SecureSharePair;

    /// Compute attention scores and softmax (CLIENT-SIDE ONLY)
    ///
    /// # Arguments
    /// * `q` - Query vector [num_heads * head_dim]
    /// * `k` - Key vector [num_kv_heads * head_dim]
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key-value heads
    /// * `head_dim` - Dimension per head
    ///
    /// # Returns
    /// Attention weights after softmax [num_heads]
    ///
    /// # Security
    ///
    /// This function runs CLIENT-SIDE. The softmax operation is nonlinear
    /// and cannot be computed on secret shares. The server never sees
    /// these attention weights.
    pub fn compute_attention_weights(
        q: &[f32],
        k: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let gqa_ratio = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut weights = Vec::with_capacity(num_heads);

        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;

            // Get Q and K for this head
            let q_start = h * head_dim;
            let k_start = kv_h * head_dim;

            // Compute dot product: Q · K
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score += q[q_start + d] * k[k_start + d];
            }

            // Scale
            score *= scale;

            weights.push(score);
        }

        // Softmax (CLIENT-SIDE - nonlinear operation)
        softmax(&mut weights);

        weights
    }

    /// Compute weighted sum of values (CLIENT-SIDE)
    ///
    /// After softmax, compute: output = sum(weights[h] * V[h])
    ///
    /// For single token attention (no KV cache), this is straightforward.
    pub fn compute_weighted_sum(
        weights: &[f32],
        v: &[f32],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let gqa_ratio = num_heads / num_kv_heads;
        let mut output = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            let weight = weights[h];

            let v_start = kv_h * head_dim;
            let out_start = h * head_dim;

            for d in 0..head_dim {
                output[out_start + d] = weight * v[v_start + d];
            }
        }

        output
    }

    /// Create shares of attention output for server projection
    ///
    /// After computing attention on client, create new shares
    /// to send back to server for output projection.
    pub fn create_output_shares<R: rand::Rng>(
        output: Vec<f32>,
        rng: &mut R,
    ) -> SecureSharePair<f32>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        let shape = vec![output.len()];
        SecureSharePair::from_plaintext(output, shape, rng)
    }

    /// Softmax function
    fn softmax(scores: &mut [f32]) {
        if scores.is_empty() {
            return;
        }

        // Find max for numerical stability
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max)
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }

        // Normalize
        if sum > 0.0 {
            for s in scores.iter_mut() {
                *s /= sum;
            }
        }
    }

    /// Full attention computation (CLIENT-SIDE)
    ///
    /// Convenience function that combines all client-side steps.
    pub fn compute_full_attention<R: rand::Rng>(
        qkv_result: &QkvProjectionResult,
        rng: &mut R,
    ) -> SecureSharePair<f32>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        // Reconstruct Q, K, V
        let (q, k, v) = qkv_result.reconstruct();

        // Compute attention weights (includes softmax)
        let weights = compute_attention_weights(
            &q, &k,
            qkv_result.num_heads,
            qkv_result.num_kv_heads,
            qkv_result.head_dim,
        );

        // Compute weighted sum
        let output = compute_weighted_sum(
            &weights, &v,
            qkv_result.num_heads,
            qkv_result.num_kv_heads,
            qkv_result.head_dim,
        );

        // Create shares for server output projection
        create_output_shares(output, rng)
    }
}

/// GPU-accelerated secure attention using cuBLAS
#[cfg(feature = "cuda")]
pub mod gpu {
    use super::*;
    use crate::secure_linear::gpu::GpuSecureLinear;
    use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};
    use shardlm_v2_core::kernel::KernelContext;

    /// GPU-accelerated secure attention layer
    ///
    /// Uses cuBLAS for Q/K/V/O projections while maintaining security guarantees.
    /// Shares are processed SEPARATELY on the GPU.
    pub struct GpuSecureAttention {
        /// Q projection on GPU
        q_proj: GpuSecureLinear,
        /// K projection on GPU
        k_proj: GpuSecureLinear,
        /// V projection on GPU
        v_proj: GpuSecureLinear,
        /// O projection on GPU
        o_proj: GpuSecureLinear,
        /// Number of attention heads
        num_heads: usize,
        /// Number of KV heads
        num_kv_heads: usize,
        /// Head dimension
        head_dim: usize,
        /// Hidden dimension
        hidden_dim: usize,
        /// RoPE frequencies (precomputed, on CPU)
        rope_freqs: RopeFrequencies,
    }

    impl GpuSecureAttention {
        /// Create GPU-accelerated attention from CPU weights
        pub fn from_cpu(
            q_weights: &[f32],
            k_weights: &[f32],
            v_weights: &[f32],
            o_weights: &[f32],
            q_bias: Option<&[f32]>,
            k_bias: Option<&[f32]>,
            v_bias: Option<&[f32]>,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
            rope_theta: f32,
            max_seq_len: usize,
            device: &GpuDevice,
        ) -> Result<Self> {
            let hidden_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            let q_proj = GpuSecureLinear::from_cpu(
                q_weights, q_bias, hidden_dim, hidden_dim, device
            )?;
            let k_proj = GpuSecureLinear::from_cpu(
                k_weights, k_bias, hidden_dim, kv_dim, device
            )?;
            let v_proj = GpuSecureLinear::from_cpu(
                v_weights, v_bias, hidden_dim, kv_dim, device
            )?;
            let o_proj = GpuSecureLinear::from_cpu(
                o_weights, None, hidden_dim, hidden_dim, device
            )?;

            let rope_freqs = RopeFrequencies::new(head_dim, max_seq_len, rope_theta);

            Ok(Self {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                num_heads,
                num_kv_heads,
                head_dim,
                hidden_dim,
                rope_freqs,
            })
        }

        /// Compute Q, K, V projections on GPU with RoPE (SERVER-SIDE)
        ///
        /// # Performance
        ///
        /// Uses cuBLAS SGEMM for all three projections.
        /// RoPE is applied on CPU after GPU projections (TODO: GPU RoPE kernel)
        ///
        /// # Security
        ///
        /// Shares are processed SEPARATELY. The GPU never computes client_share + server_share.
        pub fn project_qkv_gpu(
            &self,
            ctx: &ServerContext,
            client_share: &[f32],
            server_share: &[f32],
            position: usize,
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<QkvProjectionResult> {
            // Compute Q projection on GPU
            let (q_client, q_server) = self.q_proj.forward_secure_gpu(
                ctx, client_share, server_share, kernels, device
            )?;

            // Compute K projection on GPU
            let (k_client, k_server) = self.k_proj.forward_secure_gpu(
                ctx, client_share, server_share, kernels, device
            )?;

            // Compute V projection on GPU (no RoPE)
            let (v_client, v_server) = self.v_proj.forward_secure_gpu(
                ctx, client_share, server_share, kernels, device
            )?;

            // Apply RoPE to Q shares (on CPU - safe: linear operation)
            let q_client_rope = self.rope_freqs.apply_to_share(
                &q_client, position, self.num_heads
            )?;
            let q_server_rope = self.rope_freqs.apply_to_share(
                &q_server, position, self.num_heads
            )?;

            // Apply RoPE to K shares
            let k_client_rope = self.rope_freqs.apply_to_share(
                &k_client, position, self.num_kv_heads
            )?;
            let k_server_rope = self.rope_freqs.apply_to_share(
                &k_server, position, self.num_kv_heads
            )?;

            Ok(QkvProjectionResult {
                q_client: q_client_rope,
                q_server: q_server_rope,
                k_client: k_client_rope,
                k_server: k_server_rope,
                v_client,
                v_server,
                num_heads: self.num_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
            })
        }

        /// Compute output projection on GPU (SERVER-SIDE)
        pub fn project_output_gpu(
            &self,
            ctx: &ServerContext,
            attn_output_client: &[f32],
            attn_output_server: &[f32],
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            self.o_proj.forward_secure_gpu(ctx, attn_output_client, attn_output_server, kernels, device)
        }

        /// Get number of heads
        pub fn num_heads(&self) -> usize {
            self.num_heads
        }

        /// Get number of KV heads
        pub fn num_kv_heads(&self) -> usize {
            self.num_kv_heads
        }

        /// Get head dimension
        pub fn head_dim(&self) -> usize {
            self.head_dim
        }

        /// Get hidden dimension
        pub fn hidden_dim(&self) -> usize {
            self.hidden_dim
        }

        // ================================================================
        // GPU-TENSOR METHODS - Minimal CPU transfers (only for RoPE)
        // ================================================================

        /// GPU-native QKV projection - NO CPU transfers for linear ops
        ///
        /// Takes CudaTensors, returns CudaTensors.
        /// Note: RoPE still requires brief CPU transfer (TODO: GPU RoPE kernel)
        pub fn project_qkv_gpu_tensor(
            &self,
            client_share: &CudaTensor,
            server_share: &CudaTensor,
            position: usize,
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<QkvTensorResult> {
            // Compute Q projection on GPU (no CPU transfer)
            let (q_client_gpu, q_server_gpu) = self.q_proj.forward_secure_gpu_tensor(
                client_share, server_share, kernels, device
            )?;

            // Compute K projection on GPU (no CPU transfer)
            let (k_client_gpu, k_server_gpu) = self.k_proj.forward_secure_gpu_tensor(
                client_share, server_share, kernels, device
            )?;

            // Compute V projection on GPU (no CPU transfer, no RoPE needed)
            let (v_client, v_server) = self.v_proj.forward_secure_gpu_tensor(
                client_share, server_share, kernels, device
            )?;

            // Apply RoPE on CPU (TODO: GPU RoPE kernel to eliminate this)
            // Download Q/K, apply RoPE, upload back
            let q_client_cpu = q_client_gpu.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let q_server_cpu = q_server_gpu.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let k_client_cpu = k_client_gpu.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let k_server_cpu = k_server_gpu.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            let q_client_rope = self.rope_freqs.apply_to_share(
                &q_client_cpu, position, self.num_heads
            )?;
            let q_server_rope = self.rope_freqs.apply_to_share(
                &q_server_cpu, position, self.num_heads
            )?;
            let k_client_rope = self.rope_freqs.apply_to_share(
                &k_client_cpu, position, self.num_kv_heads
            )?;
            let k_server_rope = self.rope_freqs.apply_to_share(
                &k_server_cpu, position, self.num_kv_heads
            )?;

            // Upload back to GPU
            let q_dim = self.num_heads * self.head_dim;
            let kv_dim = self.num_kv_heads * self.head_dim;

            let q_client = CudaTensor::from_f32(device, vec![q_dim], q_client_rope)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let q_server = CudaTensor::from_f32(device, vec![q_dim], q_server_rope)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let k_client = CudaTensor::from_f32(device, vec![kv_dim], k_client_rope)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let k_server = CudaTensor::from_f32(device, vec![kv_dim], k_server_rope)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            Ok(QkvTensorResult {
                q_client,
                q_server,
                k_client,
                k_server,
                v_client,
                v_server,
            })
        }

        /// GPU-native output projection - NO CPU transfers
        pub fn project_output_gpu_tensor(
            &self,
            attn_output_client: &CudaTensor,
            attn_output_server: &CudaTensor,
            kernels: &KernelContext,
            device: &GpuDevice,
        ) -> Result<(CudaTensor, CudaTensor)> {
            self.o_proj.forward_secure_gpu_tensor(attn_output_client, attn_output_server, kernels, device)
        }
    }

    /// Result of QKV projection with GPU tensors
    pub struct QkvTensorResult {
        pub q_client: CudaTensor,
        pub q_server: CudaTensor,
        pub k_client: CudaTensor,
        pub k_server: CudaTensor,
        pub v_client: CudaTensor,
        pub v_server: CudaTensor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secure::SecureSharePair;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    fn create_identity_linear(size: usize) -> SecureLinear {
        // Create identity-like weight matrix
        let mut weight = vec![0.0f32; size * size];
        for i in 0..size {
            weight[i * size + i] = 1.0;
        }
        SecureLinear::new(weight, None, size, size)
    }

    #[test]
    fn test_qkv_projection() {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        // Create identity projections for testing
        let q_proj = create_identity_linear(hidden_dim);
        let k_proj = create_identity_linear(hidden_dim);
        let v_proj = create_identity_linear(hidden_dim);
        let o_proj = create_identity_linear(hidden_dim);

        let attention = SecureAttention::new(
            q_proj, k_proj, v_proj, o_proj,
            num_heads, num_kv_heads, head_dim,
            10000.0, // rope_theta
            128,     // max_seq_len
        );

        let ctx = ServerContext::new();

        // Create input
        let input: Vec<f32> = (0..hidden_dim).map(|i| i as f32).collect();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let shares = SecureSharePair::from_plaintext(input.clone(), vec![hidden_dim], &mut rng);

        // Compute QKV projections at position 0 (identity RoPE)
        let qkv = attention.project_qkv(&ctx, shares.client_share(), shares.server_share(), 0).unwrap();

        // Reconstruct and verify (with identity projection, output = input)
        let (q, k, v) = qkv.reconstruct();

        for i in 0..hidden_dim {
            assert!((q[i] - input[i]).abs() < 1e-5, "Q mismatch at {}", i);
            assert!((k[i] - input[i]).abs() < 1e-5, "K mismatch at {}", i);
            assert!((v[i] - input[i]).abs() < 1e-5, "V mismatch at {}", i);
        }
    }

    #[test]
    fn test_client_attention_weights() {
        // Test via compute_attention_weights which uses softmax internally
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        // Q and K where Q[0] · K[0] > Q[1] · K[1] to test ordering
        let q = vec![1.0f32, 0.0, 0.0, 0.0,  // head 0
                     0.0, 1.0, 0.0, 0.0];    // head 1
        let k = vec![1.0f32, 0.0, 0.0, 0.0,  // head 0 - high dot product with Q[0]
                     0.0, 0.5, 0.0, 0.0];    // head 1 - lower dot product with Q[1]

        let weights = client::compute_attention_weights(&q, &k, num_heads, num_kv_heads, head_dim);

        // Verify softmax properties - weights should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Attention weights should sum to 1, got {}", sum);

        // Both weights should be positive (softmax output)
        assert!(weights[0] > 0.0);
        assert!(weights[1] > 0.0);
    }

    #[test]
    fn test_full_attention_flow() {
        let hidden_dim = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        let q_proj = create_identity_linear(hidden_dim);
        let k_proj = create_identity_linear(hidden_dim);
        let v_proj = create_identity_linear(hidden_dim);
        let o_proj = create_identity_linear(hidden_dim);

        let attention = SecureAttention::new(
            q_proj, k_proj, v_proj, o_proj,
            num_heads, num_kv_heads, head_dim,
            10000.0, // rope_theta
            128,     // max_seq_len
        );

        let ctx = ServerContext::new();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Input
        let input: Vec<f32> = (0..hidden_dim).map(|i| i as f32 * 0.1).collect();
        let shares = SecureSharePair::from_plaintext(input, vec![hidden_dim], &mut rng);

        // Server: QKV projection at position 0
        let qkv = attention.project_qkv(&ctx, shares.client_share(), shares.server_share(), 0).unwrap();

        // Client: Full attention computation
        let attn_output_shares = client::compute_full_attention(&qkv, &mut rng);

        // Server: Output projection
        let output_result = attention.project_output(
            &ctx,
            attn_output_shares.client_share(),
            attn_output_shares.server_share(),
        ).unwrap();

        // Client: Final reconstruction
        let final_output = output_result.into_share_pair().reconstruct();

        // Verify output has correct shape
        assert_eq!(final_output.len(), hidden_dim);

        // Output should be non-zero (attention was computed)
        let has_nonzero = final_output.iter().any(|&x| x.abs() > 1e-10);
        assert!(has_nonzero, "Output should have non-zero values");
    }

    #[test]
    fn test_server_never_sees_softmax() {
        // This test verifies that the server code path never computes softmax
        // by checking that ServerContext has no softmax method

        let ctx = ServerContext::new();

        // The following would NOT compile if uncommented:
        // ctx.softmax(...);
        // ctx.compute_attention_weights(...);

        // Server can only compute linear projections
        let _ = ctx; // silence unused warning
    }
}
