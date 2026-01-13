//! GPU-accelerated secure nonlinear operations
//!
//! This module provides CUDA-accelerated versions of nonlinear functions
//! that operate on secret shares. The security model is the same as the
//! CPU version:
//!
//! 1. Reconstruct shares on GPU (client + server)
//! 2. Compute nonlinear function using CUDA kernel
//! 3. Re-share result (all to client share, 0 to server share)
//!
//! SECURITY NOTE: The server sees reconstructed values during computation.
//! This is the same security level as the CPU version - a practical hybrid
//! approach. For fully MPC-secure computation, polynomial approximations
//! with Beaver triples would be needed.

use crate::ServerContext;

#[cfg(feature = "cuda")]
use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};
#[cfg(feature = "cuda")]
use shardlm_v2_core::kernel::KernelContext;

/// GPU-accelerated secure SiLU activation on shares
///
/// Reconstructs x = client + server, computes SiLU(x) on GPU,
/// returns (result, 0) as new shares.
#[cfg(feature = "cuda")]
pub fn secure_silu_gpu(
    x_client: &CudaTensor,
    x_server: &CudaTensor,
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // Reconstruct on GPU: x = x_client + x_server
    let x = kernels.add(x_client, x_server)
        .map_err(|e| format!("GPU add failed: {}", e))?;

    // Apply SiLU on GPU
    let silu_result = kernels.silu(&x)
        .map_err(|e| format!("GPU SiLU failed: {}", e))?;

    // Re-share: all to client, zero to server
    let out_server = CudaTensor::zeros(device, silu_result.shape.clone())
        .map_err(|e| format!("GPU zeros failed: {}", e))?;

    Ok((silu_result, out_server))
}

/// GPU-accelerated secure RMSNorm on shares
///
/// RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
#[cfg(feature = "cuda")]
pub fn secure_rms_norm_gpu(
    x_client: &CudaTensor,
    x_server: &CudaTensor,
    gamma: &CudaTensor,
    eps: f32,
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // Reconstruct on GPU
    let x = kernels.add(x_client, x_server)
        .map_err(|e| format!("GPU add failed: {}", e))?;

    // RMSNorm kernel expects 2D input [batch, hidden_dim]
    // If input is 1D [hidden_dim], reshape to [1, hidden_dim]
    let x_2d = if x.shape.len() == 1 {
        x.clone_on_device(device)
            .map_err(|e| format!("GPU clone failed: {}", e))?
            .reshape_inplace(vec![1, x.shape[0]])
            .map_err(|e| format!("GPU reshape failed: {}", e))?
    } else {
        x
    };

    // Apply RMSNorm on GPU
    let normed = kernels.rms_norm(&x_2d, gamma, eps)
        .map_err(|e| format!("GPU RMSNorm failed: {}", e))?;

    // Reshape back to 1D if needed (squeeze first dim)
    let normed_squeezed = if normed.shape.len() == 2 && normed.shape[0] == 1 {
        normed.clone_on_device(device)
            .map_err(|e| format!("GPU clone failed: {}", e))?
            .reshape_inplace(vec![normed.shape[1]])
            .map_err(|e| format!("GPU reshape failed: {}", e))?
    } else {
        normed
    };

    // Re-share
    let out_server = CudaTensor::zeros(device, normed_squeezed.shape.clone())
        .map_err(|e| format!("GPU zeros failed: {}", e))?;

    Ok((normed_squeezed, out_server))
}

/// GPU-accelerated secure element-wise multiplication (for SwiGLU)
///
/// Computes gate * up where both are shares
#[cfg(feature = "cuda")]
pub fn secure_mul_gpu(
    a_client: &CudaTensor,
    a_server: &CudaTensor,
    b_client: &CudaTensor,
    b_server: &CudaTensor,
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // Reconstruct both operands on GPU
    let a = kernels.add(a_client, a_server)
        .map_err(|e| format!("GPU add a failed: {}", e))?;
    let b = kernels.add(b_client, b_server)
        .map_err(|e| format!("GPU add b failed: {}", e))?;

    // Multiply on GPU
    let result = kernels.mul(&a, &b)
        .map_err(|e| format!("GPU mul failed: {}", e))?;

    // Re-share
    let out_server = CudaTensor::zeros(device, result.shape.clone())
        .map_err(|e| format!("GPU zeros failed: {}", e))?;

    Ok((result, out_server))
}

/// GPU-accelerated secure SwiGLU activation
///
/// SwiGLU = SiLU(gate) * up
/// Both gate and up are secret-shared
#[cfg(feature = "cuda")]
pub fn secure_swiglu_gpu(
    gate_client: &CudaTensor,
    gate_server: &CudaTensor,
    up_client: &CudaTensor,
    up_server: &CudaTensor,
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // Reconstruct gate and up on GPU
    let gate = kernels.add(gate_client, gate_server)
        .map_err(|e| format!("GPU add gate failed: {}", e))?;
    let up = kernels.add(up_client, up_server)
        .map_err(|e| format!("GPU add up failed: {}", e))?;

    // SiLU(gate) on GPU
    let silu_gate = kernels.silu(&gate)
        .map_err(|e| format!("GPU SiLU failed: {}", e))?;

    // SiLU(gate) * up on GPU
    let result = kernels.mul(&silu_gate, &up)
        .map_err(|e| format!("GPU mul failed: {}", e))?;

    // Re-share
    let out_server = CudaTensor::zeros(device, result.shape.clone())
        .map_err(|e| format!("GPU zeros failed: {}", e))?;

    Ok((result, out_server))
}

/// GPU-accelerated secure attention with KV cache
///
/// Computes attention(Q, K_cache, V_cache) on GPU with softmax
#[cfg(feature = "cuda")]
pub fn secure_attention_gpu(
    q_client: &CudaTensor,
    q_server: &CudaTensor,
    k_cache: &CudaTensor,  // Already reconstructed K cache
    v_cache: &CudaTensor,  // Already reconstructed V cache
    scale: f32,
    q_offset: usize,
    kv_len: usize,
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // Reconstruct Q on GPU
    let q = kernels.add(q_client, q_server)
        .map_err(|e| format!("GPU add Q failed: {}", e))?;

    // Attention with KV cache on GPU (includes softmax)
    let attn_output = kernels.attention_with_kv_cache(
        &q, k_cache, v_cache, scale, q_offset, kv_len
    ).map_err(|e| format!("GPU attention failed: {}", e))?;

    // Re-share
    let out_server = CudaTensor::zeros(device, attn_output.shape.clone())
        .map_err(|e| format!("GPU zeros failed: {}", e))?;

    Ok((attn_output, out_server))
}

/// GPU-accelerated secure residual add
///
/// Computes residual + x where both are shares
#[cfg(feature = "cuda")]
pub fn secure_add_gpu(
    a_client: &CudaTensor,
    a_server: &CudaTensor,
    b_client: &CudaTensor,
    b_server: &CudaTensor,
    kernels: &KernelContext,
    _device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // For addition, we can add shares directly without reconstruction!
    // (a_c + a_s) + (b_c + b_s) = (a_c + b_c) + (a_s + b_s)
    // This is more secure as we never reconstruct
    let out_client = kernels.add(a_client, b_client)
        .map_err(|e| format!("GPU add client failed: {}", e))?;
    let out_server = kernels.add(a_server, b_server)
        .map_err(|e| format!("GPU add server failed: {}", e))?;

    Ok((out_client, out_server))
}

// =============================================================================
// Vec<f32> WRAPPER FUNCTIONS - For easy integration with existing endpoints
// =============================================================================

/// GPU-accelerated secure RMSNorm with Vec<f32> interface
///
/// Uploads shares to GPU, computes RMSNorm, downloads result.
/// Drop-in replacement for `secure_rms_norm_approx` but uses GPU.
#[cfg(feature = "cuda")]
pub fn secure_rms_norm_gpu_vec(
    x_client: &[f32],
    x_server: &[f32],
    gamma: &CudaTensor,  // Gamma already on GPU
    eps: f32,
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let n = x_client.len();

    // Upload shares to GPU
    let x_client_gpu = CudaTensor::from_f32(device, vec![n], x_client.to_vec())
        .map_err(|e| format!("Failed to upload client share: {}", e))?;
    let x_server_gpu = CudaTensor::from_f32(device, vec![n], x_server.to_vec())
        .map_err(|e| format!("Failed to upload server share: {}", e))?;

    // Run GPU RMSNorm
    let (out_client_gpu, out_server_gpu) = secure_rms_norm_gpu(
        &x_client_gpu, &x_server_gpu, gamma, eps, kernels, device
    )?;

    // Download results
    let out_client = out_client_gpu.to_f32_host(device)
        .map_err(|e| format!("Failed to download client result: {}", e))?;
    let out_server = out_server_gpu.to_f32_host(device)
        .map_err(|e| format!("Failed to download server result: {}", e))?;

    Ok((out_client, out_server))
}

/// GPU-accelerated secure SwiGLU with Vec<f32> interface
///
/// Uploads shares to GPU, computes SwiGLU = SiLU(gate) * up, downloads result.
/// Drop-in replacement for `secure_swiglu_approx` but uses GPU.
#[cfg(feature = "cuda")]
pub fn secure_swiglu_gpu_vec(
    gate_client: &[f32],
    gate_server: &[f32],
    up_client: &[f32],
    up_server: &[f32],
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let n = gate_client.len();

    // Upload shares to GPU
    let gate_client_gpu = CudaTensor::from_f32(device, vec![n], gate_client.to_vec())
        .map_err(|e| format!("Failed to upload gate client: {}", e))?;
    let gate_server_gpu = CudaTensor::from_f32(device, vec![n], gate_server.to_vec())
        .map_err(|e| format!("Failed to upload gate server: {}", e))?;
    let up_client_gpu = CudaTensor::from_f32(device, vec![n], up_client.to_vec())
        .map_err(|e| format!("Failed to upload up client: {}", e))?;
    let up_server_gpu = CudaTensor::from_f32(device, vec![n], up_server.to_vec())
        .map_err(|e| format!("Failed to upload up server: {}", e))?;

    // Run GPU SwiGLU
    let (out_client_gpu, out_server_gpu) = secure_swiglu_gpu(
        &gate_client_gpu, &gate_server_gpu,
        &up_client_gpu, &up_server_gpu,
        kernels, device
    )?;

    // Download results
    let out_client = out_client_gpu.to_f32_host(device)
        .map_err(|e| format!("Failed to download client result: {}", e))?;
    let out_server = out_server_gpu.to_f32_host(device)
        .map_err(|e| format!("Failed to download server result: {}", e))?;

    Ok((out_client, out_server))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_module_exists() {
        // Basic test to ensure module compiles
        println!("secure_nonlinear_gpu module loaded");
    }
}
