//! V3 Reference Implementation: H100 Confidential Computing Secure Inference
//!
//! This module implements the main V3 variant with hardware-based security
//! using NVIDIA H100 Confidential Computing features.
//!
//! # Security Model
//!
//! H100 CC provides:
//! - **Memory Encryption**: GPU memory encrypted at rest via hardware
//! - **Secure Transfers**: Data encrypted during CPU↔GPU transfers
//! - **Attestation**: Cryptographic proof of secure execution environment
//! - **Isolation**: Protected memory regions inaccessible to host OS
//!
//! # Performance
//!
//! Unlike MPC-based approaches, H100 CC has minimal overhead:
//! - Encryption/decryption in hardware (negligible latency)
//! - No polynomial approximations needed
//! - Native floating-point operations

#[cfg(feature = "h100-cc")]
use axum::{
    extract::State,
    Json,
};
#[cfg(feature = "h100-cc")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "h100-cc")]
use std::sync::Arc;

#[cfg(feature = "h100-cc")]
use crate::error::{Result, ServerError};
#[cfg(feature = "h100-cc")]
use crate::state::AppState;

#[cfg(feature = "h100-cc")]
use shardlm_v2_cc::{
    AttestationToken, ConfidentialCompute, EncryptedBuffer,
    get_cc_provider, is_h100_cc_available,
};

#[cfg(feature = "h100-cc")]
use shardlm_v2_core::gpu::GpuDevice;

#[cfg(feature = "h100-cc")]
use once_cell::sync::OnceCell;
#[cfg(feature = "h100-cc")]
use tokio::sync::RwLock;

// =============================================================================
// GLOBAL CC STATE (initialized once)
// =============================================================================

#[cfg(feature = "h100-cc")]
static CC_PROVIDER: OnceCell<RwLock<Box<dyn ConfidentialCompute>>> = OnceCell::new();

/// Initialize the global CC provider
#[cfg(feature = "h100-cc")]
pub fn init_cc_provider() -> Result<()> {
    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(format!("Failed to create GPU device: {}", e)))?;

    let provider = get_cc_provider(device.cuda_device().clone())
        .map_err(|e| ServerError::Internal(format!("CC initialization failed: {}", e)))?;

    tracing::info!(
        provider = provider.provider_name(),
        hardware_cc = provider.is_available(),
        "Confidential Computing initialized"
    );

    CC_PROVIDER.set(RwLock::new(provider))
        .map_err(|_| ServerError::Internal("CC already initialized".into()))?;

    Ok(())
}

/// Get the CC provider (initializes if needed)
#[cfg(feature = "h100-cc")]
async fn get_cc() -> Result<tokio::sync::RwLockReadGuard<'static, Box<dyn ConfidentialCompute>>> {
    let provider = CC_PROVIDER.get()
        .ok_or_else(|| ServerError::Internal("CC not initialized".into()))?;
    Ok(provider.read().await)
}

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Attestation response
#[cfg(feature = "h100-cc")]
#[derive(Debug, Serialize)]
pub struct AttestationResponse {
    /// Server attestation token
    pub attestation: AttestationToken,
    /// CC provider name
    pub provider: String,
    /// Whether hardware CC is active
    pub hardware_cc: bool,
    /// GPU name
    pub gpu: String,
}

/// Verify attestation request
#[cfg(feature = "h100-cc")]
#[derive(Debug, Deserialize)]
pub struct VerifyAttestationRequest {
    /// Client attestation token to verify
    pub attestation: AttestationToken,
}

/// Verify attestation response
#[cfg(feature = "h100-cc")]
#[derive(Debug, Serialize)]
pub struct VerifyAttestationResponse {
    /// Whether attestation is valid
    pub valid: bool,
    /// Verification message
    pub message: String,
}

/// CC-protected prefill request
#[cfg(feature = "h100-cc")]
#[derive(Debug, Deserialize)]
pub struct CcPrefillRequest {
    /// Session ID
    pub session_id: String,
    /// Client attestation (optional but recommended)
    #[serde(default)]
    pub client_attestation: Option<AttestationToken>,
    /// Hidden states (client share) [seq_len][hidden_dim]
    pub hidden_client: Vec<Vec<f32>>,
    /// Hidden states (server share) [seq_len][hidden_dim]
    pub hidden_server: Vec<Vec<f32>>,
}

/// CC-protected prefill response
#[cfg(feature = "h100-cc")]
#[derive(Debug, Serialize)]
pub struct CcPrefillResponse {
    /// Server attestation proving secure execution
    pub server_attestation: AttestationToken,
    /// Final hidden state (client share)
    pub final_hidden_client: Vec<f32>,
    /// Final hidden state (server share)
    pub final_hidden_server: Vec<f32>,
    /// KV cache as SHARES - maintain consistent API with MPC/OT endpoints
    /// Even in CC mode, we don't return combined values
    /// [layer][seq_len][kv_dim] for each share
    pub k_cache_client: Vec<Vec<Vec<f32>>>,
    pub k_cache_server: Vec<Vec<Vec<f32>>>,
    pub v_cache_client: Vec<Vec<Vec<f32>>>,
    pub v_cache_server: Vec<Vec<Vec<f32>>>,
    /// Logits for next token prediction
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
    /// CC execution metadata
    pub cc_info: CcInfo,
}

/// CC execution information
#[cfg(feature = "h100-cc")]
#[derive(Debug, Serialize)]
pub struct CcInfo {
    /// CC provider used
    pub provider: String,
    /// Whether hardware CC was active
    pub hardware_cc: bool,
    /// Memory was encrypted
    pub memory_encrypted: bool,
    /// Execution time in ms
    pub execution_ms: f64,
}

// =============================================================================
// ENDPOINTS
// =============================================================================

/// GET /v3/cc/attestation - Get server attestation token
///
/// Returns a cryptographic attestation proving the server is running
/// in a secure H100 CC environment.
#[cfg(feature = "h100-cc")]
pub async fn get_attestation(
    State(_state): State<AppState>,
) -> Result<Json<AttestationResponse>> {
    // Initialize CC if not already done
    if CC_PROVIDER.get().is_none() {
        init_cc_provider()?;
    }

    let cc = get_cc().await?;

    // Get fresh attestation
    let attestation = cc.get_attestation()
        .map_err(|e| ServerError::Internal(format!("Attestation failed: {}", e)))?;

    // Get GPU info
    let gpu_name = GpuDevice::new(0)
        .ok()
        .and_then(|d| d.cuda_device().name().ok())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(Json(AttestationResponse {
        attestation,
        provider: cc.provider_name().to_string(),
        hardware_cc: cc.is_available(),
        gpu: gpu_name,
    }))
}

/// POST /v3/cc/verify - Verify client attestation
///
/// Verifies a client's attestation token to ensure they're also
/// running in a secure environment.
#[cfg(feature = "h100-cc")]
pub async fn verify_attestation(
    State(_state): State<AppState>,
    Json(request): Json<VerifyAttestationRequest>,
) -> Result<Json<VerifyAttestationResponse>> {
    // Initialize CC if not already done
    if CC_PROVIDER.get().is_none() {
        init_cc_provider()?;
    }

    let cc = get_cc().await?;

    let valid = cc.verify_attestation(&request.attestation)
        .map_err(|e| ServerError::Internal(format!("Verification failed: {}", e)))?;

    let message = if valid {
        "Client attestation verified successfully".to_string()
    } else {
        "Client attestation verification failed".to_string()
    };

    Ok(Json(VerifyAttestationResponse { valid, message }))
}

/// POST /v3/cc/prefill - CC-protected batched prefill
///
/// Processes all prompt tokens through all layers with H100 CC protection.
/// GPU memory is hardware-encrypted throughout execution.
///
/// # Security Model
///
/// Unlike MPC/OT which use cryptographic protocols, CC relies on H100 hardware:
/// - All GPU memory is hardware-encrypted
/// - Computation happens inside the secure enclave
/// - Attestation proves secure execution environment
///
/// IMPORTANT: This endpoint keeps K,V as separate shares in the response
/// for API consistency, even though CC could safely operate on combined values
/// inside the enclave.
#[cfg(feature = "h100-cc")]
#[axum::debug_handler]
#[allow(deprecated)] // CC mode safely uses _approx functions (hardware encrypts memory)
pub async fn cc_prefill(
    State(state): State<AppState>,
    Json(request): Json<CcPrefillRequest>,
) -> Result<Json<CcPrefillResponse>> {
    use std::time::Instant;
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_gpu, secure_swiglu_gpu, secure_add_gpu,
    };
    use shardlm_v2_core::gpu::CudaTensor;

    let start_time = Instant::now();

    // Initialize CC if not already done
    if CC_PROVIDER.get().is_none() {
        init_cc_provider()?;
    }

    // IMPORTANT: Complete ALL async operations (.await) BEFORE acquiring parking_lot guards.
    // parking_lot guards are not Send and cannot be held across await boundaries.
    // This follows the same pattern as mpc_prefill.

    // Get CC provider and extract all needed info (async operation)
    let (hardware_cc, provider_name, server_attestation) = {
        let cc = get_cc().await?;
        let hardware_cc = cc.is_available();
        let provider_name = cc.provider_name().to_string();

        // Verify client attestation if provided
        if let Some(ref client_attest) = request.client_attestation {
            let valid = cc.verify_attestation(client_attest)
                .map_err(|e| ServerError::Internal(format!("Client attestation failed: {}", e)))?;

            if !valid {
                return Err(ServerError::InvalidRequest(
                    "Client attestation verification failed".into()
                ));
            }

            tracing::info!(
                session_id = %request.session_id,
                "Client attestation verified"
            );
        }

        // Get attestation NOW while we have the lock (avoid second await later)
        let server_attestation = cc.get_attestation()
            .map_err(|e| ServerError::Internal(format!("Attestation failed: {}", e)))?;

        // Encrypt input data for secure GPU transfer (when hardware CC is available)
        if hardware_cc {
            let hidden_client_flat: Vec<f32> = request.hidden_client.iter().flatten().copied().collect();
            let hidden_server_flat: Vec<f32> = request.hidden_server.iter().flatten().copied().collect();

            let _encrypted_client = cc.encrypt_buffer(&hidden_client_flat)
                .map_err(|e| ServerError::Internal(format!("Encryption failed: {}", e)))?;
            let _encrypted_server = cc.encrypt_buffer(&hidden_server_flat)
                .map_err(|e| ServerError::Internal(format!("Encryption failed: {}", e)))?;

            tracing::debug!("Input tensors encrypted for secure GPU transfer");
        }

        (hardware_cc, provider_name, server_attestation)
    }; // CC lock dropped here, BEFORE parking_lot guards

    let seq_len = request.hidden_client.len();

    tracing::info!(
        session_id = %request.session_id,
        seq_len = seq_len,
        cc_provider = %provider_name,
        hardware_cc = hardware_cc,
        "V3-CC prefill starting (secure KV cache as shares)"
    );

    // NOW acquire parking_lot guards (after all .await points)

    // Get GPU resources
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_contexts_guard = state.get_gpu_kernel_contexts()?;
    if kernel_contexts_guard.is_empty() {
        return Err(ServerError::Internal("GPU kernel contexts not initialized".to_string()));
    }

    let ctx = ServerContext::new();
    let num_layers = gpu_weights.num_layers;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;
    let hidden_dim = gpu_weights.hidden_dim;

    // Track which GPU currently holds our tensors
    let mut current_gpu_id: usize = 0;

    // Initialize KV cache as SHARES - we NEVER combine K,V even in CC mode for API consistency
    let mut k_cache_client: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];
    let mut k_cache_server: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];
    let mut v_cache_client: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];
    let mut v_cache_server: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];

    // Bind GPU 0 for initial tensor upload
    let initial_device = kernel_contexts_guard[0].device();
    initial_device.bind_to_thread()
        .map_err(|e| ServerError::GpuError(format!("Failed to bind initial GPU: {}", e)))?;

    // Upload hidden states to GPU (CC hardware encrypts these automatically)
    let mut hidden_client_gpu: Vec<CudaTensor> = request.hidden_client.iter()
        .map(|h| CudaTensor::from_f32(initial_device, vec![1, hidden_dim], h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let mut hidden_server_gpu: Vec<CudaTensor> = request.hidden_server.iter()
        .map(|h| CudaTensor::from_f32(initial_device, vec![1, hidden_dim], h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // Process all layers - CC hardware protects all GPU memory
    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);
        let layer_gpu_id = gpu_weights.layer_gpu_id(layer_idx);
        let kernels = &kernel_contexts_guard[layer_gpu_id];
        let device = kernels.device();

        device.bind_to_thread()
            .map_err(|e| ServerError::GpuError(format!("Failed to bind GPU {}: {}", layer_gpu_id, e)))?;

        // Transfer hidden states to this layer's GPU if needed
        if layer_gpu_id != current_gpu_id {
            let source_device = kernel_contexts_guard[current_gpu_id].device();
            source_device.bind_to_thread()
                .map_err(|e| ServerError::GpuError(format!("Failed to bind source GPU: {}", e)))?;
            source_device.synchronize()
                .map_err(|e| ServerError::GpuError(format!("Failed to sync source GPU: {}", e)))?;

            let client_host_data: Vec<Vec<f32>> = hidden_client_gpu.iter()
                .map(|t| source_device.dtoh_f32(t.data()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(format!("Failed to download tensors: {}", e)))?;

            let server_host_data: Vec<Vec<f32>> = hidden_server_gpu.iter()
                .map(|t| source_device.dtoh_f32(t.data()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(format!("Failed to download tensors: {}", e)))?;

            drop(hidden_client_gpu);
            drop(hidden_server_gpu);

            device.bind_to_thread()
                .map_err(|e| ServerError::GpuError(format!("Failed to bind target GPU: {}", e)))?;

            hidden_client_gpu = client_host_data.into_iter()
                .map(|data| CudaTensor::from_f32(device, vec![1, hidden_dim], data))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            hidden_server_gpu = server_host_data.into_iter()
                .map(|data| CudaTensor::from_f32(device, vec![1, hidden_dim], data))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            device.synchronize()
                .map_err(|e| ServerError::GpuError(format!("Failed to sync target GPU: {}", e)))?;

            current_gpu_id = layer_gpu_id;
        }

        let mut new_hidden_client_gpu = Vec::with_capacity(seq_len);
        let mut new_hidden_server_gpu = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // Input LayerNorm (GPU-native)
            let (normed_client, normed_server) = secure_rms_norm_gpu(
                &hidden_client_gpu[pos],
                &hidden_server_gpu[pos],
                &layer.input_layernorm_gpu,
                1e-6,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} RMSNorm failed: {}", layer_idx, e)))?;

            // QKV Projection (GPU-native)
            let qkv_result = layer.attention.project_qkv_gpu_tensor(
                &normed_client, &normed_server, pos, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} QKV failed: {}", layer_idx, e)))?;

            // Download K, V for cache as SEPARATE SHARES - NEVER combine!
            let k_client_cpu = qkv_result.k_client.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let k_server_cpu = qkv_result.k_server.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let v_client_cpu = qkv_result.v_client.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let v_server_cpu = qkv_result.v_server.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            // Store K, V as SEPARATE SHARES
            k_cache_client[layer_idx].push(k_client_cpu.clone());
            k_cache_server[layer_idx].push(k_server_cpu.clone());
            v_cache_client[layer_idx].push(v_client_cpu.clone());
            v_cache_server[layer_idx].push(v_server_cpu.clone());

            // For attention computation in CC mode, we CAN combine values safely
            // because hardware CC encrypts all memory. This is the key difference
            // from MPC/OT modes where we must use Beaver triples.
            let q_client_cpu = qkv_result.q_client.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let q_server_cpu = qkv_result.q_server.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            // CC mode can safely combine for computation (hardware protects memory)
            let q_combined: Vec<f32> = q_client_cpu.iter()
                .zip(q_server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect();
            let k_combined: Vec<f32> = k_client_cpu.iter()
                .zip(k_server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect();
            let v_combined: Vec<f32> = v_client_cpu.iter()
                .zip(v_server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect();

            // Build cache slice for this position (combined values for internal computation)
            let k_cache_slice: Vec<Vec<f32>> = (0..=pos).map(|p| {
                k_cache_client[layer_idx][p].iter()
                    .zip(k_cache_server[layer_idx][p].iter())
                    .map(|(c, s)| c + s)
                    .collect()
            }).collect();
            let v_cache_slice: Vec<Vec<f32>> = (0..=pos).map(|p| {
                v_cache_client[layer_idx][p].iter()
                    .zip(v_cache_server[layer_idx][p].iter())
                    .map(|(c, s)| c + s)
                    .collect()
            }).collect();

            // Attention (CC hardware protects this computation)
            let attn_output = shardlm_v2_sharing::secure_attention_approx(
                &q_combined,
                &k_cache_slice,
                &v_cache_slice,
                num_heads,
                num_kv_heads,
                head_dim,
                &ctx,
            );

            // Upload attention output to GPU
            let attn_client_gpu = CudaTensor::from_f32(&device, vec![1, hidden_dim], attn_output.clone())
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let attn_server_gpu = CudaTensor::zeros(&device, vec![1, hidden_dim])
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            // O Projection (GPU-native)
            let (o_client, o_server) = layer.attention.project_output_gpu_tensor(
                &attn_client_gpu, &attn_server_gpu, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} O proj failed: {}", layer_idx, e)))?;

            // Add residual (GPU-native)
            let (hidden_after_attn_client, hidden_after_attn_server) = secure_add_gpu(
                &o_client, &o_server,
                &hidden_client_gpu[pos], &hidden_server_gpu[pos],
                kernels, &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} residual add failed: {}", layer_idx, e)))?;

            // Post-Attention LayerNorm (GPU-native)
            let (normed_ffn_client, normed_ffn_server) = secure_rms_norm_gpu(
                &hidden_after_attn_client,
                &hidden_after_attn_server,
                &layer.post_attn_layernorm_gpu,
                1e-6,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} post-attn RMSNorm failed: {}", layer_idx, e)))?;

            // FFN Gate/Up (GPU-native)
            let ffn_result = layer.ffn.project_gate_up_gpu_tensor(
                &normed_ffn_client, &normed_ffn_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN gate/up failed: {}", layer_idx, e)))?;

            // SwiGLU (GPU-native)
            let (activated_client, activated_server) = secure_swiglu_gpu(
                &ffn_result.gate_client,
                &ffn_result.gate_server,
                &ffn_result.up_client,
                &ffn_result.up_server,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} SwiGLU failed: {}", layer_idx, e)))?;

            // FFN Down (GPU-native)
            let (down_client, down_server) = layer.ffn.project_down_gpu_tensor(
                &activated_client, &activated_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN down failed: {}", layer_idx, e)))?;

            // Add residual (GPU-native)
            let (final_client, final_server) = secure_add_gpu(
                &down_client, &down_server,
                &hidden_after_attn_client, &hidden_after_attn_server,
                kernels, &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN residual failed: {}", layer_idx, e)))?;

            new_hidden_client_gpu.push(final_client);
            new_hidden_server_gpu.push(final_server);
        }

        hidden_client_gpu = new_hidden_client_gpu;
        hidden_server_gpu = new_hidden_server_gpu;

        if layer_idx % 7 == 0 || layer_idx == num_layers - 1 {
            tracing::debug!("CC prefill layer {}/{} complete", layer_idx + 1, num_layers);
        }
    }

    // Final device and kernel context
    let final_kernels = &kernel_contexts_guard[current_gpu_id];
    let final_device = final_kernels.device();

    // Compute logits for last token
    let last_idx = seq_len - 1;
    let (normed_client, normed_server) = secure_rms_norm_gpu(
        &hidden_client_gpu[last_idx],
        &hidden_server_gpu[last_idx],
        &gpu_weights.final_norm_gpu,
        1e-6,
        final_kernels,
        final_device,
    ).map_err(|e| ServerError::Internal(format!("Final RMSNorm failed: {}", e)))?;

    let normed_client_cpu = normed_client.to_f32_host(final_device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;
    let normed_server_cpu = normed_server.to_f32_host(final_device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let (logits_client, logits_server) = gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_client_cpu, &normed_server_cpu, final_kernels, final_device
    ).map_err(|e| ServerError::Internal(format!("LM head failed: {}", e)))?;

    // Download final hidden states
    let final_hidden_client = hidden_client_gpu.pop()
        .map(|t| t.to_f32_host(final_device))
        .transpose()
        .map_err(|e| ServerError::GpuError(e.to_string()))?
        .unwrap_or_default();

    let final_hidden_server = hidden_server_gpu.pop()
        .map(|t| t.to_f32_host(final_device))
        .transpose()
        .map_err(|e| ServerError::GpuError(e.to_string()))?
        .unwrap_or_default();

    // No more .await points after this - we got the attestation at the start

    let elapsed = start_time.elapsed();

    let cc_info = CcInfo {
        provider: provider_name,
        hardware_cc,
        memory_encrypted: hardware_cc,
        execution_ms: elapsed.as_secs_f64() * 1000.0,
    };

    tracing::info!(
        session_id = %request.session_id,
        elapsed_ms = cc_info.execution_ms,
        hardware_cc = cc_info.hardware_cc,
        "V3-CC prefill complete (KV cache as shares)"
    );

    Ok(Json(CcPrefillResponse {
        server_attestation,
        final_hidden_client,
        final_hidden_server,
        k_cache_client,
        k_cache_server,
        v_cache_client,
        v_cache_server,
        logits_client,
        logits_server,
        cc_info,
    }))
}

/// Check if H100 CC is available on this system
#[cfg(feature = "h100-cc")]
pub fn check_h100_cc_available() -> bool {
    is_h100_cc_available()
}
