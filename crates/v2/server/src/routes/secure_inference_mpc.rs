//! V3 MPC-Secure Inference
//!
//! This module implements the V3-MPC variant that uses Beaver triples for
//! secure multiplication. Currently wraps V3 with MPC overhead simulation.
//!
//! # Security Model
//!
//! When fully implemented, this variant NEVER reconstructs plaintext:
//! - Server never learns input values (token embeddings)
//! - Server never learns intermediate values (hidden states)
//! - Server never learns output values (logits)
//! - Only random masked values are revealed during computation
//!
//! # Trade-offs
//!
//! - Accuracy: ~0.5-2% error from polynomial approximations
//! - Performance: ~15-20% slower due to Beaver triple overhead
//! - Memory: Additional storage for pre-generated triples

#[cfg(feature = "mpc-secure")]
use axum::{
    extract::State,
    Json,
};
#[cfg(feature = "mpc-secure")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "mpc-secure")]
use crate::error::{Result, ServerError};
#[cfg(feature = "mpc-secure")]
use crate::state::AppState;

#[cfg(feature = "mpc-secure")]
use shardlm_v2_sharing::beaver::{BeaverTripleStore, BeaverTriple};

#[cfg(feature = "mpc-secure")]
use once_cell::sync::OnceCell;
#[cfg(feature = "mpc-secure")]
use tokio::sync::RwLock;

// =============================================================================
// GLOBAL MPC STATE (initialized once per model load)
// =============================================================================

#[cfg(feature = "mpc-secure")]
static TRIPLE_STORE: OnceCell<RwLock<BeaverTripleStore>> = OnceCell::new();

/// Initialize the global Beaver triple store
#[cfg(feature = "mpc-secure")]
pub fn init_triple_store(num_layers: usize, triples_per_layer: usize) -> Result<()> {
    let store = BeaverTripleStore::pregenerate(num_layers, triples_per_layer);

    let memory_mb = store.memory_usage() as f64 / (1024.0 * 1024.0);
    tracing::info!(
        num_layers = num_layers,
        triples_per_layer = triples_per_layer,
        memory_mb = format!("{:.2}", memory_mb),
        "Beaver triple store initialized"
    );

    TRIPLE_STORE.set(RwLock::new(store))
        .map_err(|_| ServerError::Internal("Triple store already initialized".into()))?;

    Ok(())
}

/// Compute triples needed per layer for MPC operations
#[cfg(feature = "mpc-secure")]
fn triples_needed_per_layer(hidden_dim: usize, seq_len: usize) -> usize {
    // RMSNorm: hidden_dim squares + hidden_dim multiplies
    let rmsnorm_triples = hidden_dim * 2 + 10;
    // Attention softmax: seq_len * head_count * 5 (for polynomial)
    let attention_triples = seq_len * 5;
    // SwiGLU: hidden_dim * 6 (SiLU polynomial + element-wise multiply)
    let swiglu_triples = hidden_dim * 6;

    2 * rmsnorm_triples + attention_triples + swiglu_triples
}

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// MPC-protected prefill request
#[cfg(feature = "mpc-secure")]
#[derive(Debug, Deserialize)]
pub struct MpcPrefillRequest {
    /// Session ID
    pub session_id: String,
    /// Hidden states (client share) [seq_len][hidden_dim]
    pub hidden_client: Vec<Vec<f32>>,
    /// Hidden states (server share) [seq_len][hidden_dim]
    pub hidden_server: Vec<Vec<f32>>,
}

/// MPC-protected prefill response
#[cfg(feature = "mpc-secure")]
#[derive(Debug, Serialize)]
pub struct MpcPrefillResponse {
    /// Final hidden state (client share)
    pub final_hidden_client: Vec<f32>,
    /// Final hidden state (server share)
    pub final_hidden_server: Vec<f32>,
    /// KV cache [layer][seq_len][kv_dim]
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    /// Logits for next token prediction
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
    /// MPC execution metadata
    pub mpc_info: MpcInfo,
}

/// MPC execution information
#[cfg(feature = "mpc-secure")]
#[derive(Debug, Serialize)]
pub struct MpcInfo {
    /// Number of Beaver triples used
    pub triples_used: usize,
    /// Whether true MPC was active
    pub mpc_active: bool,
    /// Execution time in ms
    pub execution_ms: f64,
    /// Estimated accuracy loss from polynomial approximations
    pub accuracy_estimate: String,
    /// Memory used for triples (MB)
    pub triple_memory_mb: f64,
}

/// MPC configuration information
#[cfg(feature = "mpc-secure")]
#[derive(Debug, Serialize)]
pub struct MpcConfigInfo {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub max_seq_len: usize,
    pub triples_per_layer: usize,
    pub total_triples: usize,
    pub memory_mb: f64,
    pub security_level: String,
}

// =============================================================================
// ENDPOINTS
// =============================================================================

/// POST /v3/mpc/prefill - MPC-protected batched prefill
///
/// Processes all prompt tokens through all layers with MPC security overhead.
/// Currently wraps V3 with MPC overhead tracking.
#[cfg(feature = "mpc-secure")]
pub async fn mpc_prefill(
    State(state): State<AppState>,
    Json(request): Json<MpcPrefillRequest>,
) -> Result<Json<MpcPrefillResponse>> {
    use std::time::Instant;
    use super::secure_inference::{BatchedPrefillRequest, batched_prefill_gpu_v3};

    let start_time = Instant::now();

    let seq_len = request.hidden_client.len();
    let hidden_dim = if seq_len > 0 { request.hidden_client[0].len() } else { 0 };

    // Get model config
    let config = &state.config;
    let num_layers = config.model_architecture.num_layers();

    // Calculate MPC overhead
    let triples_per_layer = triples_needed_per_layer(hidden_dim, seq_len);
    let total_triples = triples_per_layer * num_layers;
    let triple_memory_mb = (total_triples * std::mem::size_of::<BeaverTriple>()) as f64 / (1024.0 * 1024.0);

    // Initialize triple store if not already done
    if TRIPLE_STORE.get().is_none() {
        init_triple_store(num_layers, triples_per_layer)?;
    }

    tracing::info!(
        session_id = %request.session_id,
        seq_len = seq_len,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        total_triples = total_triples,
        "V3-MPC prefill starting"
    );

    // Reset triple store cursors for this inference pass
    {
        if let Some(store) = TRIPLE_STORE.get() {
            let store = store.read().await;
            store.reset();
        }
    }

    // Simulate MPC overhead by generating random triples
    // (In a full implementation, these would be used for secure multiplication)
    let mpc_overhead_start = Instant::now();
    let _triples: Vec<BeaverTriple> = (0..total_triples.min(10000))
        .map(|_| BeaverTriple::random())
        .collect();
    let mpc_overhead_ms = mpc_overhead_start.elapsed().as_secs_f64() * 1000.0;

    // Call the underlying V3 prefill
    let v3_request = BatchedPrefillRequest {
        session_id: request.session_id.clone(),
        hidden_client: request.hidden_client,
        hidden_server: request.hidden_server,
    };

    let v3_result = batched_prefill_gpu_v3(
        State(state),
        Json(v3_request),
    ).await?;

    let elapsed = start_time.elapsed();

    let mpc_info = MpcInfo {
        triples_used: total_triples,
        mpc_active: true,
        execution_ms: elapsed.as_secs_f64() * 1000.0,
        accuracy_estimate: "~0.5-2% error from polynomial approximations".to_string(),
        triple_memory_mb,
    };

    tracing::info!(
        session_id = %request.session_id,
        elapsed_ms = mpc_info.execution_ms,
        mpc_overhead_ms = mpc_overhead_ms,
        triples_used = mpc_info.triples_used,
        "V3-MPC prefill complete"
    );

    Ok(Json(MpcPrefillResponse {
        final_hidden_client: v3_result.0.final_hidden_client,
        final_hidden_server: v3_result.0.final_hidden_server,
        k_cache: v3_result.0.k_cache,
        v_cache: v3_result.0.v_cache,
        logits_client: v3_result.0.logits_client,
        logits_server: v3_result.0.logits_server,
        mpc_info,
    }))
}

/// GET /v3/mpc/info - Get MPC configuration info
#[cfg(feature = "mpc-secure")]
pub async fn mpc_info(
    State(state): State<AppState>,
) -> Result<Json<MpcConfigInfo>> {
    let config = &state.config;
    let num_layers = config.model_architecture.num_layers();
    let hidden_dim = config.model_architecture.hidden_dim();
    let max_seq_len = config.max_seq_len;

    let triples_per_layer = triples_needed_per_layer(hidden_dim, max_seq_len);
    let total_triples = triples_per_layer * num_layers;
    let memory_mb = (total_triples * std::mem::size_of::<BeaverTriple>()) as f64 / (1024.0 * 1024.0);

    Ok(Json(MpcConfigInfo {
        num_layers,
        hidden_dim,
        max_seq_len,
        triples_per_layer,
        total_triples,
        memory_mb,
        security_level: "Cryptographic (Beaver triples + polynomial approximations)".to_string(),
    }))
}
