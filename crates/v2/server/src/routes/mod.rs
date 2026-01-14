//! API routes

pub mod health;
pub mod inference;
pub mod secure_inference;
pub mod session;
pub mod ws_streaming;

#[cfg(feature = "binary-protocol")]
pub mod binary_protocol;

#[cfg(feature = "h100-cc")]
pub mod secure_inference_cc;

#[cfg(feature = "mpc-secure")]
pub mod secure_inference_mpc;

pub mod secure_inference_ot;

use axum::{
    routing::{get, post},
    Router,
};

use crate::state::AppState;

/// Create the main router
pub fn create_router(state: AppState) -> Router {
    let router = Router::new()
        // Health and readiness
        .route("/health", get(health::health))
        .route("/ready", get(health::ready))
        .route("/v2/info", get(health::info))
        // Session management (compatible with v1 API)
        .route("/v2/session/new", post(session::create_session))
        .route("/v2/session/status", post(session::session_status))
        // INSECURE inference (for testing only - plaintext)
        .route("/v2/generate", post(inference::generate))
        .route("/v2/generate/stream", post(inference::generate_stream))
        // =================================================================
        // SECURE inference endpoints (privacy-preserving)
        // =================================================================
        // Session initialization with OT handshake
        .route("/v2/secure/session/init", post(secure_inference::init_secure_session))
        // Embedding retrieval via OT (server doesn't learn token IDs)
        .route("/v2/secure/embeddings", post(secure_inference::get_embeddings_ot))
        // Direct embedding lookup (non-private, for testing)
        .route("/v2/secure/embeddings/direct", post(secure_inference::get_embeddings_direct))
        // Layer-by-layer computation on shares (LEGACY - many HTTP round trips)
        .route("/v2/secure/layer/step", post(secure_inference::layer_step))
        .route("/v2/secure/layer/attention", post(secure_inference::process_attention_output))
        // BATCHED forward - processes ALL layers in 2 HTTP requests instead of 56
        .route("/v2/secure/forward/batched", post(secure_inference::batched_forward))
        // Final logits projection
        .route("/v2/secure/logits", post(secure_inference::compute_logits));

    // =================================================================
    // GPU-ACCELERATED secure inference (cuBLAS SGEMM)
    // Same security guarantees, 10-50x faster matrix operations
    // =================================================================
    #[cfg(feature = "cuda")]
    let router = router
        .route("/v2/secure/gpu/layer/step", post(secure_inference::layer_step_gpu))
        .route("/v2/secure/gpu/ffn", post(secure_inference::ffn_step_gpu))
        .route("/v2/secure/gpu/ffn/down", post(secure_inference::ffn_down_gpu))
        .route("/v2/secure/gpu/logits", post(secure_inference::compute_logits_gpu))
        // =================================================================
        // FULLY BATCHED: All 28 layers in ONE request (10-50x faster)
        // Uses secure polynomial approximations for nonlinear ops
        // =================================================================
        .route("/v2/secure/gpu/forward/full", post(secure_inference::batched_full_forward_gpu))
        .route("/v2/secure/gpu/generate/token", post(secure_inference::generate_token_gpu))
        // =================================================================
        // BATCHED PREFILL: All prompt tokens in ONE request
        // Eliminates N-1 HTTP round-trips for N prompt tokens
        // =================================================================
        .route("/v2/secure/gpu/prefill", post(secure_inference::batched_prefill_gpu))
        // =================================================================
        // BATCHED PREFILL V2: Fully GPU-accelerated (RMSNorm + SwiGLU on GPU)
        // =================================================================
        .route("/v2/secure/gpu/prefill_v2", post(secure_inference::batched_prefill_gpu_v2))
        // =================================================================
        // BATCHED PREFILL V3: Tensors stay on GPU (minimal CPU↔GPU transfers)
        // Only uploads once at start, downloads final logits
        // =================================================================
        .route("/v2/secure/gpu/prefill_v3", post(secure_inference::batched_prefill_gpu_v3))
        // =================================================================
        // WEBSOCKET STREAMING: Low-latency token streaming
        // Single connection, tokens streamed as generated
        // =================================================================
        .route("/v2/secure/ws/generate", get(ws_streaming::ws_generate_handler));

    // =================================================================
    // V3 GOLD STANDARD: Binary protocol + MPC secure + H100 CC
    // =================================================================
    #[cfg(all(feature = "cuda", feature = "binary-protocol"))]
    let router = router
        .route("/v3/secure/gpu/prefill", post(secure_inference::batched_prefill_gpu_v3_binary));

    // =================================================================
    // V3-CC: H100 Confidential Computing (Main V3 variant)
    // Hardware memory encryption + attestation
    // =================================================================
    #[cfg(feature = "h100-cc")]
    let router = router
        .route("/v3/cc/attestation", get(secure_inference_cc::get_attestation))
        .route("/v3/cc/verify", post(secure_inference_cc::verify_attestation))
        .route("/v3/cc/prefill", post(secure_inference_cc::cc_prefill));

    // =================================================================
    // V3-MPC: True MPC with Beaver triples (no plaintext reconstruction)
    // =================================================================
    #[cfg(all(feature = "mpc-secure", feature = "cuda"))]
    let router = router
        .route("/v3/mpc/info", get(secure_inference_mpc::mpc_info))
        .route("/v3/mpc/prefill", post(secure_inference_mpc::mpc_prefill));

    // =================================================================
    // V3-OT: Oblivious Transfer for secure function evaluation
    // Uses precomputed lookup tables with OT for nonlinear ops
    // =================================================================
    let router = router
        .route("/v3/ot/info", get(secure_inference_ot::ot_info))
        .route("/v3/ot/tables", get(secure_inference_ot::ot_tables));

    #[cfg(feature = "cuda")]
    let router = router
        .route("/v3/ot/prefill", post(secure_inference_ot::ot_prefill));

    router.with_state(state)
}
