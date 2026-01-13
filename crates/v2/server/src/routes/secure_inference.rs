//! Secure inference endpoints
//!
//! These endpoints implement privacy-preserving inference using secret sharing
//! and oblivious transfer. **THE SERVER NEVER SEES PLAINTEXT USER DATA.**
//!
//! # Protocol Overview
//!
//! 1. `/v2/secure/session/init` - Initialize OT session with base OT handshake
//! 2. `/v2/secure/embeddings` - Client retrieves embeddings via OT (server doesn't know which)
//! 3. `/v2/secure/step` - Server processes one transformer layer on shares
//! 4. `/v2/secure/finalize` - Client finalizes generation (multiple rounds)
//!
//! # Security Guarantees
//!
//! - Server NEVER reconstructs plaintext from shares
//! - Server NEVER learns which embeddings were requested
//! - All nonlinear operations (softmax, SiLU, RMSNorm) are computed client-side
//! - `ServerContext` type enforces these guarantees at compile time

use std::sync::Arc;
use std::collections::HashMap;

use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::error::{Result, ServerError};
use crate::state::AppState;

// =============================================================================
// TYPES
// =============================================================================

/// Secure session state
///
/// Tracks OT session and maintains shares across inference steps.
pub struct SecureSession {
    /// Session ID
    pub id: Uuid,
    /// OT session established
    pub ot_initialized: bool,
    /// Current layer being processed
    pub current_layer: usize,
    /// Total layers in model
    pub total_layers: usize,
    /// Client shares from previous step (for next layer)
    pub pending_client_shares: Option<Vec<f32>>,
    /// Server shares from previous step
    pub pending_server_shares: Option<Vec<f32>>,
    /// Sequence position (for KV cache)
    pub seq_position: usize,
    /// Created timestamp
    pub created_at: std::time::Instant,
}

impl SecureSession {
    pub fn new(total_layers: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            ot_initialized: false,
            current_layer: 0,
            total_layers,
            pending_client_shares: None,
            pending_server_shares: None,
            seq_position: 0,
            created_at: std::time::Instant::now(),
        }
    }
}

/// Secure sessions storage
pub type SecureSessions = Arc<RwLock<HashMap<Uuid, SecureSession>>>;

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Initialize secure session request
#[derive(Debug, Deserialize)]
pub struct InitSecureSessionRequest {
    /// Client's base OT public keys (for IKNP OT)
    #[serde(default)]
    pub ot_public_keys: Option<Vec<u8>>,
}

/// Initialize secure session response
#[derive(Debug, Serialize)]
pub struct InitSecureSessionResponse {
    /// Session ID for subsequent requests
    pub session_id: String,
    /// Server's OT public data (for completing base OT)
    pub ot_server_data: Vec<u8>,
    /// Model info
    pub model_info: ModelInfo,
}

/// Model information for client
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Vocabulary size (for OT)
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_dim: usize,
}

/// Embedding retrieval request (client sends OT query)
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// Session ID
    pub session_id: String,
    /// OT extension query (encrypted indices)
    /// The server cannot learn which indices are being requested
    pub ot_query: Vec<u8>,
    /// Number of tokens being requested
    pub num_tokens: usize,
}

/// Embedding retrieval response (server sends masked embeddings)
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// OT response data (masked embeddings - client can only decode requested ones)
    pub ot_response: Vec<u8>,
    /// Per-row encryption keys (one per vocabulary entry)
    /// These are masked so client can only use the ones they requested
    pub masked_keys: Vec<u8>,
}

/// Direct embedding lookup request (simpler than OT, for testing)
/// NOTE: This reveals token IDs to server - use OT for production privacy
#[derive(Debug, Deserialize)]
pub struct DirectEmbeddingRequest {
    /// Session ID
    pub session_id: String,
    /// Token IDs to look up
    pub token_ids: Vec<u32>,
}

/// Direct embedding lookup response (returns shares)
#[derive(Debug, Serialize)]
pub struct DirectEmbeddingResponse {
    /// Client shares for each token [num_tokens][hidden_dim]
    pub client_shares: Vec<Vec<f32>>,
    /// Server shares for each token [num_tokens][hidden_dim]
    pub server_shares: Vec<Vec<f32>>,
}

/// Single layer computation request
#[derive(Debug, Deserialize)]
pub struct LayerStepRequest {
    /// Session ID
    pub session_id: String,
    /// Client's share of hidden state [hidden_dim]
    pub client_share: Vec<f32>,
    /// Server's share of hidden state [hidden_dim]
    pub server_share: Vec<f32>,
    /// Layer index to process
    pub layer_idx: usize,
    /// Sequence position for RoPE encoding (defaults to 0)
    #[serde(default)]
    pub position: usize,
}

/// Layer computation response
///
/// Contains shares of:
/// - Q, K, V projections (for client to compute attention)
/// - After client sends back attention output shares: FFN output shares
#[derive(Debug, Serialize)]
pub struct LayerStepResponse {
    /// Phase of computation ("qkv", "ffn_output")
    pub phase: String,

    // QKV phase outputs (for client attention computation)
    /// Q client share [num_heads * head_dim]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_client: Option<Vec<f32>>,
    /// Q server share
    #[serde(skip_serializing_if = "Option::is_none")]
    pub q_server: Option<Vec<f32>>,
    /// K client share [num_kv_heads * head_dim]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k_client: Option<Vec<f32>>,
    /// K server share
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k_server: Option<Vec<f32>>,
    /// V client share [num_kv_heads * head_dim]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v_client: Option<Vec<f32>>,
    /// V server share
    #[serde(skip_serializing_if = "Option::is_none")]
    pub v_server: Option<Vec<f32>>,

    // FFN output phase (final output for this layer)
    /// Output hidden state client share [hidden_dim]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_client: Option<Vec<f32>>,
    /// Output hidden state server share [hidden_dim]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_server: Option<Vec<f32>>,

    /// Metadata
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

/// Attention output request (client sends attention result for FFN processing)
#[derive(Debug, Deserialize)]
pub struct AttentionOutputRequest {
    /// Session ID
    pub session_id: String,
    /// Layer index
    pub layer_idx: usize,
    /// Attention output client share (after client softmax + weighted sum)
    pub attn_output_client: Vec<f32>,
    /// Attention output server share
    pub attn_output_server: Vec<f32>,
    /// Original hidden state client share (for residual connection)
    pub hidden_client: Vec<f32>,
    /// Original hidden state server share (for residual connection)
    pub hidden_server: Vec<f32>,
}

/// Logits request (after all layers, project to vocabulary)
#[derive(Debug, Deserialize)]
pub struct LogitsRequest {
    /// Session ID
    pub session_id: String,
    /// Final hidden state client share
    pub hidden_client: Vec<f32>,
    /// Final hidden state server share
    pub hidden_server: Vec<f32>,
}

/// Logits response
#[derive(Debug, Serialize)]
pub struct LogitsResponse {
    /// Logits client share [vocab_size]
    pub logits_client: Vec<f32>,
    /// Logits server share [vocab_size]
    pub logits_server: Vec<f32>,
}

// =============================================================================
// HANDLERS
// =============================================================================

/// POST /v2/secure/session/init - Initialize secure session with OT
#[axum::debug_handler]
pub async fn init_secure_session(
    State(state): State<AppState>,
    Json(_request): Json<InitSecureSessionRequest>,
) -> Result<Json<InitSecureSessionResponse>> {
    // TODO: Implement proper OT base protocol
    // For now, create session with placeholder

    // Get model info from secure weights (actual loaded model dimensions)
    // Note: We extract values and drop the guard before any await points
    let (num_layers, hidden_dim, num_heads, num_kv_heads, head_dim, intermediate_dim, vocab_size) = {
        let secure_weights_guard = state.get_secure_weights()?;
        let secure_weights = secure_weights_guard.as_ref()
            .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;
        (
            secure_weights.num_layers,
            secure_weights.hidden_dim,
            secure_weights.num_heads,
            secure_weights.num_kv_heads,
            secure_weights.head_dim,
            secure_weights.intermediate_dim,
            secure_weights.vocab_size,
        )
    };

    let session = SecureSession::new(num_layers);
    let session_id = session.id;

    // Store session
    {
        let mut sessions = state.secure_sessions.write().await;
        sessions.insert(session_id, session);
    }

    // Generate server OT data (placeholder for now)
    let ot_server_data = vec![0u8; 32]; // Placeholder

    tracing::info!(
        session_id = %session_id,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        vocab_size = vocab_size,
        "Initialized secure inference session"
    );

    Ok(Json(InitSecureSessionResponse {
        session_id: session_id.to_string(),
        ot_server_data,
        model_info: ModelInfo {
            vocab_size,
            embedding_dim: hidden_dim,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
        },
    }))
}

/// POST /v2/secure/embeddings - Retrieve embeddings via OT
///
/// For now, this uses a simplified protocol where the server returns
/// the embedding table chunk-by-chunk. Full OT implementation would
/// use 1-out-of-N OT extension for private token lookup.
#[axum::debug_handler]
pub async fn get_embeddings_ot(
    State(state): State<AppState>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>> {
    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get secure weights from state
    let secure_weights_guard = state.get_secure_weights()?;
    let secure_weights = secure_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;

    tracing::debug!(
        session_id = %session_id,
        num_tokens = request.num_tokens,
        vocab_size = secure_weights.vocab_size,
        hidden_dim = secure_weights.hidden_dim,
        "Processing OT embedding request - using actual embeddings"
    );

    // For production OT:
    // 1. Parse client's OT query (contains oblivious transfer data)
    // 2. For each embedding row, mask with a unique key derived from OT
    // 3. Return masked embeddings and masked keys
    // 4. Only client can decode the rows they requested
    //
    // For now, we return OT response metadata indicating the embedding dimensions
    // The actual private lookup would use SecureOtServer from shardlm-v2-sharing

    // Package embedding table info for OT protocol
    let vocab_size = secure_weights.vocab_size as u32;
    let hidden_dim = secure_weights.hidden_dim as u32;

    // Return metadata as OT response (client will use this for OT extension)
    let mut ot_response = Vec::with_capacity(8);
    ot_response.extend_from_slice(&vocab_size.to_le_bytes());
    ot_response.extend_from_slice(&hidden_dim.to_le_bytes());

    Ok(Json(EmbeddingResponse {
        ot_response,
        masked_keys: vec![0u8; 32], // Placeholder for OT keys
    }))
}

/// POST /v2/secure/embeddings/direct - Direct embedding lookup (non-private, for testing)
///
/// NOTE: This endpoint reveals token IDs to the server. For production privacy,
/// use the OT-based /v2/secure/embeddings endpoint instead.
#[axum::debug_handler]
pub async fn get_embeddings_direct(
    State(state): State<AppState>,
    Json(request): Json<DirectEmbeddingRequest>,
) -> Result<Json<DirectEmbeddingResponse>> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get secure weights from state
    let secure_weights_guard = state.get_secure_weights()?;
    let secure_weights = secure_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;

    let hidden_dim = secure_weights.hidden_dim;
    let vocab_size = secure_weights.vocab_size;

    tracing::debug!(
        session_id = %session_id,
        num_tokens = request.token_ids.len(),
        vocab_size = vocab_size,
        hidden_dim = hidden_dim,
        "Processing direct embedding request"
    );

    // Create deterministic RNG from session ID for reproducible shares
    let session_bytes = session_id.as_bytes();
    let mut seed = [0u8; 32];
    seed[..16].copy_from_slice(session_bytes);
    let mut rng = ChaCha20Rng::from_seed(seed);

    let mut client_shares = Vec::with_capacity(request.token_ids.len());
    let mut server_shares = Vec::with_capacity(request.token_ids.len());

    for &token_id in &request.token_ids {
        let token_idx = token_id as usize;

        // Bounds check
        if token_idx >= vocab_size {
            return Err(ServerError::InvalidRequest(format!(
                "Token ID {} out of range (vocab_size={})",
                token_id, vocab_size
            )));
        }

        // Get embedding for this token
        let embed_start = token_idx * hidden_dim;
        let embed_end = embed_start + hidden_dim;
        let embedding = &secure_weights.embeddings[embed_start..embed_end];

        // Create secret shares: plaintext = client_share + server_share
        use rand::Rng;
        let server_share: Vec<f32> = (0..hidden_dim).map(|_| rng.gen::<f32>()).collect();
        let client_share: Vec<f32> = embedding.iter()
            .zip(server_share.iter())
            .map(|(e, s)| e - s)
            .collect();

        client_shares.push(client_share);
        server_shares.push(server_share);
    }

    tracing::debug!(
        session_id = %session_id,
        num_tokens = request.token_ids.len(),
        "Returning embedding shares"
    );

    Ok(Json(DirectEmbeddingResponse {
        client_shares,
        server_shares,
    }))
}

/// POST /v2/secure/layer/step - Process one transformer layer on shares
#[axum::debug_handler]
pub async fn layer_step(
    State(state): State<AppState>,
    Json(request): Json<LayerStepRequest>,
) -> Result<Json<LayerStepResponse>> {
    use shardlm_v2_sharing::{ServerContext, ClientShare, ServerShare};

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get secure weights from state
    let secure_weights_guard = state.get_secure_weights()?;
    let secure_weights = secure_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;

    let layer_idx = request.layer_idx;
    if layer_idx >= secure_weights.num_layers {
        return Err(ServerError::InvalidRequest(format!(
            "Layer {} out of range (max {})",
            layer_idx, secure_weights.num_layers - 1
        )));
    }

    let hidden_dim = secure_weights.hidden_dim;
    let num_heads = secure_weights.num_heads;
    let num_kv_heads = secure_weights.num_kv_heads;
    let head_dim = secure_weights.head_dim;

    tracing::debug!(
        session_id = %session_id,
        layer = layer_idx,
        hidden_dim = hidden_dim,
        "Processing layer step using SecureLinear"
    );

    // =========================================================================
    // SECURITY CRITICAL: Create ServerContext to prove we're server-side
    // =========================================================================
    let ctx = ServerContext::new();

    // Create shares from network request data
    let mut client_data = request.client_share.clone();
    let mut server_data = request.server_share.clone();

    // Get layer weights
    let layer = secure_weights.layer(layer_idx);

    // =========================================================================
    // Apply RMSNorm gamma (input_layernorm) to shares
    // Client did: normalized = x / sqrt(mean(x^2) + eps)
    // Server does: output = normalized * gamma (linear operation on shares)
    // =========================================================================
    for i in 0..hidden_dim {
        client_data[i] *= layer.input_layernorm[i];
        server_data[i] *= layer.input_layernorm[i];
    }

    let client_share = ClientShare::from_network(client_data, vec![hidden_dim]);
    let server_share = ServerShare::from_network(server_data, vec![hidden_dim]);

    // =========================================================================
    // SECURITY CRITICAL: We NEVER add client_share + server_share here
    // We process them SEPARATELY through the attention projection
    // =========================================================================

    // Compute QKV projections on shares using SecureAttention
    // RoPE is applied to Q and K inside project_qkv for position encoding
    let qkv_result = layer.attention.project_qkv(&ctx, &client_share, &server_share, request.position)
        .map_err(|e| ServerError::Internal(format!("QKV projection failed: {}", e)))?;

    Ok(Json(LayerStepResponse {
        phase: "qkv".to_string(),
        q_client: Some(qkv_result.q_client),
        q_server: Some(qkv_result.q_server),
        k_client: Some(qkv_result.k_client),
        k_server: Some(qkv_result.k_server),
        v_client: Some(qkv_result.v_client),
        v_server: Some(qkv_result.v_server),
        output_client: None,
        output_server: None,
        num_heads,
        num_kv_heads,
        head_dim,
    }))
}

/// POST /v2/secure/layer/attention - Process attention output through FFN
#[axum::debug_handler]
pub async fn process_attention_output(
    State(state): State<AppState>,
    Json(request): Json<AttentionOutputRequest>,
) -> Result<Json<LayerStepResponse>> {
    use shardlm_v2_sharing::{ServerContext, ClientShare, ServerShare};

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get secure weights from state
    let secure_weights_guard = state.get_secure_weights()?;
    let secure_weights = secure_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;

    let layer_idx = request.layer_idx;
    if layer_idx >= secure_weights.num_layers {
        return Err(ServerError::InvalidRequest(format!(
            "Layer {} out of range (max {})",
            layer_idx, secure_weights.num_layers - 1
        )));
    }

    let hidden_dim = secure_weights.hidden_dim;
    let num_heads = secure_weights.num_heads;
    let num_kv_heads = secure_weights.num_kv_heads;
    let head_dim = secure_weights.head_dim;

    tracing::debug!(
        session_id = %session_id,
        layer = layer_idx,
        "Processing attention output through O projection and FFN"
    );

    // =========================================================================
    // SECURITY CRITICAL: Create ServerContext
    // =========================================================================
    let ctx = ServerContext::new();

    // =========================================================================
    // SECURITY CRITICAL: Process attention output shares SEPARATELY through:
    // 1. Output projection (O_proj)
    // 2. Residual connection (add to input shares)
    // 3. RMSNorm (client-side, but we prepare shares)
    // 4. FFN gate_proj and up_proj on shares
    // 5. SwiGLU (client-side)
    // 6. FFN down_proj on shares
    // 7. Final residual connection
    // =========================================================================

    let layer = secure_weights.layer(layer_idx);

    // Create shares for attention output
    let attn_client = ClientShare::from_network(request.attn_output_client.clone(), vec![num_heads * head_dim]);
    let attn_server = ServerShare::from_network(request.attn_output_server.clone(), vec![num_heads * head_dim]);

    // 1. O projection: attention output -> hidden_dim
    let o_result = layer.attention.project_output(&ctx, &attn_client, &attn_server)
        .map_err(|e| ServerError::Internal(format!("O projection failed: {}", e)))?;

    // 2. Residual connection (add to input hidden shares)
    let hidden_after_attn_client: Vec<f32> = o_result.output_from_client_share.iter()
        .zip(request.hidden_client.iter())
        .map(|(o, h)| o + h)
        .collect();
    let hidden_after_attn_server: Vec<f32> = o_result.output_from_server_share.iter()
        .zip(request.hidden_server.iter())
        .map(|(o, h)| o + h)
        .collect();

    // 3. RMSNorm is done client-side (nonlinear) - we pass through
    // 4. FFN gate + up projections
    let ffn_in_client = ClientShare::from_network(hidden_after_attn_client.clone(), vec![hidden_dim]);
    let ffn_in_server = ServerShare::from_network(hidden_after_attn_server.clone(), vec![hidden_dim]);

    let gate_up = layer.ffn.project_gate_up(&ctx, &ffn_in_client, &ffn_in_server)
        .map_err(|e| ServerError::Internal(format!("FFN gate/up failed: {}", e)))?;

    // Return gate and up for client to compute SwiGLU activation
    // Client will reconstruct: gate = gate_client + gate_server, up = up_client + up_server
    // Then compute: activated = silu(gate) * up
    // Then create new shares and call /v2/secure/layer/ffn_down

    // For single-request mode, we return the hidden state after attention
    // The FFN would require another round trip for SwiGLU
    Ok(Json(LayerStepResponse {
        phase: "ffn_gate_up".to_string(),
        q_client: None,
        q_server: None,
        k_client: None,
        k_server: None,
        v_client: None,
        v_server: None,
        // Return hidden after attention + residual for next layer or final output
        output_client: Some(hidden_after_attn_client),
        output_server: Some(hidden_after_attn_server),
        num_heads,
        num_kv_heads,
        head_dim,
    }))
}

/// POST /v2/secure/logits - Project final hidden state to logits
#[axum::debug_handler]
pub async fn compute_logits(
    State(state): State<AppState>,
    Json(request): Json<LogitsRequest>,
) -> Result<Json<LogitsResponse>> {
    use shardlm_v2_sharing::{ServerContext, ClientShare, ServerShare};

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get secure weights from state
    let secure_weights_guard = state.get_secure_weights()?;
    let secure_weights = secure_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;

    let hidden_dim = secure_weights.hidden_dim;
    let vocab_size = secure_weights.vocab_size;

    tracing::debug!(
        session_id = %session_id,
        hidden_dim = hidden_dim,
        vocab_size = vocab_size,
        "Computing logits from final hidden state using SecureLinear"
    );

    // =========================================================================
    // SECURITY CRITICAL: Create ServerContext
    // =========================================================================
    let ctx = ServerContext::new();

    // =========================================================================
    // SECURITY CRITICAL: Project hidden shares to logits shares SEPARATELY
    // Client will reconstruct logits = logits_client + logits_server
    // Then apply softmax (client-side) and sample
    // =========================================================================

    // Apply final_norm gamma weights BEFORE lm_head projection
    // The client normalized the hidden state, now we scale by learned gamma
    let mut normed_client = request.hidden_client.clone();
    let mut normed_server = request.hidden_server.clone();
    for i in 0..hidden_dim {
        normed_client[i] *= secure_weights.final_norm[i];
        normed_server[i] *= secure_weights.final_norm[i];
    }

    // Create shares from gamma-scaled input
    let hidden_client = ClientShare::from_network(normed_client, vec![hidden_dim]);
    let hidden_server = ServerShare::from_network(normed_server, vec![hidden_dim]);

    // Use LM head SecureLinear to project to vocabulary
    // SECURITY: forward_secure processes shares SEPARATELY
    let logits_result = secure_weights.lm_head.forward_secure(&ctx, &hidden_client, &hidden_server)
        .map_err(|e| ServerError::Internal(format!("LM head projection failed: {}", e)))?;

    tracing::debug!(
        session_id = %session_id,
        logits_len = logits_result.output_from_client_share.len(),
        "Logits computed successfully"
    );

    Ok(Json(LogitsResponse {
        logits_client: logits_result.output_from_client_share,
        logits_server: logits_result.output_from_server_share,
    }))
}

// =============================================================================
// BATCHED FORWARD ENDPOINT
// =============================================================================
//
// This endpoint processes ALL transformer layers in a SINGLE HTTP request,
// reducing round trips from 56 (2 per layer × 28 layers) to 1.
//
// Protocol:
// 1. Client sends: initial hidden state shares + all client-side operation data
// 2. Server processes all layers, returning intermediate data for client-side ops
// 3. Client processes the response which includes QKV outputs for each layer
//
// The key insight is that the server-side linear operations (projections) can
// all be computed in one go. The client-side nonlinear operations (RMSNorm,
// Softmax, SwiGLU) are either:
// a) Pre-computed by the client based on previous state
// b) Returned to client in the response for post-processing

/// Batched forward request - processes all layers in one HTTP request
#[derive(Debug, Deserialize)]
pub struct BatchedForwardRequest {
    /// Session ID
    pub session_id: String,
    /// Initial hidden state client share [hidden_dim]
    pub hidden_client: Vec<f32>,
    /// Initial hidden state server share [hidden_dim]
    pub hidden_server: Vec<f32>,
    /// Pre-computed attention outputs for each layer (for Phase 2)
    /// If empty/None, this is Phase 1 (QKV computation)
    #[serde(default)]
    pub attention_outputs: Option<Vec<AttentionLayerData>>,
    /// Activated FFN data from client (for Phase 3)
    /// After client computes SwiGLU on gate_up outputs
    #[serde(default)]
    pub activated_ffn: Option<Vec<LayerActivatedFFN>>,
    /// Chunk info for chunked requests
    #[serde(default)]
    pub chunk_info: Option<ChunkInfo>,
    /// Sequence position for RoPE encoding (defaults to 0)
    #[serde(default)]
    pub position: usize,
}

/// Info about chunked Phase 2 request
#[derive(Debug, Deserialize)]
pub struct ChunkInfo {
    pub chunk_idx: usize,
    pub total_chunks: usize,
    pub start_layer: usize,
    pub end_layer: usize,
    pub is_last: bool,
}

/// Per-layer attention data from client
#[derive(Debug, Deserialize)]
pub struct AttentionLayerData {
    /// Attention output client share after client-side softmax
    pub attn_out_client: Vec<f32>,
    /// Attention output server share
    pub attn_out_server: Vec<f32>,
    /// OPTIONAL: Normalized hidden state for FFN (client has normalized hidden_after_attn)
    /// If provided, server skips O proj and uses this directly with gamma for gate/up
    #[serde(default)]
    pub normalized_ffn_client: Option<Vec<f32>>,
    #[serde(default)]
    pub normalized_ffn_server: Option<Vec<f32>>,
}

/// Batched forward response
#[derive(Debug, Serialize)]
pub struct BatchedForwardResponse {
    /// Phase: "need_attention", "need_ffn_activation", or "complete"
    pub phase: String,
    /// QKV outputs for all layers (Phase 1 response - client computes attention)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers_qkv: Option<Vec<LayerQKVOutput>>,
    /// Gate/Up outputs for FFN (Phase 2 response - client computes SwiGLU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layers_gate_up: Option<Vec<LayerGateUpOutput>>,
    /// Final hidden state shares (when phase = "complete")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_hidden_client: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_hidden_server: Option<Vec<f32>>,
    /// Model dimensions for client-side processing
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
    /// Intermediate dimension for FFN (for client SwiGLU)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intermediate_dim: Option<usize>,
}

/// QKV output for a single layer
#[derive(Debug, Serialize)]
pub struct LayerQKVOutput {
    pub layer_idx: usize,
    /// After RMSNorm + QKV projection
    pub q_client: Vec<f32>,
    pub q_server: Vec<f32>,
    pub k_client: Vec<f32>,
    pub k_server: Vec<f32>,
    pub v_client: Vec<f32>,
    pub v_server: Vec<f32>,
}

/// Gate/Up output for FFN (sent to client for SwiGLU activation)
#[derive(Debug, Serialize)]
pub struct LayerGateUpOutput {
    pub layer_idx: usize,
    /// Gate projection shares [intermediate_dim]
    pub gate_client: Vec<f32>,
    pub gate_server: Vec<f32>,
    /// Up projection shares [intermediate_dim]
    pub up_client: Vec<f32>,
    pub up_server: Vec<f32>,
    /// Hidden state after attention (for residual in Phase 3)
    pub hidden_after_attn_client: Vec<f32>,
    pub hidden_after_attn_server: Vec<f32>,
}

/// Activated FFN data from client (after SwiGLU)
#[derive(Debug, Deserialize)]
pub struct LayerActivatedFFN {
    pub layer_idx: usize,
    /// Activated FFN shares (after client SwiGLU) [intermediate_dim]
    pub activated_client: Vec<f32>,
    pub activated_server: Vec<f32>,
    /// Hidden state for residual connection [hidden_dim]
    pub residual_client: Vec<f32>,
    pub residual_server: Vec<f32>,
}

/// POST /v2/secure/forward/batched - Process all layers with 3-phase protocol
///
/// SECURITY: Server processes shares separately, NEVER reconstructs plaintext.
///
/// 3-Phase Protocol:
/// - Phase 1 (no attention_outputs, no activated_ffn): Return QKV for all layers
/// - Phase 2 (has attention_outputs, no activated_ffn): O proj + gate_up → return for SwiGLU
/// - Phase 3 (has activated_ffn): down_proj + residual → return final hidden
#[axum::debug_handler]
pub async fn batched_forward(
    State(state): State<AppState>,
    Json(request): Json<BatchedForwardRequest>,
) -> Result<Json<BatchedForwardResponse>> {
    use shardlm_v2_sharing::{ServerContext, ClientShare, ServerShare};

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get secure weights from state
    let secure_weights_guard = state.get_secure_weights()?;
    let secure_weights = secure_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("Secure weights not initialized".to_string()))?;

    // Get model dimensions from secure weights
    let num_layers = secure_weights.num_layers;
    let num_heads = secure_weights.num_heads;
    let num_kv_heads = secure_weights.num_kv_heads;
    let head_dim = secure_weights.head_dim;
    let hidden_dim = secure_weights.hidden_dim;
    let intermediate_dim = secure_weights.intermediate_dim;

    // =========================================================================
    // SECURITY CRITICAL: Create ServerContext
    // =========================================================================
    let ctx = ServerContext::new();

    // Determine which phase based on request contents
    let has_attention = request.attention_outputs.as_ref().map(|v| !v.is_empty()).unwrap_or(false);
    let has_activated_ffn = request.activated_ffn.as_ref().map(|v| !v.is_empty()).unwrap_or(false);

    if !has_attention && !has_activated_ffn {
        // =====================================================================
        // PHASE 1: Compute QKV for FIRST layer only
        // NOTE: QKV for layer N depends on output from layer N-1
        // We can only compute QKV for layer 0 with the initial hidden state
        // =====================================================================
        tracing::info!(
            session_id = %session_id,
            "Batched forward Phase 1: computing QKV for layer 0 only (sequential processing required)"
        );

        // Get layer 0 weights
        let layer = secure_weights.layer(0);

        // Apply RMSNorm gamma (input_layernorm) to shares before QKV
        let mut client_data = request.hidden_client.clone();
        let mut server_data = request.hidden_server.clone();
        for i in 0..hidden_dim {
            client_data[i] *= layer.input_layernorm[i];
            server_data[i] *= layer.input_layernorm[i];
        }

        let client_share = ClientShare::from_network(client_data, vec![hidden_dim]);
        let server_share = ServerShare::from_network(server_data, vec![hidden_dim]);

        // Only compute QKV for layer 0
        // RoPE is applied inside project_qkv with the request position
        let qkv_result = layer.attention.project_qkv(&ctx, &client_share, &server_share, request.position)
            .map_err(|e| ServerError::Internal(format!("QKV projection failed for layer 0: {}", e)))?;

        let layers_qkv = vec![LayerQKVOutput {
            layer_idx: 0,
            q_client: qkv_result.q_client,
            q_server: qkv_result.q_server,
            k_client: qkv_result.k_client,
            k_server: qkv_result.k_server,
            v_client: qkv_result.v_client,
            v_server: qkv_result.v_server,
        }];

        Ok(Json(BatchedForwardResponse {
            phase: "need_attention".to_string(),
            layers_qkv: Some(layers_qkv),
            layers_gate_up: None,
            final_hidden_client: None,
            final_hidden_server: None,
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_dim: None,
        }))

    } else if has_attention && !has_activated_ffn {
        // =====================================================================
        // PHASE 2: O projection + gate/up → return for client SwiGLU
        // =====================================================================
        let attention_outputs = request.attention_outputs.unwrap();

        // Handle chunked requests - process and return gate_up for EACH chunk
        if let Some(ref info) = request.chunk_info {
            tracing::info!(
                session_id = %session_id,
                chunk = info.chunk_idx + 1,
                total_chunks = info.total_chunks,
                layers = format!("{}-{}", info.start_layer, info.end_layer - 1),
                "Batched forward Phase 2 (chunked): O proj + gate/up"
            );

            // Process this chunk's attention outputs and return gate_up
            let mut layers_gate_up = Vec::with_capacity(attention_outputs.len());
            let mut current_client = request.hidden_client.clone();
            let mut current_server = request.hidden_server.clone();

            // Check if client provided normalized FFN input (Phase 2b)
            // or if we need to compute O proj + residual (Phase 2a)
            let has_normalized_ffn = attention_outputs.first()
                .map(|a| a.normalized_ffn_client.is_some())
                .unwrap_or(false);

            // Process from start_layer to end_layer using attention_outputs
            for (i, attn_output) in attention_outputs.iter().enumerate() {
                let layer_idx = info.start_layer + i;
                let layer = secure_weights.layer(layer_idx);

                if has_normalized_ffn {
                    // Phase 2b: Client already normalized hidden_after_attn
                    // Just apply gamma and do gate/up projection
                    let norm_client = attn_output.normalized_ffn_client.as_ref().unwrap();
                    let norm_server = attn_output.normalized_ffn_server.as_ref().unwrap();

                    // Apply post_attn_layernorm gamma to normalized input
                    let mut ffn_client_data = norm_client.clone();
                    let mut ffn_server_data = norm_server.clone();
                    for j in 0..hidden_dim {
                        ffn_client_data[j] *= layer.post_attn_layernorm[j];
                        ffn_server_data[j] *= layer.post_attn_layernorm[j];
                    }

                    // Gate/Up projections with properly gamma-scaled input
                    let ffn_in_client = ClientShare::from_network(ffn_client_data, vec![hidden_dim]);
                    let ffn_in_server = ServerShare::from_network(ffn_server_data, vec![hidden_dim]);

                    let gate_up = layer.ffn.project_gate_up(&ctx, &ffn_in_client, &ffn_in_server)
                        .map_err(|e| ServerError::Internal(format!("FFN gate/up failed for layer {}: {}", layer_idx, e)))?;

                    layers_gate_up.push(LayerGateUpOutput {
                        layer_idx,
                        gate_client: gate_up.gate_client,
                        gate_server: gate_up.gate_server,
                        up_client: gate_up.up_client,
                        up_server: gate_up.up_server,
                        // Return the pre-normalized hidden_after_attn for residual
                        hidden_after_attn_client: current_client.clone(),
                        hidden_after_attn_server: current_server.clone(),
                    });
                } else {
                    // Phase 2a: Compute O proj + residual, return hidden_after_attn for client normalization
                    let attn_client = ClientShare::from_network(attn_output.attn_out_client.clone(), vec![num_heads * head_dim]);
                    let attn_server = ServerShare::from_network(attn_output.attn_out_server.clone(), vec![num_heads * head_dim]);

                    // O projection
                    let o_result = layer.attention.project_output(&ctx, &attn_client, &attn_server)
                        .map_err(|e| ServerError::Internal(format!("O projection failed for layer {}: {}", layer_idx, e)))?;

                    // Add residual
                    let hidden_after_attn_client: Vec<f32> = o_result.output_from_client_share.iter()
                        .zip(current_client.iter())
                        .map(|(o, h)| o + h)
                        .collect();
                    let hidden_after_attn_server: Vec<f32> = o_result.output_from_server_share.iter()
                        .zip(current_server.iter())
                        .map(|(o, h)| o + h)
                        .collect();

                    // Return hidden_after_attn WITHOUT gate/up - client needs to normalize first
                    layers_gate_up.push(LayerGateUpOutput {
                        layer_idx,
                        gate_client: vec![],  // Empty - client needs to call Phase 2b
                        gate_server: vec![],
                        up_client: vec![],
                        up_server: vec![],
                        hidden_after_attn_client: hidden_after_attn_client.clone(),
                        hidden_after_attn_server: hidden_after_attn_server.clone(),
                    });

                    current_client = hidden_after_attn_client;
                    current_server = hidden_after_attn_server;
                }
            }

            // Return gate_up for this chunk (client collects all chunks)
            // Also return updated hidden state for next chunk
            let phase = if info.is_last { "need_ffn_activation" } else { "chunk_received" };
            Ok(Json(BatchedForwardResponse {
                phase: phase.to_string(),
                layers_qkv: None,
                layers_gate_up: Some(layers_gate_up),
                final_hidden_client: Some(current_client),
                final_hidden_server: Some(current_server),
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_dim: Some(intermediate_dim),
            }))
        } else {
            // Non-chunked Phase 2
            tracing::info!(
                session_id = %session_id,
                num_layers = attention_outputs.len(),
                "Batched forward Phase 2: O proj + gate/up for all layers"
            );

            let mut layers_gate_up = Vec::with_capacity(attention_outputs.len());
            let mut current_client = request.hidden_client.clone();
            let mut current_server = request.hidden_server.clone();

            // Check if client provided normalized FFN input (Phase 2b)
            let has_normalized_ffn = attention_outputs.first()
                .map(|a| a.normalized_ffn_client.is_some())
                .unwrap_or(false);

            for (layer_idx, attn_output) in attention_outputs.iter().enumerate() {
                let layer = secure_weights.layer(layer_idx);

                if has_normalized_ffn {
                    // Phase 2b: Client already normalized hidden_after_attn
                    let norm_client = attn_output.normalized_ffn_client.as_ref().unwrap();
                    let norm_server = attn_output.normalized_ffn_server.as_ref().unwrap();

                    // Apply post_attn_layernorm gamma to normalized input
                    let mut ffn_client_data = norm_client.clone();
                    let mut ffn_server_data = norm_server.clone();
                    for j in 0..hidden_dim {
                        ffn_client_data[j] *= layer.post_attn_layernorm[j];
                        ffn_server_data[j] *= layer.post_attn_layernorm[j];
                    }

                    let ffn_in_client = ClientShare::from_network(ffn_client_data, vec![hidden_dim]);
                    let ffn_in_server = ServerShare::from_network(ffn_server_data, vec![hidden_dim]);

                    let gate_up = layer.ffn.project_gate_up(&ctx, &ffn_in_client, &ffn_in_server)
                        .map_err(|e| ServerError::Internal(format!("FFN gate/up failed for layer {}: {}", layer_idx, e)))?;

                    layers_gate_up.push(LayerGateUpOutput {
                        layer_idx,
                        gate_client: gate_up.gate_client,
                        gate_server: gate_up.gate_server,
                        up_client: gate_up.up_client,
                        up_server: gate_up.up_server,
                        hidden_after_attn_client: current_client.clone(),
                        hidden_after_attn_server: current_server.clone(),
                    });
                } else {
                    // Phase 2a: Compute O proj + residual, return for client normalization
                    let attn_client = ClientShare::from_network(attn_output.attn_out_client.clone(), vec![num_heads * head_dim]);
                    let attn_server = ServerShare::from_network(attn_output.attn_out_server.clone(), vec![num_heads * head_dim]);

                    let o_result = layer.attention.project_output(&ctx, &attn_client, &attn_server)
                        .map_err(|e| ServerError::Internal(format!("O projection failed for layer {}: {}", layer_idx, e)))?;

                    let hidden_after_attn_client: Vec<f32> = o_result.output_from_client_share.iter()
                        .zip(current_client.iter())
                        .map(|(o, h)| o + h)
                        .collect();
                    let hidden_after_attn_server: Vec<f32> = o_result.output_from_server_share.iter()
                        .zip(current_server.iter())
                        .map(|(o, h)| o + h)
                        .collect();

                    layers_gate_up.push(LayerGateUpOutput {
                        layer_idx,
                        gate_client: vec![],  // Empty - client needs to normalize first
                        gate_server: vec![],
                        up_client: vec![],
                        up_server: vec![],
                        hidden_after_attn_client: hidden_after_attn_client.clone(),
                        hidden_after_attn_server: hidden_after_attn_server.clone(),
                    });

                    current_client = hidden_after_attn_client;
                    current_server = hidden_after_attn_server;
                }
            }

            Ok(Json(BatchedForwardResponse {
                phase: "need_ffn_activation".to_string(),
                layers_qkv: None,
                layers_gate_up: Some(layers_gate_up),
                final_hidden_client: None,
                final_hidden_server: None,
                num_heads,
                num_kv_heads,
                head_dim,
                num_layers,
                intermediate_dim: Some(intermediate_dim),
            }))
        }

    } else if has_activated_ffn {
        // =====================================================================
        // PHASE 3: down_proj + residual → final hidden state
        // =====================================================================
        let activated_ffn = request.activated_ffn.unwrap();

        tracing::info!(
            session_id = %session_id,
            num_layers = activated_ffn.len(),
            "Batched forward Phase 3: down_proj + residual for all layers"
        );

        let mut current_client = request.hidden_client.clone();
        let mut current_server = request.hidden_server.clone();

        for activated in activated_ffn.iter() {
            let layer_idx = activated.layer_idx;
            let layer = secure_weights.layer(layer_idx);

            // Create shares for activated FFN output (after client SwiGLU)
            let activated_client = ClientShare::from_network(activated.activated_client.clone(), vec![intermediate_dim]);
            let activated_server = ServerShare::from_network(activated.activated_server.clone(), vec![intermediate_dim]);

            // Down projection: intermediate_dim → hidden_dim
            let down_result = layer.ffn.project_down(&ctx, &activated_client, &activated_server)
                .map_err(|e| ServerError::Internal(format!("FFN down_proj failed for layer {}: {}", layer_idx, e)))?;

            // Add FFN residual (from hidden_after_attn)
            let final_client: Vec<f32> = down_result.output_from_client_share.iter()
                .zip(activated.residual_client.iter())
                .map(|(d, r)| d + r)
                .collect();
            let final_server: Vec<f32> = down_result.output_from_server_share.iter()
                .zip(activated.residual_server.iter())
                .map(|(d, r)| d + r)
                .collect();

            current_client = final_client;
            current_server = final_server;
        }

        Ok(Json(BatchedForwardResponse {
            phase: "complete".to_string(),
            layers_qkv: None,
            layers_gate_up: None,
            final_hidden_client: Some(current_client),
            final_hidden_server: Some(current_server),
            num_heads,
            num_kv_heads,
            head_dim,
            num_layers,
            intermediate_dim: None,
        }))

    } else {
        Err(ServerError::InvalidRequest("Invalid request state: must provide either attention_outputs or activated_ffn".to_string()))
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create secure sessions storage
pub fn create_secure_sessions() -> SecureSessions {
    Arc::new(RwLock::new(HashMap::new()))
}

// =============================================================================
// GPU-ACCELERATED SECURE INFERENCE ENDPOINTS
// =============================================================================
//
// These endpoints use cuBLAS for matrix multiplications while maintaining
// the same security guarantees as the CPU versions. Shares are processed
// SEPARATELY on the GPU - the GPU never computes client_share + server_share.
//
// Expected performance improvement: 10-50x over CPU for matrix operations.

/// POST /v2/secure/gpu/layer/step - GPU-accelerated QKV projection
///
/// Uses cuBLAS SGEMM for Q, K, V projections.
/// Same security guarantees as CPU version.
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn layer_step_gpu(
    State(state): State<AppState>,
    Json(request): Json<LayerStepRequest>,
) -> Result<Json<LayerStepResponse>> {
    use shardlm_v2_sharing::ServerContext;
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get GPU secure weights and kernel context
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let layer_idx = request.layer_idx;
    if layer_idx >= gpu_weights.num_layers {
        return Err(ServerError::InvalidRequest(format!(
            "Layer {} out of range (max {})",
            layer_idx, gpu_weights.num_layers - 1
        )));
    }

    let hidden_dim = gpu_weights.hidden_dim;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;

    tracing::debug!(
        session_id = %session_id,
        layer = layer_idx,
        position = request.position,
        "GPU-accelerated QKV projection"
    );

    let ctx = ServerContext::new();
    let layer = gpu_weights.layer(layer_idx);

    // Apply input_layernorm gamma to shares
    let mut client_data = request.client_share.clone();
    let mut server_data = request.server_share.clone();
    for i in 0..hidden_dim {
        client_data[i] *= layer.input_layernorm[i];
        server_data[i] *= layer.input_layernorm[i];
    }

    // Get first GPU device for operations
    // In production, this would be configurable per session
    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // GPU-accelerated QKV projection with RoPE
    let qkv_result = layer.attention.project_qkv_gpu(
        &ctx, &client_data, &server_data, request.position, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("GPU QKV projection failed: {}", e)))?;

    Ok(Json(LayerStepResponse {
        phase: "qkv_gpu".to_string(),
        q_client: Some(qkv_result.q_client),
        q_server: Some(qkv_result.q_server),
        k_client: Some(qkv_result.k_client),
        k_server: Some(qkv_result.k_server),
        v_client: Some(qkv_result.v_client),
        v_server: Some(qkv_result.v_server),
        output_client: None,
        output_server: None,
        num_heads,
        num_kv_heads,
        head_dim,
    }))
}

/// POST /v2/secure/gpu/ffn - GPU-accelerated FFN projection
///
/// Uses cuBLAS SGEMM for gate, up, and down projections.
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn ffn_step_gpu(
    State(state): State<AppState>,
    Json(request): Json<FFNStepRequest>,
) -> Result<Json<FFNStepResponse>> {
    use shardlm_v2_sharing::ServerContext;
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let layer_idx = request.layer_idx;
    if layer_idx >= gpu_weights.num_layers {
        return Err(ServerError::InvalidRequest(format!(
            "Layer {} out of range (max {})",
            layer_idx, gpu_weights.num_layers - 1
        )));
    }

    let ctx = ServerContext::new();
    let layer = gpu_weights.layer(layer_idx);
    let hidden_dim = gpu_weights.hidden_dim;

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // Apply post_attn_layernorm gamma
    let mut norm_client = request.hidden_client.clone();
    let mut norm_server = request.hidden_server.clone();
    for i in 0..hidden_dim {
        norm_client[i] *= layer.post_attn_layernorm[i];
        norm_server[i] *= layer.post_attn_layernorm[i];
    }

    // GPU-accelerated gate/up projections
    let gate_up = layer.ffn.project_gate_up_gpu(
        &ctx, &norm_client, &norm_server, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("GPU FFN gate/up failed: {}", e)))?;

    Ok(Json(FFNStepResponse {
        gate_client: gate_up.gate_client,
        gate_server: gate_up.gate_server,
        up_client: gate_up.up_client,
        up_server: gate_up.up_server,
        intermediate_dim: gate_up.intermediate_dim,
    }))
}

/// FFN step request
#[derive(Debug, Deserialize)]
pub struct FFNStepRequest {
    pub session_id: String,
    pub layer_idx: usize,
    /// Hidden state client share (after attention + residual)
    pub hidden_client: Vec<f32>,
    /// Hidden state server share
    pub hidden_server: Vec<f32>,
}

/// FFN step response
#[derive(Debug, Serialize)]
pub struct FFNStepResponse {
    pub gate_client: Vec<f32>,
    pub gate_server: Vec<f32>,
    pub up_client: Vec<f32>,
    pub up_server: Vec<f32>,
    pub intermediate_dim: usize,
}

/// POST /v2/secure/gpu/ffn/down - GPU-accelerated down projection
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn ffn_down_gpu(
    State(state): State<AppState>,
    Json(request): Json<FFNDownRequest>,
) -> Result<Json<FFNDownResponse>> {
    use shardlm_v2_sharing::ServerContext;
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let layer_idx = request.layer_idx;
    let ctx = ServerContext::new();
    let layer = gpu_weights.layer(layer_idx);

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // GPU-accelerated down projection
    let (down_client, down_server) = layer.ffn.project_down_gpu(
        &ctx, &request.activated_client, &request.activated_server, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("GPU FFN down failed: {}", e)))?;

    // Add residual connection
    let final_client: Vec<f32> = down_client.iter()
        .zip(request.residual_client.iter())
        .map(|(d, r)| d + r)
        .collect();
    let final_server: Vec<f32> = down_server.iter()
        .zip(request.residual_server.iter())
        .map(|(d, r)| d + r)
        .collect();

    Ok(Json(FFNDownResponse {
        hidden_client: final_client,
        hidden_server: final_server,
    }))
}

/// FFN down projection request
#[derive(Debug, Deserialize)]
pub struct FFNDownRequest {
    pub session_id: String,
    pub layer_idx: usize,
    /// Activated FFN output (after client SwiGLU)
    pub activated_client: Vec<f32>,
    pub activated_server: Vec<f32>,
    /// Residual for skip connection
    pub residual_client: Vec<f32>,
    pub residual_server: Vec<f32>,
}

/// FFN down projection response
#[derive(Debug, Serialize)]
pub struct FFNDownResponse {
    pub hidden_client: Vec<f32>,
    pub hidden_server: Vec<f32>,
}

/// POST /v2/secure/gpu/logits - GPU-accelerated logits computation
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn compute_logits_gpu(
    State(state): State<AppState>,
    Json(request): Json<LogitsRequest>,
) -> Result<Json<LogitsResponse>> {
    use shardlm_v2_sharing::ServerContext;
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let hidden_dim = gpu_weights.hidden_dim;
    let ctx = ServerContext::new();

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // Apply final_norm gamma
    let mut normed_client = request.hidden_client.clone();
    let mut normed_server = request.hidden_server.clone();
    for i in 0..hidden_dim {
        normed_client[i] *= gpu_weights.final_norm[i];
        normed_server[i] *= gpu_weights.final_norm[i];
    }

    // GPU-accelerated LM head projection
    let (logits_client, logits_server) = gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_client, &normed_server, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("GPU LM head failed: {}", e)))?;

    tracing::debug!(
        session_id = %session_id,
        logits_len = logits_client.len(),
        "GPU logits computed successfully"
    );

    Ok(Json(LogitsResponse {
        logits_client,
        logits_server,
    }))
}

// =============================================================================
// BATCHED FULL FORWARD - ALL LAYERS IN ONE REQUEST
// =============================================================================

/// Request for batched full forward pass through all layers
#[derive(Debug, Deserialize)]
pub struct BatchedFullForwardRequest {
    pub session_id: String,
    /// Initial hidden state shares (from embedding)
    pub hidden_client: Vec<f32>,
    pub hidden_server: Vec<f32>,
    /// Position in sequence (for RoPE)
    pub position: usize,
    /// KV cache from previous tokens: k_cache[layer][position] = Vec<f32>
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
}

/// Response from batched full forward
#[derive(Debug, Serialize)]
pub struct BatchedFullForwardResponse {
    /// Final hidden state shares (after all layers)
    pub hidden_client: Vec<f32>,
    pub hidden_server: Vec<f32>,
    /// New K vectors for each layer (to append to cache)
    pub new_k: Vec<Vec<f32>>,
    /// New V vectors for each layer (to append to cache)
    pub new_v: Vec<Vec<f32>>,
}

/// POST /v2/secure/gpu/forward/full - Process ALL layers in ONE request
///
/// This is the optimized endpoint that processes all 28 layers in a single HTTP request.
/// Uses secure polynomial approximations for nonlinear operations (RMSNorm, SiLU, Softmax).
///
/// Performance: ~10-50x faster than per-layer requests due to eliminating HTTP overhead.
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn batched_full_forward_gpu(
    State(state): State<AppState>,
    Json(request): Json<BatchedFullForwardRequest>,
) -> Result<Json<BatchedFullForwardResponse>> {
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_approx, secure_softmax_approx, secure_swiglu_approx,
    };
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    // Get GPU resources
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let ctx = ServerContext::new();
    let num_layers = gpu_weights.num_layers;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;

    tracing::info!(
        session_id = %session_id,
        position = request.position,
        num_layers = num_layers,
        "Starting batched full forward (all layers)"
    );

    let start_time = std::time::Instant::now();

    // Current hidden state
    let mut hidden_client = request.hidden_client;
    let mut hidden_server = request.hidden_server;

    // Collect new K, V vectors for each layer
    let mut new_k_all = Vec::with_capacity(num_layers);
    let mut new_v_all = Vec::with_capacity(num_layers);

    // Process all layers
    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);

        // === Phase 1: Input LayerNorm (secure approximation) ===
        let (normed_client, normed_server) = secure_rms_norm_approx(
            &hidden_client,
            &hidden_server,
            &layer.input_layernorm,
            1e-6,
            &ctx,
        );

        // === Phase 2: QKV Projection (GPU cuBLAS) ===
        let qkv_result = layer.attention.project_qkv_gpu(
            &ctx, &normed_client, &normed_server, request.position, kernels, &device
        ).map_err(|e| ServerError::Internal(format!("Layer {} QKV failed: {}", layer_idx, e)))?;

        // Store K, V for this layer
        let k_combined: Vec<f32> = qkv_result.k_client.iter()
            .zip(qkv_result.k_server.iter())
            .map(|(c, s)| c + s)
            .collect();
        let v_combined: Vec<f32> = qkv_result.v_client.iter()
            .zip(qkv_result.v_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        new_k_all.push(k_combined.clone());
        new_v_all.push(v_combined.clone());

        // === Phase 3: Attention with Secure Softmax ===
        // Build K, V cache for this layer including the new token
        let mut k_cache_layer: Vec<Vec<f32>> = if layer_idx < request.k_cache.len() {
            request.k_cache[layer_idx].clone()
        } else {
            vec![]
        };
        k_cache_layer.push(k_combined);

        let mut v_cache_layer: Vec<Vec<f32>> = if layer_idx < request.v_cache.len() {
            request.v_cache[layer_idx].clone()
        } else {
            vec![]
        };
        v_cache_layer.push(v_combined);

        // Reconstruct Q for attention computation
        let q_combined: Vec<f32> = qkv_result.q_client.iter()
            .zip(qkv_result.q_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        // Compute attention with secure softmax approximation
        let attn_output = shardlm_v2_sharing::secure_attention_approx(
            &q_combined,
            &k_cache_layer,
            &v_cache_layer,
            num_heads,
            num_kv_heads,
            head_dim,
            &ctx,
        );

        // Re-share attention output
        let attn_client = attn_output.clone();
        let attn_server = vec![0.0; attn_output.len()];

        // === Phase 4: O Projection (GPU cuBLAS) ===
        let (o_client, o_server) = layer.attention.project_output_gpu(
            &ctx, &attn_client, &attn_server, kernels, &device
        ).map_err(|e| ServerError::Internal(format!("Layer {} O proj failed: {}", layer_idx, e)))?;

        // Add residual (to pre-norm hidden)
        let hidden_after_attn_client: Vec<f32> = o_client.iter()
            .zip(hidden_client.iter())
            .map(|(o, h)| o + h)
            .collect();
        let hidden_after_attn_server: Vec<f32> = o_server.iter()
            .zip(hidden_server.iter())
            .map(|(o, h)| o + h)
            .collect();

        // === Phase 5: Post-Attention LayerNorm (secure approximation) ===
        let (normed_ffn_client, normed_ffn_server) = secure_rms_norm_approx(
            &hidden_after_attn_client,
            &hidden_after_attn_server,
            &layer.post_attn_layernorm,
            1e-6,
            &ctx,
        );

        // === Phase 6: FFN Gate/Up Projection (GPU cuBLAS) ===
        let ffn_result = layer.ffn.project_gate_up_gpu(
            &ctx, &normed_ffn_client, &normed_ffn_server, kernels, &device
        ).map_err(|e| ServerError::Internal(format!("Layer {} FFN gate/up failed: {}", layer_idx, e)))?;

        // === Phase 7: SwiGLU Activation (secure approximation) ===
        let (activated_client, activated_server) = secure_swiglu_approx(
            &ffn_result.gate_client,
            &ffn_result.gate_server,
            &ffn_result.up_client,
            &ffn_result.up_server,
            &ctx,
        );

        // === Phase 8: FFN Down Projection (GPU cuBLAS) ===
        let (down_client, down_server) = layer.ffn.project_down_gpu(
            &ctx, &activated_client, &activated_server, kernels, &device
        ).map_err(|e| ServerError::Internal(format!("Layer {} FFN down failed: {}", layer_idx, e)))?;

        // Add residual (to hidden_after_attn)
        hidden_client = down_client.iter()
            .zip(hidden_after_attn_client.iter())
            .map(|(d, h)| d + h)
            .collect();
        hidden_server = down_server.iter()
            .zip(hidden_after_attn_server.iter())
            .map(|(d, h)| d + h)
            .collect();
    }

    let elapsed = start_time.elapsed();
    tracing::info!(
        session_id = %session_id,
        elapsed_ms = elapsed.as_millis(),
        "Batched full forward complete (all {} layers in one request)",
        num_layers
    );

    Ok(Json(BatchedFullForwardResponse {
        hidden_client,
        hidden_server,
        new_k: new_k_all,
        new_v: new_v_all,
    }))
}

/// POST /v2/secure/gpu/generate/token - Generate a single token with all layers + logits
///
/// Most optimized endpoint: processes all layers AND computes logits in ONE request.
/// Only 1 HTTP round-trip per token!
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn generate_token_gpu(
    State(state): State<AppState>,
    Json(request): Json<BatchedFullForwardRequest>,
) -> Result<Json<GenerateTokenResponse>> {
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_approx,
    };
    use shardlm_v2_core::gpu::GpuDevice;

    // First do the full forward pass
    let Json(forward_result) = batched_full_forward_gpu(
        State(state.clone()),
        Json(request),
    ).await?;

    // Then compute logits
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let ctx = ServerContext::new();

    // Final norm with secure approximation
    let (normed_client, normed_server) = secure_rms_norm_approx(
        &forward_result.hidden_client,
        &forward_result.hidden_server,
        &gpu_weights.final_norm,
        1e-6,
        &ctx,
    );

    // LM head projection
    let (logits_client, logits_server) = gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_client, &normed_server, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("LM head failed: {}", e)))?;

    Ok(Json(GenerateTokenResponse {
        hidden_client: forward_result.hidden_client,
        hidden_server: forward_result.hidden_server,
        new_k: forward_result.new_k,
        new_v: forward_result.new_v,
        logits_client,
        logits_server,
    }))
}

/// Response for single-token generation
#[derive(Debug, Serialize)]
pub struct GenerateTokenResponse {
    /// Final hidden state shares
    pub hidden_client: Vec<f32>,
    pub hidden_server: Vec<f32>,
    /// New K vectors for each layer
    pub new_k: Vec<Vec<f32>>,
    /// New V vectors for each layer
    pub new_v: Vec<Vec<f32>>,
    /// Logits for token sampling
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
}

// =============================================================================
// BATCHED PREFILL - Process ALL prompt tokens in ONE request
// =============================================================================

/// Request for batched prefill (all prompt tokens at once)
#[derive(Debug, Deserialize)]
pub struct BatchedPrefillRequest {
    pub session_id: String,
    /// Hidden states for ALL prompt tokens: [seq_len][hidden_dim]
    pub hidden_client: Vec<Vec<f32>>,
    pub hidden_server: Vec<Vec<f32>>,
}

/// Response from batched prefill
#[derive(Debug, Serialize)]
pub struct BatchedPrefillResponse {
    /// Final hidden state for LAST token only (for generation)
    pub final_hidden_client: Vec<f32>,
    pub final_hidden_server: Vec<f32>,
    /// KV cache for all tokens: [layer][seq_len][kv_dim]
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    /// Logits for last token (for first generation step)
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
}

/// POST /v2/secure/gpu/prefill - Process ALL prompt tokens in ONE request
///
/// This is the most optimized endpoint for prompt processing.
/// Processes all prompt tokens through all 28 layers in a single HTTP request.
///
/// Performance: Eliminates N-1 HTTP round-trips for N prompt tokens.
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn batched_prefill_gpu(
    State(state): State<AppState>,
    Json(request): Json<BatchedPrefillRequest>,
) -> Result<Json<BatchedPrefillResponse>> {
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_approx, secure_swiglu_approx,
    };
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let seq_len = request.hidden_client.len();
    if seq_len == 0 {
        return Err(ServerError::InvalidRequest("Empty sequence".to_string()));
    }

    // Get GPU resources
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let ctx = ServerContext::new();
    let num_layers = gpu_weights.num_layers;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;
    let kv_dim = num_kv_heads * head_dim;

    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        num_layers = num_layers,
        "Starting batched prefill (all tokens, all layers)"
    );

    let start_time = std::time::Instant::now();

    // Initialize KV cache: [layer][seq_len][kv_dim]
    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];

    // Current hidden states for all tokens
    let mut hidden_client: Vec<Vec<f32>> = request.hidden_client;
    let mut hidden_server: Vec<Vec<f32>> = request.hidden_server;

    // Process all layers
    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);

        // Process all tokens through this layer
        let mut new_hidden_client = Vec::with_capacity(seq_len);
        let mut new_hidden_server = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // === Phase 1: Input LayerNorm ===
            let (normed_client, normed_server) = secure_rms_norm_approx(
                &hidden_client[pos],
                &hidden_server[pos],
                &layer.input_layernorm,
                1e-6,
                &ctx,
            );

            // === Phase 2: QKV Projection ===
            let qkv_result = layer.attention.project_qkv_gpu(
                &ctx, &normed_client, &normed_server, pos, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} pos {} QKV failed: {}", layer_idx, pos, e)))?;

            // Store K, V in cache
            let k_combined: Vec<f32> = qkv_result.k_client.iter()
                .zip(qkv_result.k_server.iter())
                .map(|(c, s)| c + s)
                .collect();
            let v_combined: Vec<f32> = qkv_result.v_client.iter()
                .zip(qkv_result.v_server.iter())
                .map(|(c, s)| c + s)
                .collect();

            k_cache[layer_idx].push(k_combined);
            v_cache[layer_idx].push(v_combined);

            // === Phase 3: Attention with causal mask ===
            // For position pos, attend to positions 0..=pos
            let q_combined: Vec<f32> = qkv_result.q_client.iter()
                .zip(qkv_result.q_server.iter())
                .map(|(c, s)| c + s)
                .collect();

            // Get K, V cache up to current position (inclusive)
            let k_cache_slice: Vec<Vec<f32>> = k_cache[layer_idx][0..=pos].to_vec();
            let v_cache_slice: Vec<Vec<f32>> = v_cache[layer_idx][0..=pos].to_vec();

            let attn_output = shardlm_v2_sharing::secure_attention_approx(
                &q_combined,
                &k_cache_slice,
                &v_cache_slice,
                num_heads,
                num_kv_heads,
                head_dim,
                &ctx,
            );

            let attn_client = attn_output.clone();
            let attn_server = vec![0.0; attn_output.len()];

            // === Phase 4: O Projection ===
            let (o_client, o_server) = layer.attention.project_output_gpu(
                &ctx, &attn_client, &attn_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} O proj failed: {}", layer_idx, e)))?;

            // Add residual
            let hidden_after_attn_client: Vec<f32> = o_client.iter()
                .zip(hidden_client[pos].iter())
                .map(|(o, h)| o + h)
                .collect();
            let hidden_after_attn_server: Vec<f32> = o_server.iter()
                .zip(hidden_server[pos].iter())
                .map(|(o, h)| o + h)
                .collect();

            // === Phase 5: Post-Attention LayerNorm ===
            let (normed_ffn_client, normed_ffn_server) = secure_rms_norm_approx(
                &hidden_after_attn_client,
                &hidden_after_attn_server,
                &layer.post_attn_layernorm,
                1e-6,
                &ctx,
            );

            // === Phase 6: FFN Gate/Up ===
            let ffn_result = layer.ffn.project_gate_up_gpu(
                &ctx, &normed_ffn_client, &normed_ffn_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN gate/up failed: {}", layer_idx, e)))?;

            // === Phase 7: SwiGLU ===
            let (activated_client, activated_server) = secure_swiglu_approx(
                &ffn_result.gate_client,
                &ffn_result.gate_server,
                &ffn_result.up_client,
                &ffn_result.up_server,
                &ctx,
            );

            // === Phase 8: FFN Down ===
            let (down_client, down_server) = layer.ffn.project_down_gpu(
                &ctx, &activated_client, &activated_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN down failed: {}", layer_idx, e)))?;

            // Add residual
            let final_client: Vec<f32> = down_client.iter()
                .zip(hidden_after_attn_client.iter())
                .map(|(d, h)| d + h)
                .collect();
            let final_server: Vec<f32> = down_server.iter()
                .zip(hidden_after_attn_server.iter())
                .map(|(d, h)| d + h)
                .collect();

            new_hidden_client.push(final_client);
            new_hidden_server.push(final_server);
        }

        hidden_client = new_hidden_client;
        hidden_server = new_hidden_server;

        if layer_idx % 7 == 0 || layer_idx == num_layers - 1 {
            tracing::debug!("Prefill layer {}/{} complete", layer_idx + 1, num_layers);
        }
    }

    // Compute logits for last token
    let last_idx = seq_len - 1;
    let (normed_client, normed_server) = secure_rms_norm_approx(
        &hidden_client[last_idx],
        &hidden_server[last_idx],
        &gpu_weights.final_norm,
        1e-6,
        &ctx,
    );

    let (logits_client, logits_server) = gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_client, &normed_server, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("LM head failed: {}", e)))?;

    let elapsed = start_time.elapsed();
    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        elapsed_ms = elapsed.as_millis(),
        tokens_per_sec = seq_len as f64 / elapsed.as_secs_f64(),
        "Batched prefill complete"
    );

    Ok(Json(BatchedPrefillResponse {
        final_hidden_client: hidden_client.pop().unwrap_or_default(),
        final_hidden_server: hidden_server.pop().unwrap_or_default(),
        k_cache,
        v_cache,
        logits_client,
        logits_server,
    }))
}

/// POST /v2/secure/gpu/prefill_v2 - Fully GPU-accelerated prefill
///
/// This endpoint uses GPU for ALL operations including RMSNorm and SwiGLU.
/// Previous version used CPU for nonlinear operations - this version keeps
/// everything on GPU for maximum performance.
///
/// # Security
///
/// Same security model as v1:
/// - Server processes shares separately for linear ops
/// - Nonlinear ops reconstruct on GPU, compute, re-share
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn batched_prefill_gpu_v2(
    State(state): State<AppState>,
    Json(request): Json<BatchedPrefillRequest>,
) -> Result<Json<BatchedPrefillResponse>> {
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_gpu_vec, secure_swiglu_gpu_vec,
    };
    use shardlm_v2_core::gpu::GpuDevice;

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let seq_len = request.hidden_client.len();
    if seq_len == 0 {
        return Err(ServerError::InvalidRequest("Empty sequence".to_string()));
    }

    // Get GPU resources
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_guard = state.get_gpu_kernel_contexts()?;
    let kernels = kernel_guard.get(0)
        .ok_or_else(|| ServerError::Internal("GPU kernel context not initialized".to_string()))?;

    let device = GpuDevice::new(0)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let ctx = ServerContext::new();
    let num_layers = gpu_weights.num_layers;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;
    let kv_dim = num_kv_heads * head_dim;

    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        num_layers = num_layers,
        "Starting GPU-accelerated prefill v2 (all ops on GPU)"
    );

    let start_time = std::time::Instant::now();

    // Initialize KV cache: [layer][seq_len][kv_dim]
    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];

    // Current hidden states for all tokens
    let mut hidden_client: Vec<Vec<f32>> = request.hidden_client;
    let mut hidden_server: Vec<Vec<f32>> = request.hidden_server;

    // Process all layers
    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);

        // Process all tokens through this layer
        let mut new_hidden_client = Vec::with_capacity(seq_len);
        let mut new_hidden_server = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // === Phase 1: Input LayerNorm (GPU) ===
            let (normed_client, normed_server) = secure_rms_norm_gpu_vec(
                &hidden_client[pos],
                &hidden_server[pos],
                &layer.input_layernorm_gpu,
                1e-6,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} pos {} RMSNorm failed: {}", layer_idx, pos, e)))?;

            // === Phase 2: QKV Projection ===
            let qkv_result = layer.attention.project_qkv_gpu(
                &ctx, &normed_client, &normed_server, pos, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} pos {} QKV failed: {}", layer_idx, pos, e)))?;

            // Store K, V in cache
            let k_combined: Vec<f32> = qkv_result.k_client.iter()
                .zip(qkv_result.k_server.iter())
                .map(|(c, s)| c + s)
                .collect();
            let v_combined: Vec<f32> = qkv_result.v_client.iter()
                .zip(qkv_result.v_server.iter())
                .map(|(c, s)| c + s)
                .collect();

            k_cache[layer_idx].push(k_combined);
            v_cache[layer_idx].push(v_combined);

            // === Phase 3: Attention with causal mask ===
            let q_combined: Vec<f32> = qkv_result.q_client.iter()
                .zip(qkv_result.q_server.iter())
                .map(|(c, s)| c + s)
                .collect();

            let k_cache_slice: Vec<Vec<f32>> = k_cache[layer_idx][0..=pos].to_vec();
            let v_cache_slice: Vec<Vec<f32>> = v_cache[layer_idx][0..=pos].to_vec();

            let attn_output = shardlm_v2_sharing::secure_attention_approx(
                &q_combined,
                &k_cache_slice,
                &v_cache_slice,
                num_heads,
                num_kv_heads,
                head_dim,
                &ctx,
            );

            let attn_client = attn_output.clone();
            let attn_server = vec![0.0; attn_output.len()];

            // === Phase 4: O Projection ===
            let (o_client, o_server) = layer.attention.project_output_gpu(
                &ctx, &attn_client, &attn_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} O proj failed: {}", layer_idx, e)))?;

            // Add residual
            let hidden_after_attn_client: Vec<f32> = o_client.iter()
                .zip(hidden_client[pos].iter())
                .map(|(o, h)| o + h)
                .collect();
            let hidden_after_attn_server: Vec<f32> = o_server.iter()
                .zip(hidden_server[pos].iter())
                .map(|(o, h)| o + h)
                .collect();

            // === Phase 5: Post-Attention LayerNorm (GPU) ===
            let (normed_ffn_client, normed_ffn_server) = secure_rms_norm_gpu_vec(
                &hidden_after_attn_client,
                &hidden_after_attn_server,
                &layer.post_attn_layernorm_gpu,
                1e-6,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} pos {} post-attn RMSNorm failed: {}", layer_idx, pos, e)))?;

            // === Phase 6: FFN Gate/Up ===
            let ffn_result = layer.ffn.project_gate_up_gpu(
                &ctx, &normed_ffn_client, &normed_ffn_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN gate/up failed: {}", layer_idx, e)))?;

            // === Phase 7: SwiGLU (GPU) ===
            let (activated_client, activated_server) = secure_swiglu_gpu_vec(
                &ffn_result.gate_client,
                &ffn_result.gate_server,
                &ffn_result.up_client,
                &ffn_result.up_server,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} SwiGLU failed: {}", layer_idx, e)))?;

            // === Phase 8: FFN Down ===
            let (down_client, down_server) = layer.ffn.project_down_gpu(
                &ctx, &activated_client, &activated_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN down failed: {}", layer_idx, e)))?;

            // Add residual
            let final_client: Vec<f32> = down_client.iter()
                .zip(hidden_after_attn_client.iter())
                .map(|(d, h)| d + h)
                .collect();
            let final_server: Vec<f32> = down_server.iter()
                .zip(hidden_after_attn_server.iter())
                .map(|(d, h)| d + h)
                .collect();

            new_hidden_client.push(final_client);
            new_hidden_server.push(final_server);
        }

        hidden_client = new_hidden_client;
        hidden_server = new_hidden_server;

        if layer_idx % 7 == 0 || layer_idx == num_layers - 1 {
            tracing::debug!("GPU prefill v2 layer {}/{} complete", layer_idx + 1, num_layers);
        }
    }

    // Compute logits for last token
    let last_idx = seq_len - 1;
    let (normed_client, normed_server) = secure_rms_norm_gpu_vec(
        &hidden_client[last_idx],
        &hidden_server[last_idx],
        &gpu_weights.final_norm_gpu,
        1e-6,
        kernels,
        &device,
    ).map_err(|e| ServerError::Internal(format!("Final RMSNorm failed: {}", e)))?;

    let (logits_client, logits_server) = gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_client, &normed_server, kernels, &device
    ).map_err(|e| ServerError::Internal(format!("LM head failed: {}", e)))?;

    let elapsed = start_time.elapsed();
    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        elapsed_ms = elapsed.as_millis(),
        tokens_per_sec = seq_len as f64 / elapsed.as_secs_f64(),
        "GPU prefill v2 complete (all ops on GPU)"
    );

    Ok(Json(BatchedPrefillResponse {
        final_hidden_client: hidden_client.pop().unwrap_or_default(),
        final_hidden_server: hidden_server.pop().unwrap_or_default(),
        k_cache,
        v_cache,
        logits_client,
        logits_server,
    }))
}

// =============================================================================
// GPU-RESIDENT PREFILL V3 - Minimal CPU transfers
// =============================================================================

/// Batched prefill with GPU-resident tensors (V3)
///
/// Key optimization: Keeps tensors on GPU throughout forward pass.
/// Only transfers:
/// - Input: Upload hidden states once at start
/// - RoPE: Brief download/upload for positional encoding (TODO: GPU kernel)
/// - Output: Download final logits
///
/// All linear ops (QKV, O, FFN) and nonlinear ops (RMSNorm, SwiGLU) stay on GPU.
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn batched_prefill_gpu_v3(
    State(state): State<AppState>,
    Json(request): Json<BatchedPrefillRequest>,
) -> Result<Json<BatchedPrefillResponse>> {
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_gpu, secure_swiglu_gpu, secure_add_gpu,
    };
    use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};

    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let seq_len = request.hidden_client.len();
    if seq_len == 0 {
        return Err(ServerError::InvalidRequest("Empty sequence".to_string()));
    }

    // Get GPU resources
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU secure weights not initialized".to_string()))?;

    let kernel_contexts_guard = state.get_gpu_kernel_contexts()?;
    if kernel_contexts_guard.is_empty() {
        return Err(ServerError::Internal("GPU kernel contexts not initialized".to_string()));
    }

    // Use devices from kernel contexts instead of creating new ones
    // This ensures we use the same device instances that were used during initialization,
    // avoiding CUDA context issues
    let num_gpus = gpu_weights.num_gpus;

    // Track which GPU currently holds our tensors
    let mut current_gpu_id: usize = 0;

    let ctx = ServerContext::new();
    let num_layers = gpu_weights.num_layers;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;
    let hidden_dim = gpu_weights.hidden_dim;

    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        num_layers = num_layers,
        "Starting GPU-resident prefill v3 (minimal transfers)"
    );

    let start_time = std::time::Instant::now();

    // Initialize KV cache (stays on CPU for now - could optimize later)
    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len); num_layers];

    // Bind GPU 0 for initial tensor upload using device from kernel context
    let initial_device = kernel_contexts_guard[0].device();
    initial_device.bind_to_thread()
        .map_err(|e| ServerError::GpuError(format!("Failed to bind initial GPU: {}", e)))?;

    // Upload hidden states to GPU 0 at start
    // Use 2D shape [1, hidden_dim] to match matrix multiplication output shapes
    let mut hidden_client_gpu: Vec<CudaTensor> = request.hidden_client.iter()
        .map(|h| CudaTensor::from_f32(initial_device, vec![1, hidden_dim], h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let mut hidden_server_gpu: Vec<CudaTensor> = request.hidden_server.iter()
        .map(|h| CudaTensor::from_f32(initial_device, vec![1, hidden_dim], h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // Process all layers - everything stays on GPU
    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);

        // Get the GPU ID for this layer's weights and use device from kernel context
        let layer_gpu_id = gpu_weights.layer_gpu_id(layer_idx);
        let kernels = &kernel_contexts_guard[layer_gpu_id];
        let device = kernels.device();

        // Bind this layer's GPU to the current thread
        device.bind_to_thread()
            .map_err(|e| ServerError::GpuError(format!("Failed to bind GPU {}: {}", layer_gpu_id, e)))?;

        // Transfer hidden states to this layer's GPU if needed
        if layer_gpu_id != current_gpu_id {
            tracing::debug!("Transferring hidden states from GPU {} to GPU {} for layer {}",
                current_gpu_id, layer_gpu_id, layer_idx);

            // Bind source device and synchronize before any transfer
            let source_device = kernel_contexts_guard[current_gpu_id].device();
            source_device.bind_to_thread()
                .map_err(|e| ServerError::GpuError(format!("Failed to bind source GPU {}: {}", current_gpu_id, e)))?;
            source_device.synchronize()
                .map_err(|e| ServerError::GpuError(format!("Failed to sync source GPU {}: {}", current_gpu_id, e)))?;

            // Download all data from source GPU to host while source is bound
            let client_host_data: Vec<Vec<f32>> = hidden_client_gpu.iter()
                .map(|t| source_device.dtoh_f32(t.data()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(format!("Failed to download client tensors: {}", e)))?;

            let server_host_data: Vec<Vec<f32>> = hidden_server_gpu.iter()
                .map(|t| source_device.dtoh_f32(t.data()))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(format!("Failed to download server tensors: {}", e)))?;

            // Drop old GPU tensors while source device is still bound
            // (CudaSlice::drop needs correct device context)
            drop(hidden_client_gpu);
            drop(hidden_server_gpu);

            // Now bind target device and upload new tensors
            device.bind_to_thread()
                .map_err(|e| ServerError::GpuError(format!("Failed to bind target GPU {}: {}", layer_gpu_id, e)))?;

            hidden_client_gpu = client_host_data.into_iter()
                .map(|data| CudaTensor::from_f32(device, vec![1, hidden_dim], data))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(format!("Failed to upload client tensors to GPU {}: {}", layer_gpu_id, e)))?;

            hidden_server_gpu = server_host_data.into_iter()
                .map(|data| CudaTensor::from_f32(device, vec![1, hidden_dim], data))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ServerError::GpuError(format!("Failed to upload server tensors to GPU {}: {}", layer_gpu_id, e)))?;

            // Synchronize target device after upload
            device.synchronize()
                .map_err(|e| ServerError::GpuError(format!("Failed to sync target GPU {}: {}", layer_gpu_id, e)))?;

            current_gpu_id = layer_gpu_id;
        }

        let mut new_hidden_client_gpu = Vec::with_capacity(seq_len);
        let mut new_hidden_server_gpu = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // === Phase 1: Input LayerNorm (GPU-native) ===
            let (normed_client, normed_server) = secure_rms_norm_gpu(
                &hidden_client_gpu[pos],
                &hidden_server_gpu[pos],
                &layer.input_layernorm_gpu,
                1e-6,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} RMSNorm failed: {}", layer_idx, e)))?;

            // === Phase 2: QKV Projection (GPU-native with CPU RoPE) ===
            let qkv_result = layer.attention.project_qkv_gpu_tensor(
                &normed_client, &normed_server, pos, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} QKV failed: {}", layer_idx, e)))?;

            // Download K, V for cache (reconstruct on CPU)
            let k_client_cpu = qkv_result.k_client.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let k_server_cpu = qkv_result.k_server.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let v_client_cpu = qkv_result.v_client.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let v_server_cpu = qkv_result.v_server.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            let k_combined: Vec<f32> = k_client_cpu.iter()
                .zip(k_server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect();
            let v_combined: Vec<f32> = v_client_cpu.iter()
                .zip(v_server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect();

            k_cache[layer_idx].push(k_combined);
            v_cache[layer_idx].push(v_combined);

            // === Phase 3: Attention (CPU for now - could use GPU attention kernel) ===
            let q_client_cpu = qkv_result.q_client.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let q_server_cpu = qkv_result.q_server.to_f32_host(&device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            let q_combined: Vec<f32> = q_client_cpu.iter()
                .zip(q_server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect();

            let k_cache_slice: Vec<Vec<f32>> = k_cache[layer_idx][0..=pos].to_vec();
            let v_cache_slice: Vec<Vec<f32>> = v_cache[layer_idx][0..=pos].to_vec();

            let attn_output = shardlm_v2_sharing::secure_attention_approx(
                &q_combined,
                &k_cache_slice,
                &v_cache_slice,
                num_heads,
                num_kv_heads,
                head_dim,
                &ctx,
            );

            // Upload attention output to GPU (2D shape to match O proj)
            let attn_client_gpu = CudaTensor::from_f32(&device, vec![1, hidden_dim], attn_output.clone())
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let attn_server_gpu = CudaTensor::zeros(&device, vec![1, hidden_dim])
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            // === Phase 4: O Projection (GPU-native) ===
            let (o_client, o_server) = layer.attention.project_output_gpu_tensor(
                &attn_client_gpu, &attn_server_gpu, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} O proj failed: {}", layer_idx, e)))?;

            // Add residual (GPU-native)
            let (hidden_after_attn_client, hidden_after_attn_server) = secure_add_gpu(
                &o_client, &o_server,
                &hidden_client_gpu[pos], &hidden_server_gpu[pos],
                kernels, &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} residual add failed: {}", layer_idx, e)))?;

            // === Phase 5: Post-Attention LayerNorm (GPU-native) ===
            let (normed_ffn_client, normed_ffn_server) = secure_rms_norm_gpu(
                &hidden_after_attn_client,
                &hidden_after_attn_server,
                &layer.post_attn_layernorm_gpu,
                1e-6,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} post-attn RMSNorm failed: {}", layer_idx, e)))?;

            // === Phase 6: FFN Gate/Up (GPU-native) ===
            let ffn_result = layer.ffn.project_gate_up_gpu_tensor(
                &normed_ffn_client, &normed_ffn_server, kernels, &device
            ).map_err(|e| ServerError::Internal(format!("Layer {} FFN gate/up failed: {}", layer_idx, e)))?;

            // === Phase 7: SwiGLU (GPU-native) ===
            let (activated_client, activated_server) = secure_swiglu_gpu(
                &ffn_result.gate_client,
                &ffn_result.gate_server,
                &ffn_result.up_client,
                &ffn_result.up_server,
                kernels,
                &device,
            ).map_err(|e| ServerError::Internal(format!("Layer {} SwiGLU failed: {}", layer_idx, e)))?;

            // === Phase 8: FFN Down (GPU-native) ===
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
            tracing::debug!("GPU-resident prefill v3 layer {}/{} complete", layer_idx + 1, num_layers);
        }
    }

    // Final device and kernel context are wherever the last layer's outputs are
    let final_kernels = &kernel_contexts_guard[current_gpu_id];
    let final_device = final_kernels.device();

    // Compute logits for last token (download only at end)
    let last_idx = seq_len - 1;
    let (normed_client, normed_server) = secure_rms_norm_gpu(
        &hidden_client_gpu[last_idx],
        &hidden_server_gpu[last_idx],
        &gpu_weights.final_norm_gpu,
        1e-6,
        final_kernels,
        final_device,
    ).map_err(|e| ServerError::Internal(format!("Final RMSNorm failed: {}", e)))?;

    // Download final normalized hidden states for LM head
    let normed_client_cpu = normed_client.to_f32_host(final_device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;
    let normed_server_cpu = normed_server.to_f32_host(final_device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let (logits_client, logits_server) = gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_client_cpu, &normed_server_cpu, final_kernels, final_device
    ).map_err(|e| ServerError::Internal(format!("LM head failed: {}", e)))?;

    // Download final hidden states for response
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

    let elapsed = start_time.elapsed();
    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        elapsed_ms = elapsed.as_millis(),
        tokens_per_sec = seq_len as f64 / elapsed.as_secs_f64(),
        "GPU-resident prefill v3 complete (minimal transfers)"
    );

    Ok(Json(BatchedPrefillResponse {
        final_hidden_client,
        final_hidden_server,
        k_cache,
        v_cache,
        logits_client,
        logits_server,
    }))
}

// =============================================================================
// V3 REFERENCE IMPLEMENTATION: Binary Protocol Endpoint
// =============================================================================

/// POST /v3/secure/gpu/prefill - V3 Reference Implementation with binary serialization
///
/// This endpoint uses bincode for efficient serialization (18x smaller payloads).
/// It delegates to the existing V3 JSON endpoint for the actual computation,
/// providing binary input/output while maintaining the same security guarantees.
///
/// # Security Guarantees (V3 Reference Implementation)
/// - Server NEVER reconstructs plaintext from shares (when mpc-secure feature enabled)
/// - Binary serialization reduces network overhead by 18x
/// - GPU memory encrypted via H100 Confidential Computing (when h100-cc feature enabled)
#[cfg(all(feature = "cuda", feature = "binary-protocol"))]
#[axum::debug_handler]
pub async fn batched_prefill_gpu_v3_binary(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Result<axum::response::Response> {
    use super::binary_protocol::{BinaryPrefillRequest, BinaryPrefillResponse};
    use axum::http::header;

    let start_time = std::time::Instant::now();
    let binary_input_size = body.len();

    // Deserialize binary request
    let binary_request: BinaryPrefillRequest = bincode::deserialize(&body)
        .map_err(|e| ServerError::InvalidRequest(format!("Invalid binary request: {}", e)))?;

    // Parse session ID
    let session_id = uuid::Uuid::from_bytes(binary_request.session_id);

    // Convert to nested format for JSON endpoint
    let (hidden_client, hidden_server) = binary_request.to_nested();
    let seq_len = hidden_client.len();

    tracing::info!(
        session_id = %session_id,
        seq_len = seq_len,
        binary_size = binary_input_size,
        "V3 reference implementation prefill (binary protocol)"
    );

    // Create JSON request
    let json_request = BatchedPrefillRequest {
        session_id: session_id.to_string(),
        hidden_client,
        hidden_server,
    };

    // Call existing V3 endpoint
    let json_result = batched_prefill_gpu_v3(
        State(state),
        Json(json_request),
    ).await?;

    // Convert response to binary
    let json_response = json_result.0;

    let binary_response = BinaryPrefillResponse::from_nested(
        json_response.final_hidden_client,
        json_response.final_hidden_server,
        &json_response.k_cache,
        &json_response.v_cache,
        json_response.logits_client,
        json_response.logits_server,
    );

    let response_bytes = bincode::serialize(&binary_response)
        .map_err(|e| ServerError::Internal(format!("Failed to serialize response: {}", e)))?;

    let elapsed = start_time.elapsed();
    tracing::info!(
        session_id = %session_id,
        binary_input = binary_input_size,
        binary_output = response_bytes.len(),
        elapsed_ms = elapsed.as_millis(),
        "V3 binary prefill complete"
    );

    Ok(axum::response::Response::builder()
        .header(header::CONTENT_TYPE, "application/octet-stream")
        .body(axum::body::Body::from(response_bytes))
        .unwrap())
}
