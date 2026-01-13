//! Inference endpoints
//!
//! These endpoints handle secure inference computation:
//! - Forward pass through transformer layers
//! - Logit computation
//!
//! Protocol (simplified for now - runs transformer in plaintext):
//! 1. Client has hidden state H (from summing embeddings)
//! 2. Client creates additive shares: H = H_c + H_s
//! 3. Client sends BOTH shares to server
//! 4. Server reconstructs H, runs transformer, creates output shares
//! 5. Server sends both output shares back
//! 6. Client reconstructs: logits = logits_c + logits_s

use axum::{extract::State, Json};
use rand::RngCore;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shardlm_model::{compute_logits, TransformerState};
use uuid::Uuid;

use crate::error::{Result, ServerError};
use crate::state::AppState;

/// Forward pass request with secret-shared hidden state
#[derive(Debug, Deserialize)]
pub struct ForwardRequest {
    /// Session ID
    pub session_id: String,
    /// Client's share of hidden state: H_c (fixed-point i32 values)
    pub hidden_client: Vec<i32>,
    /// Server's share of hidden state: H_s (fixed-point i32 values)
    /// Note: In production, this would come from OT, not from client
    pub hidden_server: Vec<i32>,
    /// Position in sequence (for KV cache)
    #[serde(default)]
    pub position: usize,
    /// Whether to run full transformer (false = just lm_head)
    #[serde(default = "default_run_transformer")]
    pub run_transformer: bool,
}

fn default_run_transformer() -> bool {
    true
}

/// Forward pass response with secret-shared logits
#[derive(Debug, Serialize)]
pub struct ForwardResponse {
    /// Client's share of logits: L_c
    pub logits_client: Vec<i32>,
    /// Server's share of logits: L_s
    pub logits_server: Vec<i32>,
    /// Top-k token predictions (computed from reconstructed logits for convenience)
    pub top_tokens: Vec<TokenPrediction>,
}

/// Token prediction
#[derive(Debug, Serialize)]
pub struct TokenPrediction {
    /// Token ID
    pub token_id: u32,
    /// Logit value (fixed-point, reconstructed)
    pub logit: i32,
}

/// POST /v1/inference/forward - Run secure forward pass
///
/// Takes client's and server's hidden state shares, runs transformer computation,
/// returns both output shares for client reconstruction.
pub async fn forward(
    State(state): State<AppState>,
    Json(request): Json<ForwardRequest>,
) -> Result<Json<ForwardResponse>> {
    // Check server is ready
    if !state.is_ready() {
        return Err(ServerError::ModelNotLoaded);
    }

    // Get model weights
    let weights = state.get_weights()?;

    let hidden_size = weights.config.hidden_size;

    // Validate input dimensions
    if request.hidden_client.len() != hidden_size {
        return Err(ServerError::InvalidFrame(format!(
            "Hidden client size mismatch: expected {}, got {}",
            hidden_size,
            request.hidden_client.len()
        )));
    }
    if request.hidden_server.len() != hidden_size {
        return Err(ServerError::InvalidFrame(format!(
            "Hidden server size mismatch: expected {}, got {}",
            hidden_size,
            request.hidden_server.len()
        )));
    }

    let scale = weights.scale;

    // Reconstruct the hidden state from shares
    let hidden_state: Vec<i32> = request
        .hidden_client
        .iter()
        .zip(&request.hidden_server)
        .map(|(&hc, &hs)| hc.wrapping_add(hs))
        .collect();

    // Parse session ID
    let session_id = Uuid::parse_str(&request.session_id).map_err(|_| {
        ServerError::InvalidFrame(format!("Invalid session ID: {}", request.session_id))
    })?;

    // Run transformer computation (plaintext for now)
    let final_hidden = if request.run_transformer {
        // Create transformer state
        let transformer = TransformerState::new(weights.config.clone(), scale);

        // Get session and use its persistent KV cache
        let session = state.sessions.get(&session_id)?;

        // Use session's KV cache (created lazily if needed)
        session.with_kv_cache(
            weights.num_layers(),
            weights.config.max_position_embeddings,
            weights.config.num_key_value_heads,
            weights.config.head_dim(),
            |kv_cache| {
                // Run forward pass for single token
                transformer.forward_single(weights, &hidden_state, request.position, kv_cache)
            },
        )
    } else {
        // Skip transformer, use input directly
        hidden_state.clone()
    };

    // Compute logits from final hidden state
    let logits = compute_logits(&final_hidden, &weights.lm_head, scale);

    // Create output shares using fast ChaCha20 PRNG
    // Seed from session ID + position for deterministic but cryptographically secure randomness
    let seed = session_id.as_u128() as u64 ^ (request.position as u64);
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let logits_server: Vec<i32> = (0..logits.len())
        .map(|_| rng.next_u32() as i32)
        .collect();

    let logits_client: Vec<i32> = logits
        .iter()
        .zip(&logits_server)
        .map(|(&l, &ls)| l.wrapping_sub(ls))
        .collect();

    // Find top-k predictions from logits
    let mut indexed: Vec<(usize, i32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

    let top_tokens: Vec<TokenPrediction> = indexed
        .iter()
        .take(10)
        .map(|&(token_id, logit)| TokenPrediction {
            token_id: token_id as u32,
            logit,
        })
        .collect();

    tracing::debug!(
        session_id = %request.session_id,
        position = request.position,
        run_transformer = request.run_transformer,
        top_token = top_tokens.first().map(|t| t.token_id).unwrap_or(0),
        "Forward pass completed"
    );

    Ok(Json(ForwardResponse {
        logits_client,
        logits_server,
        top_tokens,
    }))
}

/// Compute matrix-vector multiply: output = input @ weight
/// where weight is [in_features x out_features] in row-major order
#[allow(dead_code)]
fn compute_matmul(
    input: &[i32],
    weight: &[i32],
    in_features: usize,
    out_features: usize,
    scale: u8,
) -> Vec<i32> {
    let mut output = vec![0i64; out_features];

    for i in 0..in_features {
        let xi = input[i] as i64;
        for j in 0..out_features {
            let wij = weight[i * out_features + j] as i64;
            output[j] += xi * wij;
        }
    }

    // Scale down and convert to i32
    output.iter().map(|&x| (x >> scale) as i32).collect()
}
