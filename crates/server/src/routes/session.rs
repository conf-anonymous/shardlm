//! Session management endpoints

use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Result, ServerError};
use crate::session::SessionStatus;
use crate::state::AppState;

/// Session creation response
#[derive(Serialize)]
pub struct SessionResponse {
    pub session_id: Uuid,
    pub protocol_version: u8,
    pub max_prompt_len: u16,
    pub ot_kappa: u8,
    pub cipher: &'static str,
    pub model: String,
    pub embedding_dim: usize,
    pub vocab_size: usize,
    /// TTL in seconds for this session
    pub ttl_secs: i64,
}

/// Session ID request
#[derive(Deserialize)]
pub struct SessionIdRequest {
    pub session_id: String,
}

/// Session refresh response
#[derive(Serialize)]
pub struct RefreshResponse {
    pub session_id: Uuid,
    /// New TTL in seconds after refresh
    pub ttl_secs: i64,
}

/// POST /v1/session/new - Create a new session
pub async fn create_session(State(state): State<AppState>) -> Result<Json<SessionResponse>> {
    // Create session
    let session_id = state.sessions.create_session()?;

    // Get model info
    let (embedding_dim, vocab_size) = state
        .embeddings
        .as_ref()
        .as_ref()
        .map(|e| (e.embed_dim, e.vocab_size))
        .unwrap_or((2048, 32000));

    // Get TTL from session store
    let ttl_secs = state.sessions.default_ttl().as_secs() as i64;

    tracing::info!(session_id = %session_id, ttl_secs, "Session created");

    Ok(Json(SessionResponse {
        session_id,
        protocol_version: 1,
        max_prompt_len: state.config.max_prompt_len,
        ot_kappa: state.config.ot_kappa,
        cipher: "AES-CTR",
        model: state.info.model_name.clone(),
        embedding_dim,
        vocab_size,
        ttl_secs,
    }))
}

/// POST /v1/session/status - Get session status (without extending TTL)
pub async fn session_status(
    State(state): State<AppState>,
    Json(request): Json<SessionIdRequest>,
) -> Result<Json<SessionStatus>> {
    let session_id = Uuid::parse_str(&request.session_id).map_err(|_| {
        ServerError::InvalidFrame(format!("Invalid session ID: {}", request.session_id))
    })?;

    let status = state.sessions.status(&session_id)?;

    tracing::debug!(session_id = %session_id, ttl_secs = status.ttl_secs, "Session status checked");

    Ok(Json(status))
}

/// POST /v1/session/refresh - Extend session TTL
pub async fn session_refresh(
    State(state): State<AppState>,
    Json(request): Json<SessionIdRequest>,
) -> Result<Json<RefreshResponse>> {
    let session_id = Uuid::parse_str(&request.session_id).map_err(|_| {
        ServerError::InvalidFrame(format!("Invalid session ID: {}", request.session_id))
    })?;

    let ttl_secs = state.sessions.refresh(&session_id)?;

    tracing::info!(session_id = %session_id, ttl_secs, "Session refreshed");

    Ok(Json(RefreshResponse {
        session_id,
        ttl_secs,
    }))
}
