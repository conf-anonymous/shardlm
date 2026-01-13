//! Session management endpoints

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;
use crate::state::AppState;

/// Create session request
#[derive(Debug, Deserialize)]
pub struct CreateSessionRequest {
    /// Optional client identifier
    #[serde(default)]
    pub client_id: Option<String>,
}

/// Create session response
#[derive(Debug, Serialize)]
pub struct CreateSessionResponse {
    pub session_id: String,
    pub model_name: String,
    pub max_new_tokens: usize,
}

/// Session status request
#[derive(Debug, Deserialize)]
pub struct SessionStatusRequest {
    pub session_id: String,
}

/// Session status response
#[derive(Debug, Serialize)]
pub struct SessionStatusResponse {
    pub session_id: String,
    pub active: bool,
    pub request_count: u64,
    pub age_secs: u64,
}

/// POST /v2/session/new - Create a new session
pub async fn create_session(
    State(state): State<AppState>,
    Json(_request): Json<CreateSessionRequest>,
) -> Result<Json<CreateSessionResponse>> {
    let session_id = state.create_session();

    tracing::info!(session_id = %session_id, "Created new session");

    Ok(Json(CreateSessionResponse {
        session_id: session_id.to_string(),
        model_name: state.info.model_name.clone(),
        max_new_tokens: state.config.max_new_tokens,
    }))
}

/// POST /v2/session/status - Get session status
pub async fn session_status(
    State(state): State<AppState>,
    Json(request): Json<SessionStatusRequest>,
) -> Result<Json<SessionStatusResponse>> {
    let session_id = Uuid::parse_str(&request.session_id)
        .map_err(|_| crate::error::ServerError::InvalidRequest("Invalid session ID".to_string()))?;

    let session = state.get_session(&session_id)?;
    let age_secs = session.created_at.elapsed().as_secs();
    let request_count = session
        .request_count
        .load(std::sync::atomic::Ordering::Relaxed);
    let active = !session.is_expired(state.config.session_ttl_secs);

    Ok(Json(SessionStatusResponse {
        session_id: request.session_id,
        active,
        request_count,
        age_secs,
    }))
}
