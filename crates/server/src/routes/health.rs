//! Health, readiness, and info endpoints

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Serialize;

use crate::state::AppState;

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// Readiness check response
#[derive(Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub model_loaded: bool,
    pub sessions_active: usize,
}

/// Server info response
#[derive(Serialize)]
pub struct InfoResponse {
    pub version: String,
    pub git_commit: Option<String>,
    pub model: String,
    pub uptime_secs: u64,
    pub protocol_version: u8,
    pub embedding_dim: usize,
    pub vocab_size: usize,
    pub max_prompt_len: u16,
    pub ot_kappa: u8,
}

/// GET /health - Liveness check
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse { status: "ok" })
}

/// GET /ready - Readiness check
pub async fn ready(State(state): State<AppState>) -> impl IntoResponse {
    let is_ready = state.is_ready();
    let model_loaded = state.embeddings.is_some();
    let sessions_active = state.sessions.len();

    let response = ReadyResponse {
        ready: is_ready,
        model_loaded,
        sessions_active,
    };

    if is_ready {
        (StatusCode::OK, Json(response))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(response))
    }
}

/// GET /v1/info - Server information
pub async fn info(State(state): State<AppState>) -> impl IntoResponse {
    let (embedding_dim, vocab_size) = state
        .embeddings
        .as_ref()
        .as_ref()
        .map(|e| (e.embed_dim, e.vocab_size))
        .unwrap_or((0, 0));

    Json(InfoResponse {
        version: state.info.version.clone(),
        git_commit: state.info.git_commit.clone(),
        model: state.info.model_name.clone(),
        uptime_secs: state.uptime().as_secs(),
        protocol_version: 1,
        embedding_dim,
        vocab_size,
        max_prompt_len: state.config.max_prompt_len,
        ot_kappa: state.config.ot_kappa,
    })
}
