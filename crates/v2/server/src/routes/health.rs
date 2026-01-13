//! Health check endpoints

use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;

use crate::state::AppState;

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
}

/// Readiness response
#[derive(Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub model_loaded: bool,
    pub gpu_count: usize,
}

/// Server info response
#[derive(Serialize)]
pub struct InfoResponse {
    pub version: String,
    pub model_name: String,
    pub num_gpus: usize,
    pub uptime_secs: u64,
    pub active_sessions: usize,
    pub max_new_tokens: usize,
    pub default_temperature: f32,
}

/// GET /health - Basic health check
pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

/// GET /ready - Readiness check (model loaded)
pub async fn ready(State(state): State<AppState>) -> (StatusCode, Json<ReadyResponse>) {
    let is_ready = state.is_ready();
    let status = if is_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status,
        Json(ReadyResponse {
            ready: is_ready,
            model_loaded: is_ready,
            gpu_count: state.config.num_gpus,
        }),
    )
}

/// GET /v2/info - Server information
pub async fn info(State(state): State<AppState>) -> Json<InfoResponse> {
    Json(InfoResponse {
        version: state.info.version.clone(),
        model_name: state.info.model_name.clone(),
        num_gpus: state.info.num_gpus,
        uptime_secs: state.uptime().as_secs(),
        active_sessions: state.session_count(),
        max_new_tokens: state.config.max_new_tokens,
        default_temperature: state.config.default_temperature,
    })
}
