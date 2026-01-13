//! Server error types

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

/// Server result type
pub type Result<T> = std::result::Result<T, ServerError>;

/// Server errors
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Session expired: {0}")]
    SessionExpired(String),

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<shardlm_v2_model::ModelError> for ServerError {
    fn from(e: shardlm_v2_model::ModelError) -> Self {
        ServerError::InferenceError(e.to_string())
    }
}

impl From<shardlm_v2_core::V2Error> for ServerError {
    fn from(e: shardlm_v2_core::V2Error) -> Self {
        ServerError::GpuError(e.to_string())
    }
}

/// Error response body
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ServerError::ModelNotLoaded => (StatusCode::SERVICE_UNAVAILABLE, "model_not_loaded"),
            ServerError::SessionNotFound(_) => (StatusCode::NOT_FOUND, "session_not_found"),
            ServerError::SessionExpired(_) => (StatusCode::GONE, "session_expired"),
            ServerError::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            ServerError::InferenceError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "inference_error"),
            ServerError::GpuError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "gpu_error"),
            ServerError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
        };

        let body = Json(ErrorResponse {
            error: self.to_string(),
            code: code.to_string(),
        });

        (status, body).into_response()
    }
}
