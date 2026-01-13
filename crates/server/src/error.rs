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

/// Server error types
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Session expired: {0}")]
    SessionExpired(String),

    #[error("Session not ready: {0}")]
    SessionNotReady(String),

    #[error("Invalid counter: expected {expected}, got {got}")]
    InvalidCounter { expected: u64, got: u64 },

    #[error("Max requests exceeded")]
    MaxRequestsExceeded,

    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    #[error("Protocol error: {0}")]
    ProtocolError(String),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Error response body
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    code: String,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            ServerError::SessionNotFound(_) => (StatusCode::NOT_FOUND, "SESSION_NOT_FOUND"),
            ServerError::SessionExpired(_) => (StatusCode::GONE, "SESSION_EXPIRED"),
            ServerError::SessionNotReady(_) => (StatusCode::PRECONDITION_FAILED, "SESSION_NOT_READY"),
            ServerError::InvalidCounter { .. } => (StatusCode::CONFLICT, "INVALID_COUNTER"),
            ServerError::MaxRequestsExceeded => (StatusCode::TOO_MANY_REQUESTS, "MAX_REQUESTS_EXCEEDED"),
            ServerError::InvalidFrame(_) => (StatusCode::BAD_REQUEST, "INVALID_FRAME"),
            ServerError::ProtocolError(_) => (StatusCode::BAD_REQUEST, "PROTOCOL_ERROR"),
            ServerError::ModelNotLoaded => (StatusCode::SERVICE_UNAVAILABLE, "MODEL_NOT_LOADED"),
            ServerError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
        };

        let body = ErrorResponse {
            error: self.to_string(),
            code: code.to_string(),
        };

        (status, Json(body)).into_response()
    }
}

impl From<shardlm_ot::OtError> for ServerError {
    fn from(err: shardlm_ot::OtError) -> Self {
        ServerError::ProtocolError(err.to_string())
    }
}

impl From<shardlm_model::ModelError> for ServerError {
    fn from(err: shardlm_model::ModelError) -> Self {
        ServerError::Internal(err.to_string())
    }
}
