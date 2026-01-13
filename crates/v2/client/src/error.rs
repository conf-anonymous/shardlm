//! Error types for the ShardLM V2 client

use thiserror::Error;

/// Client errors
#[derive(Error, Debug)]
pub enum ClientError {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Server returned an error
    #[error("Server error ({status}): {message}")]
    Server { status: u16, message: String },

    /// Invalid server URL
    #[error("Invalid server URL: {0}")]
    InvalidUrl(String),

    /// Session not initialized
    #[error("Session not initialized. Call create_session() first.")]
    NoSession,

    /// Invalid response from server
    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// WebSocket error
    #[error("WebSocket error: {0}")]
    WebSocket(String),

    /// Timeout
    #[error("Request timed out after {0}ms")]
    Timeout(u64),

    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Tokenizer error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
}

/// Result type for client operations
pub type Result<T> = std::result::Result<T, ClientError>;
