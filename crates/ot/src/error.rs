//! OT error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum OtError {
    #[error("Session not initialized")]
    SessionNotInitialized,

    #[error("Session already initialized")]
    SessionAlreadyInitialized,

    #[error("Session expired")]
    SessionExpired,

    #[error("Invalid session ID")]
    InvalidSessionId,

    #[error("Counter mismatch: expected {expected}, got {got}")]
    CounterMismatch { expected: u64, got: u64 },

    #[error("Invalid index: {index} >= {max}")]
    InvalidIndex { index: u32, max: u32 },

    #[error("Batch size exceeded: {size} > {max}")]
    BatchSizeExceeded { size: usize, max: usize },

    #[error("Invalid message format")]
    InvalidMessageFormat,

    #[error("Cryptographic error: {0}")]
    CryptoError(String),

    #[error("Protocol error: {0}")]
    ProtocolError(#[from] shardlm_protocol::ProtocolError),
}

pub type Result<T> = std::result::Result<T, OtError>;
