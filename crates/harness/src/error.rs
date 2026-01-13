//! Harness error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum HarnessError {
    #[error("Model error: {0}")]
    Model(#[from] shardlm_model::ModelError),

    #[error("OT error: {0}")]
    Ot(#[from] shardlm_ot::OtError),

    #[error("Sharing error: {0}")]
    Sharing(#[from] shardlm_sharing::SharingError),

    #[error("Fixed-point error: {0}")]
    FixedPoint(#[from] shardlm_fixed_point::FixedPointError),

    #[error("Protocol error: {0}")]
    Protocol(#[from] shardlm_protocol::ProtocolError),

    #[error("Verification failed: {message}")]
    VerificationFailed { message: String },

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, HarnessError>;
