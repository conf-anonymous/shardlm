//! Secret sharing error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SharingError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Scale mismatch: expected {expected}, got {got}")]
    ScaleMismatch { expected: u8, got: u8 },

    #[error("Fixed-point error: {0}")]
    FixedPoint(#[from] shardlm_fixed_point::FixedPointError),

    #[error("Operation not ready: prerequisite step not completed")]
    NotReady,
}

pub type Result<T> = std::result::Result<T, SharingError>;
