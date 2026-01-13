//! Sharing error types

use thiserror::Error;

pub type Result<T> = std::result::Result<T, SharingError>;

#[derive(Debug, Error)]
pub enum SharingError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Device mismatch: shares must be on same device")]
    DeviceMismatch,

    #[error("Invalid share: {0}")]
    InvalidShare(String),

    #[error("Reconstruction failed: {0}")]
    ReconstructionFailed(String),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),
}
