//! Model error types

use shardlm_v2_core::V2Error;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Core error: {0}")]
    Core(#[from] V2Error),

    #[error("Model not found at path: {0}")]
    ModelNotFound(String),

    #[error("Invalid model format: {0}")]
    InvalidFormat(String),

    #[error("Missing weight: {0}")]
    MissingWeight(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Weight shape mismatch for {name}: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Unsupported data type: {0}")]
    UnsupportedDtype(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}
