//! Model loading error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Safetensors error: {0}")]
    Safetensors(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Missing tensor: {0}")]
    MissingTensor(String),

    #[error("Invalid tensor shape: expected {expected:?}, got {got:?}")]
    InvalidShape { expected: Vec<usize>, got: Vec<usize> },

    #[error("Invalid tensor dtype: expected {expected}, got {got}")]
    InvalidDtype { expected: String, got: String },

    #[error("Fixed-point error: {0}")]
    FixedPoint(#[from] shardlm_fixed_point::FixedPointError),

    #[error("Model config error: {0}")]
    Config(String),

    #[error("RoPE error: {0}")]
    Rope(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;
