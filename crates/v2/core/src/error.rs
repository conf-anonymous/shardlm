//! v2 Error types

use thiserror::Error;

/// v2 Result type
pub type Result<T> = std::result::Result<T, V2Error>;

/// v2 Error types
#[derive(Debug, Error)]
pub enum V2Error {
    // Configuration errors
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    // GPU errors
    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("GPU error: {0}")]
    Gpu(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Insufficient VRAM: required {required_gb:.1}GB, available {available_gb:.1}GB")]
    InsufficientVram {
        required_gb: f32,
        available_gb: f32,
    },

    #[error("GPU {gpu_id} not available")]
    GpuNotAvailable { gpu_id: usize },

    #[error("Tensor parallel setup failed: {0}")]
    TensorParallelError(String),

    // Model errors
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("Weight shape mismatch: expected {expected:?}, got {got:?}")]
    WeightShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    // Session errors
    #[error("Session not found")]
    SessionNotFound,

    #[error("Session expired")]
    SessionExpired,

    #[error("Session limit reached")]
    SessionLimitReached,

    #[error("Request limit reached for session")]
    RequestLimitReached,

    // OT errors
    #[error("OT protocol error: {0}")]
    OtProtocolError(String),

    #[error("Base OT not complete")]
    BaseOtIncomplete,

    #[error("Counter mismatch: expected {expected}, got {got}")]
    CounterMismatch { expected: u64, got: u64 },

    // Inference errors
    #[error("Context length exceeded: max {max}, got {got}")]
    ContextLengthExceeded { max: usize, got: usize },

    #[error("Generation length exceeded: max {max}, got {got}")]
    GenerationLengthExceeded { max: usize, got: usize },

    #[error("Invalid token ID: {0}")]
    InvalidTokenId(u32),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    // Network errors
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Timeout")]
    Timeout,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    // Internal errors
    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl V2Error {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            V2Error::Timeout | V2Error::ConnectionError(_) | V2Error::RateLimitExceeded
        )
    }

    /// Get HTTP status code for this error
    pub fn status_code(&self) -> u16 {
        match self {
            V2Error::InvalidConfig(_) => 400,
            V2Error::SessionNotFound => 404,
            V2Error::SessionExpired => 410,
            V2Error::SessionLimitReached | V2Error::RequestLimitReached => 429,
            V2Error::RateLimitExceeded => 429,
            V2Error::ContextLengthExceeded { .. } | V2Error::GenerationLengthExceeded { .. } => 400,
            V2Error::Timeout => 504,
            _ => 500,
        }
    }
}
