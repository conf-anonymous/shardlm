//! Fixed-point error types

use thiserror::Error;

#[derive(Error, Debug)]
pub enum FixedPointError {
    #[error("Overflow during fixed-point conversion: value {value} exceeds i32 range")]
    Overflow { value: f64 },

    #[error("Underflow during fixed-point conversion: value {value} too small")]
    Underflow { value: f64 },

    #[error("Scale mismatch: expected {expected}, got {got}")]
    ScaleMismatch { expected: u8, got: u8 },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid scale: {0} (must be 0-30)")]
    InvalidScale(u8),
}

pub type Result<T> = std::result::Result<T, FixedPointError>;
