//! Error types for Confidential Computing module

use thiserror::Error;

/// Result type for CC operations
pub type Result<T> = std::result::Result<T, CcError>;

/// Confidential Computing error types
#[derive(Error, Debug)]
pub enum CcError {
    /// CC not available on this system
    #[error("Confidential Computing not available: {0}")]
    NotAvailable(String),

    /// Attestation failed
    #[error("Attestation failed: {0}")]
    AttestationFailed(String),

    /// Encryption error
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Decryption error
    #[error("Decryption error: {0}")]
    DecryptionError(String),

    /// Key exchange error
    #[error("Key exchange error: {0}")]
    KeyExchangeError(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    AllocationError(String),

    /// Verification error
    #[error("Verification error: {0}")]
    VerificationError(String),
}

impl From<cudarc::driver::DriverError> for CcError {
    fn from(err: cudarc::driver::DriverError) -> Self {
        CcError::CudaError(err.to_string())
    }
}

#[cfg(feature = "software-cc")]
impl From<aes_gcm::Error> for CcError {
    fn from(_err: aes_gcm::Error) -> Self {
        CcError::EncryptionError("AES-GCM error".to_string())
    }
}
