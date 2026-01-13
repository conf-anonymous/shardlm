//! Protocol error types

use thiserror::Error;

pub type Result<T> = std::result::Result<T, ProtocolError>;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("Invalid magic bytes")]
    InvalidMagic,

    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u8),

    #[error("Unknown message type: 0x{0:04x}")]
    UnknownMessageType(u16),

    #[error("Invalid payload length: expected {expected}, got {got}")]
    InvalidPayloadLength { expected: usize, got: usize },

    #[error("Payload too large: max {max}, got {got}")]
    PayloadTooLarge { max: usize, got: usize },

    #[error("CRC mismatch: expected 0x{expected:08x}, got 0x{got:08x}")]
    CrcMismatch { expected: u32, got: u32 },

    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),

    #[error("Decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
