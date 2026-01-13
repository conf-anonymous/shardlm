//! Protocol error types

use thiserror::Error;

use crate::ErrorCode;

#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Invalid magic bytes")]
    InvalidMagic,

    #[error("Version mismatch: expected {expected}, got {got}")]
    VersionMismatch { expected: u16, got: u16 },

    #[error("Unknown message type: 0x{0:04X}")]
    UnknownMessageType(u16),

    #[error("Invalid header length: expected {expected}, got {got}")]
    InvalidHeaderLength { expected: u16, got: u16 },

    #[error("Header CRC mismatch: expected 0x{expected:08X}, got 0x{got:08X}")]
    HeaderCrcMismatch { expected: u32, got: u32 },

    #[error("Payload CRC mismatch: expected 0x{expected:08X}, got 0x{got:08X}")]
    PayloadCrcMismatch { expected: u32, got: u32 },

    #[error("Buffer too short: need {need} bytes, have {have}")]
    BufferTooShort { need: usize, have: usize },

    #[error("Invalid payload length for message type")]
    InvalidPayloadLength,

    #[error("Prompt length exceeds maximum: {len} > {max}")]
    PromptTooLong { len: u16, max: u16 },

    #[error("Invalid counter: expected {expected}, got {got}")]
    InvalidCounter { expected: u64, got: u64 },

    #[error("Session not found")]
    SessionNotFound,

    #[error("Session expired")]
    SessionExpired,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Protocol error: {code:?} - {message}")]
    ProtocolErrorResponse { code: ErrorCode, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;
