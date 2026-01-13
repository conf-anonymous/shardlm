//! Protocol constants per WIRE_FORMAT_SPEC.md

/// Protocol magic bytes: "SLMT"
pub const MAGIC: [u8; 4] = [0x53, 0x4C, 0x4D, 0x54];

/// Protocol version (v1)
pub const PROTOCOL_VERSION: u16 = 0x0001;

/// Maximum prompt length
pub const L_MAX: u16 = 64;

/// Fixed header length in bytes (v1)
pub const HEADER_LEN: u16 = 48;

/// Session ID length in bytes
pub const SESSION_ID_LEN: usize = 16;

/// Nonce length in bytes
pub const NONCE_LEN: usize = 8;

/// Message type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MsgType {
    /// Session initialization request (Client → Server)
    OtSessionInit = 0x0001,
    /// Session parameters response (Server → Client)
    OtSessionParams = 0x0002,
    /// Base OT handshake message (Bidirectional)
    OtBaseOtMsg = 0x0003,
    /// Session ready confirmation (Server → Client)
    OtSessionReady = 0x0004,
    /// Embedding fetch request (Client → Server)
    EmbedFetchRequest = 0x0010,
    /// Embedding fetch response (Server → Client)
    EmbedFetchResponse = 0x0011,
    /// Error message
    Error = 0x00F0,
}

impl MsgType {
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(MsgType::OtSessionInit),
            0x0002 => Some(MsgType::OtSessionParams),
            0x0003 => Some(MsgType::OtBaseOtMsg),
            0x0004 => Some(MsgType::OtSessionReady),
            0x0010 => Some(MsgType::EmbedFetchRequest),
            0x0011 => Some(MsgType::EmbedFetchResponse),
            0x00F0 => Some(MsgType::Error),
            _ => None,
        }
    }

    pub fn to_u16(self) -> u16 {
        self as u16
    }
}

/// OT suite identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum OtSuiteId {
    /// IKNP-style OT extension + 1-out-of-N selection layer
    IknpExtension = 0x0101,
    /// KOS-style OT extension + 1-out-of-N selection layer
    KosExtension = 0x0102,
}

impl OtSuiteId {
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0101 => Some(OtSuiteId::IknpExtension),
            0x0102 => Some(OtSuiteId::KosExtension),
            _ => None,
        }
    }

    pub fn to_u16(self) -> u16 {
        self as u16
    }
}

/// Value type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueType {
    /// i32 fixed-point
    I32FixedPoint = 1,
}

/// Row layout identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RowLayout {
    /// Contiguous i32[d]
    ContiguousI32 = 1,
}

/// Query format identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum QueryFormat {
    /// Batched OT v1
    BatchedOtV1 = 1,
}

/// Error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ErrorCode {
    BadVersion = 0x00000001,
    BadCrc = 0x00000002,
    UnknownMsgType = 0x00000003,
    SessionNotFound = 0x00000004,
    SessionExpired = 0x00000005,
    BadCtr = 0x00000006,
    InvalidParam = 0x00000007,
    OtHandshakeFailed = 0x00000008,
    QueryDecodeFailed = 0x00000009,
    Internal = 0x0000000A,
}

impl ErrorCode {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x00000001 => Some(ErrorCode::BadVersion),
            0x00000002 => Some(ErrorCode::BadCrc),
            0x00000003 => Some(ErrorCode::UnknownMsgType),
            0x00000004 => Some(ErrorCode::SessionNotFound),
            0x00000005 => Some(ErrorCode::SessionExpired),
            0x00000006 => Some(ErrorCode::BadCtr),
            0x00000007 => Some(ErrorCode::InvalidParam),
            0x00000008 => Some(ErrorCode::OtHandshakeFailed),
            0x00000009 => Some(ErrorCode::QueryDecodeFailed),
            0x0000000A => Some(ErrorCode::Internal),
            _ => None,
        }
    }

    pub fn to_u32(self) -> u32 {
        self as u32
    }
}
