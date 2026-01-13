//! Protocol constants for v2

/// Protocol magic bytes for v2: "SLM2"
pub const MAGIC_V2: [u8; 4] = [0x53, 0x4C, 0x4D, 0x32];

/// Protocol version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ProtocolVersion {
    /// v1: Original protocol (Qwen 2.5 1.5B, CPU)
    V1 = 1,
    /// v2: Production protocol (Llama 70B, CUDA)
    V2 = 2,
}

/// Message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum MsgType {
    // Session management (0x00xx)
    /// Version negotiation request
    VersionRequest = 0x0001,
    /// Version negotiation response
    VersionResponse = 0x0002,
    /// Session init request
    SessionInit = 0x0010,
    /// Session parameters (server → client)
    SessionParams = 0x0011,
    /// Session ready acknowledgment
    SessionReady = 0x0012,
    /// Session refresh request
    SessionRefresh = 0x0013,
    /// Session close
    SessionClose = 0x001F,

    // Base OT (0x01xx)
    /// Base OT message 1 (receiver → sender)
    BaseOtMsg1 = 0x0101,
    /// Base OT message 2 (sender → receiver)
    BaseOtMsg2 = 0x0102,
    /// Base OT message 3 (receiver → sender)
    BaseOtMsg3 = 0x0103,
    /// Base OT complete acknowledgment
    BaseOtComplete = 0x010F,

    // OT Extension (0x02xx)
    /// OT extension setup
    OtExtSetup = 0x0201,
    /// OT extension response
    OtExtResponse = 0x0202,

    // Inference requests (0x10xx)
    /// Query request (client → server)
    QueryRequest = 0x1001,
    /// Query response (server → client)
    QueryResponse = 0x1002,
    /// Streaming token (server → client)
    StreamToken = 0x1003,
    /// Generation complete
    GenerationComplete = 0x100F,

    // KV Cache (0x11xx)
    /// KV cache update
    KvCacheUpdate = 0x1101,
    /// KV cache clear
    KvCacheClear = 0x1102,

    // Error (0xFFxx)
    /// Error response
    Error = 0xFF00,
}

impl MsgType {
    /// Parse from u16
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x0001 => Some(Self::VersionRequest),
            0x0002 => Some(Self::VersionResponse),
            0x0010 => Some(Self::SessionInit),
            0x0011 => Some(Self::SessionParams),
            0x0012 => Some(Self::SessionReady),
            0x0013 => Some(Self::SessionRefresh),
            0x001F => Some(Self::SessionClose),
            0x0101 => Some(Self::BaseOtMsg1),
            0x0102 => Some(Self::BaseOtMsg2),
            0x0103 => Some(Self::BaseOtMsg3),
            0x010F => Some(Self::BaseOtComplete),
            0x0201 => Some(Self::OtExtSetup),
            0x0202 => Some(Self::OtExtResponse),
            0x1001 => Some(Self::QueryRequest),
            0x1002 => Some(Self::QueryResponse),
            0x1003 => Some(Self::StreamToken),
            0x100F => Some(Self::GenerationComplete),
            0x1101 => Some(Self::KvCacheUpdate),
            0x1102 => Some(Self::KvCacheClear),
            0xFF00 => Some(Self::Error),
            _ => None,
        }
    }

    /// Check if this message type expects a response
    pub fn expects_response(&self) -> bool {
        matches!(
            self,
            Self::VersionRequest
                | Self::SessionInit
                | Self::SessionRefresh
                | Self::BaseOtMsg1
                | Self::BaseOtMsg3
                | Self::OtExtSetup
                | Self::QueryRequest
        )
    }
}

/// Message flags
pub mod flags {
    /// Compressed payload (zstd)
    pub const COMPRESSED: u16 = 0x0001;
    /// Encrypted payload (AES-GCM)
    pub const ENCRYPTED: u16 = 0x0002;
    /// Requires acknowledgment
    pub const ACK_REQUIRED: u16 = 0x0004;
    /// Part of a multi-part message
    pub const MULTIPART: u16 = 0x0008;
    /// Last part of multi-part message
    pub const MULTIPART_END: u16 = 0x0010;
}

/// Maximum payload size (1GB for large context)
pub const MAX_PAYLOAD_SIZE: usize = 1024 * 1024 * 1024;

/// Header size in bytes
pub const HEADER_SIZE: usize = 16;
