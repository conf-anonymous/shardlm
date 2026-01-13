//! Protocol messages for v2

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

use crate::constants::MsgType;
use crate::error::{ProtocolError, Result};
use crate::header::Header;

/// Session parameters for v2 (Server → Client)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionParamsV2 {
    /// Protocol version (2)
    pub protocol_version: u8,
    /// Maximum prompt length (tokens)
    pub max_prompt_len: u32,
    /// Maximum generation length (tokens)
    pub max_generation_len: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: u32,
    /// Vocabulary size
    pub vocab_size: u32,
    /// Number of layers
    pub num_layers: u32,
    /// Value type (1 = FP16, 2 = BF16)
    pub value_type: u8,
    /// Selected OT suite ID
    pub ot_suite: u16,
    /// Session TTL in seconds
    pub session_ttl_sec: u32,
    /// Maximum requests per session
    pub max_requests: u32,
    /// Server capabilities bitmap
    pub capabilities: u32,
}

impl Default for SessionParamsV2 {
    fn default() -> Self {
        Self {
            protocol_version: 2,
            max_prompt_len: 65536,     // 64K input tokens
            max_generation_len: 65536, // 64K output tokens
            hidden_dim: 8192,          // Llama 70B
            num_heads: 64,
            num_kv_heads: 8,
            vocab_size: 128256, // Llama 3.x
            num_layers: 80,
            value_type: 2, // BF16
            ot_suite: 0x0201, // GPU-accelerated IKNP
            session_ttl_sec: 3600,
            max_requests: 10000,
            capabilities: 0,
        }
    }
}

impl SessionParamsV2 {
    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);

        buf.write_u8(self.protocol_version).unwrap();
        buf.write_u8(0).unwrap(); // padding
        buf.write_u16::<LittleEndian>(0).unwrap(); // padding
        buf.write_u32::<LittleEndian>(self.max_prompt_len).unwrap();
        buf.write_u32::<LittleEndian>(self.max_generation_len).unwrap();
        buf.write_u32::<LittleEndian>(self.hidden_dim).unwrap();
        buf.write_u32::<LittleEndian>(self.num_heads).unwrap();
        buf.write_u32::<LittleEndian>(self.num_kv_heads).unwrap();
        buf.write_u32::<LittleEndian>(self.vocab_size).unwrap();
        buf.write_u32::<LittleEndian>(self.num_layers).unwrap();
        buf.write_u8(self.value_type).unwrap();
        buf.write_u8(0).unwrap(); // padding
        buf.write_u16::<LittleEndian>(self.ot_suite).unwrap();
        buf.write_u32::<LittleEndian>(self.session_ttl_sec).unwrap();
        buf.write_u32::<LittleEndian>(self.max_requests).unwrap();
        buf.write_u32::<LittleEndian>(self.capabilities).unwrap();

        buf
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 48 {
            return Err(ProtocolError::InvalidPayloadLength {
                expected: 48,
                got: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);

        let protocol_version = cursor.read_u8()?;
        let _ = cursor.read_u8()?; // padding
        let _ = cursor.read_u16::<LittleEndian>()?; // padding
        let max_prompt_len = cursor.read_u32::<LittleEndian>()?;
        let max_generation_len = cursor.read_u32::<LittleEndian>()?;
        let hidden_dim = cursor.read_u32::<LittleEndian>()?;
        let num_heads = cursor.read_u32::<LittleEndian>()?;
        let num_kv_heads = cursor.read_u32::<LittleEndian>()?;
        let vocab_size = cursor.read_u32::<LittleEndian>()?;
        let num_layers = cursor.read_u32::<LittleEndian>()?;
        let value_type = cursor.read_u8()?;
        let _ = cursor.read_u8()?; // padding
        let ot_suite = cursor.read_u16::<LittleEndian>()?;
        let session_ttl_sec = cursor.read_u32::<LittleEndian>()?;
        let max_requests = cursor.read_u32::<LittleEndian>()?;
        let capabilities = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            protocol_version,
            max_prompt_len,
            max_generation_len,
            hidden_dim,
            num_heads,
            num_kv_heads,
            vocab_size,
            num_layers,
            value_type,
            ot_suite,
            session_ttl_sec,
            max_requests,
            capabilities,
        })
    }
}

/// Query request (Client → Server)
#[derive(Debug, Clone)]
pub struct QueryRequestV2 {
    /// Session ID (16 bytes)
    pub session_id: [u8; 16],
    /// Request counter
    pub counter: u64,
    /// Number of input tokens
    pub num_tokens: u32,
    /// Maximum tokens to generate
    pub max_new_tokens: u32,
    /// Temperature (fixed-point, scale 1000)
    pub temperature: u16,
    /// Top-p (fixed-point, scale 1000)
    pub top_p: u16,
    /// Client's share of embeddings (follows header)
    pub embedding_share: Vec<u8>,
}

impl QueryRequestV2 {
    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(40 + self.embedding_share.len());

        buf.extend_from_slice(&self.session_id);
        buf.write_u64::<LittleEndian>(self.counter).unwrap();
        buf.write_u32::<LittleEndian>(self.num_tokens).unwrap();
        buf.write_u32::<LittleEndian>(self.max_new_tokens).unwrap();
        buf.write_u16::<LittleEndian>(self.temperature).unwrap();
        buf.write_u16::<LittleEndian>(self.top_p).unwrap();
        buf.extend_from_slice(&self.embedding_share);

        buf
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 40 {
            return Err(ProtocolError::InvalidPayloadLength {
                expected: 40,
                got: data.len(),
            });
        }

        let mut session_id = [0u8; 16];
        session_id.copy_from_slice(&data[0..16]);

        let mut cursor = Cursor::new(&data[16..]);

        let counter = cursor.read_u64::<LittleEndian>()?;
        let num_tokens = cursor.read_u32::<LittleEndian>()?;
        let max_new_tokens = cursor.read_u32::<LittleEndian>()?;
        let temperature = cursor.read_u16::<LittleEndian>()?;
        let top_p = cursor.read_u16::<LittleEndian>()?;

        let embedding_share = data[40..].to_vec();

        Ok(Self {
            session_id,
            counter,
            num_tokens,
            max_new_tokens,
            temperature,
            top_p,
            embedding_share,
        })
    }
}

/// Streaming token response (Server → Client)
#[derive(Debug, Clone)]
pub struct StreamTokenV2 {
    /// Token ID
    pub token_id: u32,
    /// Token position in sequence
    pub position: u32,
    /// Is this the last token?
    pub is_final: bool,
    /// Generation latency in microseconds
    pub latency_us: u32,
}

impl StreamTokenV2 {
    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);

        buf.write_u32::<LittleEndian>(self.token_id).unwrap();
        buf.write_u32::<LittleEndian>(self.position).unwrap();
        buf.write_u8(if self.is_final { 1 } else { 0 }).unwrap();
        buf.write_u8(0).unwrap(); // padding
        buf.write_u16::<LittleEndian>(0).unwrap(); // padding
        buf.write_u32::<LittleEndian>(self.latency_us).unwrap();

        buf
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(ProtocolError::InvalidPayloadLength {
                expected: 16,
                got: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);

        let token_id = cursor.read_u32::<LittleEndian>()?;
        let position = cursor.read_u32::<LittleEndian>()?;
        let is_final = cursor.read_u8()? != 0;
        let _ = cursor.read_u8()?;
        let _ = cursor.read_u16::<LittleEndian>()?;
        let latency_us = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            token_id,
            position,
            is_final,
            latency_us,
        })
    }
}

/// Error response
#[derive(Debug, Clone)]
pub struct ErrorResponse {
    /// Error code
    pub code: u32,
    /// Error message
    pub message: String,
}

impl ErrorResponse {
    /// Create a new error response
    pub fn new(code: u32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Encode to bytes
    pub fn encode(&self) -> Vec<u8> {
        let msg_bytes = self.message.as_bytes();
        let mut buf = Vec::with_capacity(8 + msg_bytes.len());

        buf.write_u32::<LittleEndian>(self.code).unwrap();
        buf.write_u32::<LittleEndian>(msg_bytes.len() as u32).unwrap();
        buf.extend_from_slice(msg_bytes);

        buf
    }

    /// Decode from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(ProtocolError::InvalidPayloadLength {
                expected: 8,
                got: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);

        let code = cursor.read_u32::<LittleEndian>()?;
        let msg_len = cursor.read_u32::<LittleEndian>()? as usize;

        if data.len() < 8 + msg_len {
            return Err(ProtocolError::InvalidPayloadLength {
                expected: 8 + msg_len,
                got: data.len(),
            });
        }

        let message = String::from_utf8_lossy(&data[8..8 + msg_len]).to_string();

        Ok(Self { code, message })
    }
}

/// Server capabilities bitmap
pub mod capabilities {
    /// Supports compression
    pub const COMPRESSION: u32 = 1 << 0;
    /// Supports encryption
    pub const ENCRYPTION: u32 = 1 << 1;
    /// Supports streaming
    pub const STREAMING: u32 = 1 << 2;
    /// Supports batch requests
    pub const BATCHING: u32 = 1 << 3;
    /// GPU-accelerated OT
    pub const GPU_OT: u32 = 1 << 4;
    /// Flash attention available
    pub const FLASH_ATTENTION: u32 = 1 << 5;
    /// Tensor parallelism enabled
    pub const TENSOR_PARALLEL: u32 = 1 << 6;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_params_roundtrip() {
        let params = SessionParamsV2::default();
        let encoded = params.encode();
        let decoded = SessionParamsV2::decode(&encoded).unwrap();

        assert_eq!(params.protocol_version, decoded.protocol_version);
        assert_eq!(params.max_prompt_len, decoded.max_prompt_len);
        assert_eq!(params.hidden_dim, decoded.hidden_dim);
        assert_eq!(params.vocab_size, decoded.vocab_size);
    }

    #[test]
    fn test_stream_token_roundtrip() {
        let token = StreamTokenV2 {
            token_id: 12345,
            position: 42,
            is_final: true,
            latency_us: 1500,
        };

        let encoded = token.encode();
        let decoded = StreamTokenV2::decode(&encoded).unwrap();

        assert_eq!(token.token_id, decoded.token_id);
        assert_eq!(token.position, decoded.position);
        assert_eq!(token.is_final, decoded.is_final);
        assert_eq!(token.latency_us, decoded.latency_us);
    }

    #[test]
    fn test_error_response_roundtrip() {
        let error = ErrorResponse::new(500, "Internal server error");
        let encoded = error.encode();
        let decoded = ErrorResponse::decode(&encoded).unwrap();

        assert_eq!(error.code, decoded.code);
        assert_eq!(error.message, decoded.message);
    }
}
