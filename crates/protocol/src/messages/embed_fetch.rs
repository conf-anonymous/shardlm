//! EMBED_FETCH_REQUEST and EMBED_FETCH_RESPONSE message payloads

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::{L_MAX, MsgType, QueryFormat};
use crate::error::{ProtocolError, Result};
use crate::messages::Payload;

/// EMBED_FETCH_REQUEST payload (Client → Server)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedFetchRequestPayload {
    /// Number of tokens requested (1..64)
    pub len: u16,
    /// Query format (1 = BatchedOTv1)
    pub query_format: u16,
    /// Opaque query blob for OT extension
    pub query_blob: Vec<u8>,
}

impl EmbedFetchRequestPayload {
    pub fn new(len: u16, query_blob: Vec<u8>) -> Result<Self> {
        if len == 0 || len > L_MAX {
            return Err(ProtocolError::PromptTooLong { len, max: L_MAX });
        }
        Ok(Self {
            len,
            query_format: QueryFormat::BatchedOtV1 as u16,
            query_blob,
        })
    }
}

impl Payload for EmbedFetchRequestPayload {
    const MSG_TYPE: MsgType = MsgType::EmbedFetchRequest;

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.query_blob.len());

        buf.write_u16::<LittleEndian>(self.len).unwrap();
        buf.write_u16::<LittleEndian>(0).unwrap(); // reserved
        buf.write_u16::<LittleEndian>(self.query_format).unwrap();
        buf.write_u16::<LittleEndian>(0).unwrap(); // reserved2
        buf.write_u32::<LittleEndian>(self.query_blob.len() as u32)
            .unwrap();
        buf.write_all(&self.query_blob).unwrap();

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let len = cursor.read_u16::<LittleEndian>()?;
        let _reserved = cursor.read_u16::<LittleEndian>()?;
        let query_format = cursor.read_u16::<LittleEndian>()?;
        let _reserved2 = cursor.read_u16::<LittleEndian>()?;
        let query_blob_len = cursor.read_u32::<LittleEndian>()? as usize;

        let pos = cursor.position() as usize;
        if data.len() < pos + query_blob_len {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut query_blob = vec![0u8; query_blob_len];
        cursor.read_exact(&mut query_blob)?;

        if len == 0 || len > L_MAX {
            return Err(ProtocolError::PromptTooLong { len, max: L_MAX });
        }

        Ok(Self {
            len,
            query_format,
            query_blob,
        })
    }
}

/// EMBED_FETCH_RESPONSE payload (Server → Client)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedFetchResponsePayload {
    /// Number of tokens (echo from request)
    pub len: u16,
    /// Row size in bytes (= d * 4)
    pub row_bytes: u32,
    /// Response format (1 = BatchedOTv1)
    pub response_format: u16,
    /// Opaque response blob from OT extension
    pub response_blob: Vec<u8>,
}

impl EmbedFetchResponsePayload {
    pub fn new(len: u16, row_bytes: u32, response_blob: Vec<u8>) -> Self {
        Self {
            len,
            row_bytes,
            response_format: QueryFormat::BatchedOtV1 as u16,
            response_blob,
        }
    }
}

impl Payload for EmbedFetchResponsePayload {
    const MSG_TYPE: MsgType = MsgType::EmbedFetchResponse;

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.response_blob.len());

        buf.write_u16::<LittleEndian>(self.len).unwrap();
        buf.write_u16::<LittleEndian>(0).unwrap(); // reserved
        buf.write_u32::<LittleEndian>(self.row_bytes).unwrap();
        buf.write_u16::<LittleEndian>(self.response_format).unwrap();
        buf.write_u16::<LittleEndian>(0).unwrap(); // reserved2
        buf.write_u32::<LittleEndian>(self.response_blob.len() as u32)
            .unwrap();
        buf.write_all(&self.response_blob).unwrap();

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let len = cursor.read_u16::<LittleEndian>()?;
        let _reserved = cursor.read_u16::<LittleEndian>()?;
        let row_bytes = cursor.read_u32::<LittleEndian>()?;
        let response_format = cursor.read_u16::<LittleEndian>()?;
        let _reserved2 = cursor.read_u16::<LittleEndian>()?;
        let response_blob_len = cursor.read_u32::<LittleEndian>()? as usize;

        let pos = cursor.position() as usize;
        if data.len() < pos + response_blob_len {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut response_blob = vec![0u8; response_blob_len];
        cursor.read_exact(&mut response_blob)?;

        Ok(Self {
            len,
            row_bytes,
            response_format,
            response_blob,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_fetch_request_roundtrip() {
        let payload = EmbedFetchRequestPayload::new(32, vec![1, 2, 3, 4]).unwrap();
        let encoded = payload.encode();
        let decoded = EmbedFetchRequestPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn test_embed_fetch_request_max_len() {
        let payload = EmbedFetchRequestPayload::new(64, vec![]).unwrap();
        assert_eq!(payload.len, 64);
    }

    #[test]
    fn test_embed_fetch_request_too_long() {
        let result = EmbedFetchRequestPayload::new(65, vec![]);
        assert!(matches!(
            result,
            Err(ProtocolError::PromptTooLong { len: 65, max: 64 })
        ));
    }

    #[test]
    fn test_embed_fetch_response_roundtrip() {
        let payload = EmbedFetchResponsePayload::new(32, 8192, vec![5, 6, 7, 8]);
        let encoded = payload.encode();
        let decoded = EmbedFetchResponsePayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }
}
