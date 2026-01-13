//! ERROR message payload

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::{ErrorCode, MsgType};
use crate::error::{ProtocolError, Result};
use crate::messages::Payload;

/// ERROR payload
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorPayload {
    /// Error code
    pub err_code: u32,
    /// Error detail code
    pub err_detail: u32,
    /// Human-readable error message
    pub message: String,
}

impl ErrorPayload {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            err_code: code.to_u32(),
            err_detail: 0,
            message: message.into(),
        }
    }

    pub fn with_detail(code: ErrorCode, detail: u32, message: impl Into<String>) -> Self {
        Self {
            err_code: code.to_u32(),
            err_detail: detail,
            message: message.into(),
        }
    }

    pub fn error_code(&self) -> Option<ErrorCode> {
        ErrorCode::from_u32(self.err_code)
    }
}

impl Payload for ErrorPayload {
    const MSG_TYPE: MsgType = MsgType::Error;

    fn encode(&self) -> Vec<u8> {
        let msg_bytes = self.message.as_bytes();
        let mut buf = Vec::with_capacity(12 + msg_bytes.len());

        buf.write_u32::<LittleEndian>(self.err_code).unwrap();
        buf.write_u32::<LittleEndian>(self.err_detail).unwrap();
        buf.write_u16::<LittleEndian>(msg_bytes.len() as u16)
            .unwrap();
        buf.write_u16::<LittleEndian>(0).unwrap(); // reserved
        buf.write_all(msg_bytes).unwrap();

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let err_code = cursor.read_u32::<LittleEndian>()?;
        let err_detail = cursor.read_u32::<LittleEndian>()?;
        let message_len = cursor.read_u16::<LittleEndian>()? as usize;
        let _reserved = cursor.read_u16::<LittleEndian>()?;

        let pos = cursor.position() as usize;
        if data.len() < pos + message_len {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut msg_bytes = vec![0u8; message_len];
        cursor.read_exact(&mut msg_bytes)?;

        let message = String::from_utf8_lossy(&msg_bytes).into_owned();

        Ok(Self {
            err_code,
            err_detail,
            message,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_roundtrip() {
        let payload = ErrorPayload::new(ErrorCode::SessionNotFound, "Session not found");
        let encoded = payload.encode();
        let decoded = ErrorPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
        assert_eq!(decoded.error_code(), Some(ErrorCode::SessionNotFound));
    }

    #[test]
    fn test_error_with_detail() {
        let payload = ErrorPayload::with_detail(ErrorCode::BadCtr, 42, "Expected counter 42");
        let encoded = payload.encode();
        let decoded = ErrorPayload::decode(&encoded).unwrap();
        assert_eq!(decoded.err_detail, 42);
    }
}
