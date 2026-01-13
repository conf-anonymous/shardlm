//! OT_SESSION_PARAMS message payload

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::constants::MsgType;
use crate::error::{ProtocolError, Result};
use crate::messages::Payload;

/// OT_SESSION_PARAMS payload (Server → Client)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtSessionParamsPayload {
    /// Accepted maximum prompt length (64)
    pub accepted_lmax: u16,
    /// Server-provided hidden size d
    pub accepted_d: u16,
    /// Vocabulary size V
    pub vocab_size: u32,
    /// Value type (1 = i32 fixed-point)
    pub value_type: u8,
    /// Fixed-point scale S
    pub fixed_point_scale: u8,
    /// Row layout (1 = contiguous i32[d])
    pub row_layout: u8,
    /// Selected OT suite ID
    pub selected_ot_suite: u16,
    /// Suite flags
    pub suite_flags: u16,
    /// Session time-to-live in seconds
    pub session_ttl_sec: u32,
    /// Maximum requests allowed in session
    pub max_requests: u32,
}

impl Default for OtSessionParamsPayload {
    fn default() -> Self {
        Self {
            accepted_lmax: 1024,
            accepted_d: 2048, // TinyLlama hidden size
            vocab_size: 32000, // TinyLlama vocab size
            value_type: 1,
            fixed_point_scale: 16,
            row_layout: 1,
            selected_ot_suite: 0x0101, // IKNP
            suite_flags: 0,
            session_ttl_sec: 3600,
            max_requests: 10000,
        }
    }
}

impl Payload for OtSessionParamsPayload {
    const MSG_TYPE: MsgType = MsgType::OtSessionParams;

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(24);

        buf.write_u16::<LittleEndian>(self.accepted_lmax).unwrap();
        buf.write_u16::<LittleEndian>(self.accepted_d).unwrap();
        buf.write_u32::<LittleEndian>(self.vocab_size).unwrap();
        buf.write_u8(self.value_type).unwrap();
        buf.write_u8(self.fixed_point_scale).unwrap();
        buf.write_u8(self.row_layout).unwrap();
        buf.write_u8(0).unwrap(); // reserved
        buf.write_u16::<LittleEndian>(self.selected_ot_suite)
            .unwrap();
        buf.write_u16::<LittleEndian>(self.suite_flags).unwrap();
        buf.write_u32::<LittleEndian>(self.session_ttl_sec).unwrap();
        buf.write_u32::<LittleEndian>(self.max_requests).unwrap();

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 24 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let accepted_lmax = cursor.read_u16::<LittleEndian>()?;
        let accepted_d = cursor.read_u16::<LittleEndian>()?;
        let vocab_size = cursor.read_u32::<LittleEndian>()?;
        let value_type = cursor.read_u8()?;
        let fixed_point_scale = cursor.read_u8()?;
        let row_layout = cursor.read_u8()?;
        let _reserved = cursor.read_u8()?;
        let selected_ot_suite = cursor.read_u16::<LittleEndian>()?;
        let suite_flags = cursor.read_u16::<LittleEndian>()?;
        let session_ttl_sec = cursor.read_u32::<LittleEndian>()?;
        let max_requests = cursor.read_u32::<LittleEndian>()?;

        Ok(Self {
            accepted_lmax,
            accepted_d,
            vocab_size,
            value_type,
            fixed_point_scale,
            row_layout,
            selected_ot_suite,
            suite_flags,
            session_ttl_sec,
            max_requests,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_params_roundtrip() {
        let payload = OtSessionParamsPayload::default();
        let encoded = payload.encode();
        let decoded = OtSessionParamsPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }
}
