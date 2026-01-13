//! OT_SESSION_READY message payload

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::MsgType;
use crate::error::{ProtocolError, Result};
use crate::messages::Payload;

/// OT_SESSION_READY payload (Server → Client)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtSessionReadyPayload {
    /// Success indicator (1 = ok)
    pub ok: bool,
    /// Session time-to-live in seconds
    pub session_ttl_sec: u32,
    /// Maximum requests allowed in session
    pub max_requests: u32,
    /// Starting counter value (MUST be 1 for v1)
    pub starting_ctr: u64,
}

impl Default for OtSessionReadyPayload {
    fn default() -> Self {
        Self {
            ok: true,
            session_ttl_sec: 3600,
            max_requests: 10000,
            starting_ctr: 1,
        }
    }
}

impl Payload for OtSessionReadyPayload {
    const MSG_TYPE: MsgType = MsgType::OtSessionReady;

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);

        buf.write_u8(if self.ok { 1 } else { 0 }).unwrap();
        buf.write_all(&[0, 0, 0]).unwrap(); // reserved
        buf.write_u32::<LittleEndian>(self.session_ttl_sec).unwrap();
        buf.write_u32::<LittleEndian>(self.max_requests).unwrap();
        buf.write_u64::<LittleEndian>(self.starting_ctr).unwrap();

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 20 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let ok = cursor.read_u8()? != 0;
        let mut reserved = [0u8; 3];
        cursor.read_exact(&mut reserved)?;
        let session_ttl_sec = cursor.read_u32::<LittleEndian>()?;
        let max_requests = cursor.read_u32::<LittleEndian>()?;
        let starting_ctr = cursor.read_u64::<LittleEndian>()?;

        Ok(Self {
            ok,
            session_ttl_sec,
            max_requests,
            starting_ctr,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_ready_roundtrip() {
        let payload = OtSessionReadyPayload::default();
        let encoded = payload.encode();
        let decoded = OtSessionReadyPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }
}
