//! OT_BASE_OT_MSG message payload

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::MsgType;
use crate::error::{ProtocolError, Result};
use crate::messages::Payload;

/// OT_BASE_OT_MSG payload (Bidirectional)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtBaseOtMsgPayload {
    /// Phase number (starts at 1, increments)
    pub phase: u16,
    /// Whether this is the final message in the handshake
    pub is_final: bool,
    /// Opaque blob containing OT library's base-OT bytes
    pub blob: Vec<u8>,
}

impl OtBaseOtMsgPayload {
    pub fn new(phase: u16, is_final: bool, blob: Vec<u8>) -> Self {
        Self {
            phase,
            is_final,
            blob,
        }
    }
}

impl Payload for OtBaseOtMsgPayload {
    const MSG_TYPE: MsgType = MsgType::OtBaseOtMsg;

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.blob.len());

        buf.write_u16::<LittleEndian>(self.phase).unwrap();
        buf.write_u16::<LittleEndian>(if self.is_final { 1 } else { 0 })
            .unwrap();
        buf.write_u32::<LittleEndian>(self.blob.len() as u32)
            .unwrap();
        buf.write_all(&self.blob).unwrap();

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let phase = cursor.read_u16::<LittleEndian>()?;
        let is_final_raw = cursor.read_u16::<LittleEndian>()?;
        let is_final = is_final_raw != 0;
        let blob_len = cursor.read_u32::<LittleEndian>()? as usize;

        let pos = cursor.position() as usize;
        if data.len() < pos + blob_len {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut blob = vec![0u8; blob_len];
        cursor.read_exact(&mut blob)?;

        Ok(Self {
            phase,
            is_final,
            blob,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_ot_roundtrip() {
        let payload = OtBaseOtMsgPayload::new(1, false, vec![1, 2, 3, 4, 5]);
        let encoded = payload.encode();
        let decoded = OtBaseOtMsgPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn test_base_ot_final() {
        let payload = OtBaseOtMsgPayload::new(3, true, vec![]);
        let encoded = payload.encode();
        let decoded = OtBaseOtMsgPayload::decode(&encoded).unwrap();
        assert!(decoded.is_final);
        assert_eq!(decoded.phase, 3);
    }
}
