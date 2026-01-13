//! Protocol header

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::Cursor;

use crate::constants::{MsgType, HEADER_SIZE, MAGIC_V2, MAX_PAYLOAD_SIZE};
use crate::error::{ProtocolError, Result};

/// Protocol message header
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Message type
    pub msg_type: MsgType,
    /// Message flags
    pub flags: u16,
    /// Payload length in bytes
    pub payload_len: u32,
    /// CRC32 of payload
    pub crc32: u32,
}

impl Header {
    /// Create a new header
    pub fn new(msg_type: MsgType, flags: u16, payload_len: u32, crc32: u32) -> Self {
        Self {
            msg_type,
            flags,
            payload_len,
            crc32,
        }
    }

    /// Create header for payload with automatic CRC calculation
    pub fn for_payload(msg_type: MsgType, flags: u16, payload: &[u8]) -> Self {
        let crc32 = crc32fast::hash(payload);
        Self::new(msg_type, flags, payload.len() as u32, crc32)
    }

    /// Encode header to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.extend_from_slice(&MAGIC_V2);
        buf.write_u16::<LittleEndian>(self.msg_type as u16).unwrap();
        buf.write_u16::<LittleEndian>(self.flags).unwrap();
        buf.write_u32::<LittleEndian>(self.payload_len).unwrap();
        buf.write_u32::<LittleEndian>(self.crc32).unwrap();
        buf
    }

    /// Decode header from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_SIZE {
            return Err(ProtocolError::InvalidPayloadLength {
                expected: HEADER_SIZE,
                got: data.len(),
            });
        }

        // Check magic
        if &data[0..4] != MAGIC_V2 {
            return Err(ProtocolError::InvalidMagic);
        }

        let mut cursor = Cursor::new(&data[4..]);

        let msg_type_raw = cursor.read_u16::<LittleEndian>()?;
        let msg_type = MsgType::from_u16(msg_type_raw)
            .ok_or(ProtocolError::UnknownMessageType(msg_type_raw))?;

        let flags = cursor.read_u16::<LittleEndian>()?;
        let payload_len = cursor.read_u32::<LittleEndian>()?;
        let crc32 = cursor.read_u32::<LittleEndian>()?;

        if payload_len as usize > MAX_PAYLOAD_SIZE {
            return Err(ProtocolError::PayloadTooLarge {
                max: MAX_PAYLOAD_SIZE,
                got: payload_len as usize,
            });
        }

        Ok(Self {
            msg_type,
            flags,
            payload_len,
            crc32,
        })
    }

    /// Verify payload CRC
    pub fn verify_crc(&self, payload: &[u8]) -> Result<()> {
        let computed = crc32fast::hash(payload);
        if computed != self.crc32 {
            return Err(ProtocolError::CrcMismatch {
                expected: self.crc32,
                got: computed,
            });
        }
        Ok(())
    }

    /// Check if payload is compressed
    pub fn is_compressed(&self) -> bool {
        self.flags & crate::constants::flags::COMPRESSED != 0
    }

    /// Check if payload is encrypted
    pub fn is_encrypted(&self) -> bool {
        self.flags & crate::constants::flags::ENCRYPTED != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = Header::new(MsgType::QueryRequest, 0, 1024, 0xDEADBEEF);
        let encoded = header.encode();
        let decoded = Header::decode(&encoded).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn test_header_for_payload() {
        let payload = b"test payload data";
        let header = Header::for_payload(MsgType::QueryRequest, 0, payload);
        assert_eq!(header.payload_len, payload.len() as u32);
        header.verify_crc(payload).unwrap();
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = MAGIC_V2.to_vec();
        data[0] = 0x00; // Corrupt magic
        data.extend_from_slice(&[0u8; 12]);

        let result = Header::decode(&data);
        assert!(matches!(result, Err(ProtocolError::InvalidMagic)));
    }
}
