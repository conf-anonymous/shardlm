//! Frame encoding/decoding per WIRE_FORMAT_SPEC.md

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::{HEADER_LEN, MAGIC, MsgType, PROTOCOL_VERSION};
use crate::error::{ProtocolError, Result};
use crate::header::Header;

/// Frame preamble size (before header and payload):
/// magic(4) + version(2) + msg_type(2) + header_len(2) + payload_len(4) + header_crc(4) + payload_crc(4) = 22
const PREAMBLE_SIZE: usize = 22;

/// A complete wire frame
#[derive(Debug, Clone)]
pub struct Frame {
    /// Message type
    pub msg_type: MsgType,
    /// Header (48 bytes in v1)
    pub header: Header,
    /// Payload bytes (can be empty)
    pub payload: Vec<u8>,
}

impl Frame {
    /// Create a new frame with the given message type, header, and payload
    pub fn new(msg_type: MsgType, header: Header, payload: Vec<u8>) -> Self {
        Self {
            msg_type,
            header,
            payload,
        }
    }

    /// Encode the frame to bytes
    pub fn encode(&self) -> Vec<u8> {
        let header_bytes = self.header.encode();
        let header_crc = crc32fast::hash(&header_bytes);
        let payload_crc = crc32fast::hash(&self.payload);

        let total_size = PREAMBLE_SIZE + header_bytes.len() + self.payload.len();
        let mut buf = Vec::with_capacity(total_size);

        // Magic
        buf.write_all(&MAGIC).unwrap();
        // Version
        buf.write_u16::<LittleEndian>(PROTOCOL_VERSION).unwrap();
        // Message type
        buf.write_u16::<LittleEndian>(self.msg_type.to_u16())
            .unwrap();
        // Header length
        buf.write_u16::<LittleEndian>(HEADER_LEN).unwrap();
        // Payload length
        buf.write_u32::<LittleEndian>(self.payload.len() as u32)
            .unwrap();
        // Header CRC
        buf.write_u32::<LittleEndian>(header_crc).unwrap();
        // Payload CRC
        buf.write_u32::<LittleEndian>(payload_crc).unwrap();
        // Header
        buf.write_all(&header_bytes).unwrap();
        // Payload
        buf.write_all(&self.payload).unwrap();

        buf
    }

    /// Decode a frame from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < PREAMBLE_SIZE {
            return Err(ProtocolError::BufferTooShort {
                need: PREAMBLE_SIZE,
                have: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);

        // Magic
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(ProtocolError::InvalidMagic);
        }

        // Version
        let version = cursor.read_u16::<LittleEndian>()?;
        if version != PROTOCOL_VERSION {
            return Err(ProtocolError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                got: version,
            });
        }

        // Message type
        let msg_type_raw = cursor.read_u16::<LittleEndian>()?;
        let msg_type =
            MsgType::from_u16(msg_type_raw).ok_or(ProtocolError::UnknownMessageType(msg_type_raw))?;

        // Header length
        let header_len = cursor.read_u16::<LittleEndian>()?;
        if header_len != HEADER_LEN {
            return Err(ProtocolError::InvalidHeaderLength {
                expected: HEADER_LEN,
                got: header_len,
            });
        }

        // Payload length
        let payload_len = cursor.read_u32::<LittleEndian>()? as usize;

        // Header CRC
        let expected_header_crc = cursor.read_u32::<LittleEndian>()?;

        // Payload CRC
        let expected_payload_crc = cursor.read_u32::<LittleEndian>()?;

        // Check we have enough data
        let total_needed = PREAMBLE_SIZE + header_len as usize + payload_len;
        if data.len() < total_needed {
            return Err(ProtocolError::BufferTooShort {
                need: total_needed,
                have: data.len(),
            });
        }

        // Read and verify header
        let header_start = cursor.position() as usize;
        let header_end = header_start + header_len as usize;
        let header_bytes = &data[header_start..header_end];

        let actual_header_crc = crc32fast::hash(header_bytes);
        if actual_header_crc != expected_header_crc {
            return Err(ProtocolError::HeaderCrcMismatch {
                expected: expected_header_crc,
                got: actual_header_crc,
            });
        }

        let header = Header::decode(header_bytes)?;

        // Read and verify payload
        let payload_start = header_end;
        let payload_end = payload_start + payload_len;
        let payload = data[payload_start..payload_end].to_vec();

        let actual_payload_crc = crc32fast::hash(&payload);
        if actual_payload_crc != expected_payload_crc {
            return Err(ProtocolError::PayloadCrcMismatch {
                expected: expected_payload_crc,
                got: actual_payload_crc,
            });
        }

        Ok(Self {
            msg_type,
            header,
            payload,
        })
    }

    /// Get the total encoded size of this frame
    pub fn encoded_size(&self) -> usize {
        PREAMBLE_SIZE + HEADER_LEN as usize + self.payload.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_roundtrip() {
        let header = Header::new([0x11; 16], 42);
        let payload = vec![1, 2, 3, 4, 5];
        let frame = Frame::new(MsgType::EmbedFetchRequest, header.clone(), payload.clone());

        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MsgType::EmbedFetchRequest);
        assert_eq!(decoded.header, header);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_frame_empty_payload() {
        let header = Header::pre_session();
        let frame = Frame::new(MsgType::OtSessionInit, header.clone(), vec![]);

        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();

        assert_eq!(decoded.msg_type, MsgType::OtSessionInit);
        assert_eq!(decoded.payload.len(), 0);
    }

    #[test]
    fn test_invalid_magic() {
        let frame = Frame::new(MsgType::OtSessionInit, Header::pre_session(), vec![]);
        let mut encoded = frame.encode();
        encoded[0] = 0xFF; // Corrupt magic

        let result = Frame::decode(&encoded);
        assert!(matches!(result, Err(ProtocolError::InvalidMagic)));
    }

    #[test]
    fn test_header_crc_mismatch() {
        let frame = Frame::new(MsgType::OtSessionInit, Header::pre_session(), vec![]);
        let mut encoded = frame.encode();
        // Corrupt the header CRC (bytes 14-17)
        encoded[14] = 0xFF;

        let result = Frame::decode(&encoded);
        assert!(matches!(result, Err(ProtocolError::HeaderCrcMismatch { .. })));
    }

    #[test]
    fn test_payload_crc_mismatch() {
        let frame = Frame::new(
            MsgType::OtSessionInit,
            Header::pre_session(),
            vec![1, 2, 3],
        );
        let mut encoded = frame.encode();
        // Corrupt the last payload byte
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;

        let result = Frame::decode(&encoded);
        assert!(matches!(
            result,
            Err(ProtocolError::PayloadCrcMismatch { .. })
        ));
    }
}
