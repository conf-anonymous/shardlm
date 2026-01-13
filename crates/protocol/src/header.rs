//! Common header (48 bytes) per WIRE_FORMAT_SPEC.md

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::{HEADER_LEN, NONCE_LEN, SESSION_ID_LEN};
use crate::error::{ProtocolError, Result};

/// Session identifier (16 bytes)
pub type SessionId = [u8; SESSION_ID_LEN];

/// Nonce (8 bytes)
pub type Nonce = [u8; NONCE_LEN];

/// Common header structure (48 bytes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Session ID (all-zero for pre-session messages)
    pub session_id: SessionId,
    /// Per-session counter (0 for pre-session messages)
    pub ctr: u64,
    /// Flags bitfield (v1 uses 0)
    pub flags: u32,
    /// Reserved (must be 0)
    pub reserved: u32,
    /// Client nonce (random per request, or 0 if not used)
    pub client_nonce: Nonce,
    /// Server nonce (random per response, or 0 if not used)
    pub server_nonce: Nonce,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            session_id: [0u8; SESSION_ID_LEN],
            ctr: 0,
            flags: 0,
            reserved: 0,
            client_nonce: [0u8; NONCE_LEN],
            server_nonce: [0u8; NONCE_LEN],
        }
    }
}

impl Header {
    /// Create a new header with the given session ID and counter
    pub fn new(session_id: SessionId, ctr: u64) -> Self {
        Self {
            session_id,
            ctr,
            ..Default::default()
        }
    }

    /// Create a pre-session header (all zeros except optional client nonce)
    pub fn pre_session() -> Self {
        Self::default()
    }

    /// Set the client nonce
    pub fn with_client_nonce(mut self, nonce: Nonce) -> Self {
        self.client_nonce = nonce;
        self
    }

    /// Set the server nonce
    pub fn with_server_nonce(mut self, nonce: Nonce) -> Self {
        self.server_nonce = nonce;
        self
    }

    /// Encode the header to bytes
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_LEN as usize);
        self.write_to(&mut buf).expect("Vec write should not fail");
        buf
    }

    /// Write the header to a writer
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.session_id)?;
        writer.write_u64::<LittleEndian>(self.ctr)?;
        writer.write_u32::<LittleEndian>(self.flags)?;
        writer.write_u32::<LittleEndian>(self.reserved)?;
        writer.write_all(&self.client_nonce)?;
        writer.write_all(&self.server_nonce)?;
        Ok(())
    }

    /// Decode a header from bytes
    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < HEADER_LEN as usize {
            return Err(ProtocolError::BufferTooShort {
                need: HEADER_LEN as usize,
                have: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);

        let mut session_id = [0u8; SESSION_ID_LEN];
        cursor.read_exact(&mut session_id)?;

        let ctr = cursor.read_u64::<LittleEndian>()?;
        let flags = cursor.read_u32::<LittleEndian>()?;
        let reserved = cursor.read_u32::<LittleEndian>()?;

        let mut client_nonce = [0u8; NONCE_LEN];
        cursor.read_exact(&mut client_nonce)?;

        let mut server_nonce = [0u8; NONCE_LEN];
        cursor.read_exact(&mut server_nonce)?;

        Ok(Self {
            session_id,
            ctr,
            flags,
            reserved,
            client_nonce,
            server_nonce,
        })
    }

    /// Check if this is a pre-session header (session_id is all zeros)
    pub fn is_pre_session(&self) -> bool {
        self.session_id == [0u8; SESSION_ID_LEN]
    }

    /// Generate a random client nonce
    pub fn generate_client_nonce(&mut self) {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut self.client_nonce);
    }

    /// Generate a random server nonce
    pub fn generate_server_nonce(&mut self) {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut self.server_nonce);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_roundtrip() {
        let header = Header {
            session_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ctr: 12345,
            flags: 0,
            reserved: 0,
            client_nonce: [0xAA; NONCE_LEN],
            server_nonce: [0xBB; NONCE_LEN],
        };

        let encoded = header.encode();
        assert_eq!(encoded.len(), HEADER_LEN as usize);

        let decoded = Header::decode(&encoded).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn test_pre_session_header() {
        let header = Header::pre_session();
        assert!(header.is_pre_session());
        assert_eq!(header.ctr, 0);
    }
}
