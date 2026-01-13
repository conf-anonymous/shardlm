//! OT_SESSION_INIT message payload

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Cursor, Read, Write};

use crate::constants::{MsgType, OtSuiteId};
use crate::error::{ProtocolError, Result};
use crate::messages::Payload;

/// OT suite entry in session init
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtSuiteEntry {
    pub suite_id: u16,
    pub flags: u16,
}

/// OT_SESSION_INIT payload (Client → Server)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtSessionInitPayload {
    /// Client capabilities bitmap
    pub client_caps: u32,
    /// Requested maximum prompt length (must be 64 for v1)
    pub requested_lmax: u16,
    /// Requested hidden size (0 means server decides)
    pub requested_d: u16,
    /// Value type (1 = i32 fixed-point)
    pub value_type: u8,
    /// Fixed-point scale S (e.g., 16 for 2^16)
    pub fixed_point_scale: u8,
    /// Row layout (1 = contiguous i32[d])
    pub row_layout: u8,
    /// Supported OT suites
    pub ot_suites: Vec<OtSuiteEntry>,
}

impl Default for OtSessionInitPayload {
    fn default() -> Self {
        Self {
            client_caps: 0,
            requested_lmax: 64,
            requested_d: 0,
            value_type: 1,
            fixed_point_scale: 16,
            row_layout: 1,
            ot_suites: vec![OtSuiteEntry {
                suite_id: OtSuiteId::IknpExtension.to_u16(),
                flags: 0,
            }],
        }
    }
}

impl Payload for OtSessionInitPayload {
    const MSG_TYPE: MsgType = MsgType::OtSessionInit;

    fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.write_u32::<LittleEndian>(self.client_caps).unwrap();
        buf.write_u16::<LittleEndian>(self.requested_lmax).unwrap();
        buf.write_u16::<LittleEndian>(self.requested_d).unwrap();
        buf.write_u8(self.value_type).unwrap();
        buf.write_u8(self.fixed_point_scale).unwrap();
        buf.write_u8(self.row_layout).unwrap();
        buf.write_u8(0).unwrap(); // reserved
        buf.write_u8(self.ot_suites.len() as u8).unwrap();
        buf.write_all(&[0, 0, 0]).unwrap(); // reserved2

        for suite in &self.ot_suites {
            buf.write_u16::<LittleEndian>(suite.suite_id).unwrap();
            buf.write_u16::<LittleEndian>(suite.flags).unwrap();
        }

        buf
    }

    fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 16 {
            return Err(ProtocolError::InvalidPayloadLength);
        }

        let mut cursor = Cursor::new(data);

        let client_caps = cursor.read_u32::<LittleEndian>()?;
        let requested_lmax = cursor.read_u16::<LittleEndian>()?;
        let requested_d = cursor.read_u16::<LittleEndian>()?;
        let value_type = cursor.read_u8()?;
        let fixed_point_scale = cursor.read_u8()?;
        let row_layout = cursor.read_u8()?;
        let _reserved = cursor.read_u8()?;
        let ot_suite_count = cursor.read_u8()?;
        let mut reserved2 = [0u8; 3];
        cursor.read_exact(&mut reserved2)?;

        let mut ot_suites = Vec::with_capacity(ot_suite_count as usize);
        for _ in 0..ot_suite_count {
            let suite_id = cursor.read_u16::<LittleEndian>()?;
            let flags = cursor.read_u16::<LittleEndian>()?;
            ot_suites.push(OtSuiteEntry { suite_id, flags });
        }

        Ok(Self {
            client_caps,
            requested_lmax,
            requested_d,
            value_type,
            fixed_point_scale,
            row_layout,
            ot_suites,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_init_roundtrip() {
        let payload = OtSessionInitPayload::default();
        let encoded = payload.encode();
        let decoded = OtSessionInitPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn test_session_init_multiple_suites() {
        let payload = OtSessionInitPayload {
            ot_suites: vec![
                OtSuiteEntry {
                    suite_id: OtSuiteId::IknpExtension.to_u16(),
                    flags: 0,
                },
                OtSuiteEntry {
                    suite_id: OtSuiteId::KosExtension.to_u16(),
                    flags: 1,
                },
            ],
            ..Default::default()
        };
        let encoded = payload.encode();
        let decoded = OtSessionInitPayload::decode(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }
}
