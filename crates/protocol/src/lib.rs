//! ShardLM Protocol Crate
//!
//! Wire format implementation for ShardLM v1 OT protocol.
//! Implements framing, message types, and CRC validation per WIRE_FORMAT_SPEC.md

mod constants;
mod error;
mod frame;
mod header;
mod messages;

pub use constants::*;
pub use error::{ProtocolError, Result};
pub use frame::Frame;
pub use header::Header;
pub use messages::*;
