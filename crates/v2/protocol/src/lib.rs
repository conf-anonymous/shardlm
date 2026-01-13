//! ShardLM v2 Protocol - Extended protocol for production deployment
//!
//! # Protocol Changes from v1
//!
//! | Field | v1 | v2 |
//! |-------|-----|-----|
//! | max_prompt_len | u16 (64K max) | u32 (4B max) |
//! | vocab_size | u32 (32K typical) | u32 (128K for Llama 3.x) |
//! | hidden_dim | u16 (2K typical) | u32 (8K+ for 70B) |
//! | context_len | Not specified | u32 (128K max) |
//! | Protocol version | 1 | 2 |
//!
//! # Wire Format
//!
//! All messages use little-endian encoding with the following header:
//!
//! ```text
//! +----------+----------+---------+---------+----------+
//! | Magic(4) | Type(2)  | Flags(2)| Len(4)  | CRC32(4) |
//! +----------+----------+---------+---------+----------+
//! |                  Payload (Len bytes)               |
//! +----------------------------------------------------+
//! ```

pub mod constants;
pub mod error;
pub mod header;
pub mod messages;

pub use constants::{MsgType, ProtocolVersion, MAGIC_V2};
pub use error::{ProtocolError, Result};
pub use header::Header;
