//! ShardLM Server
//!
//! Privacy-preserving LLM inference API server.
//!
//! This server implements the ShardLM protocol endpoints for:
//! - Session management (create, expire)
//! - OT handshake (base OT initialization)
//! - Embedding fetch (OT extension)
//!
//! The server never sees plaintext tokens or outputs - all sensitive
//! data is protected via oblivious transfer protocols.

pub mod config;
pub mod error;
pub mod session;
pub mod routes;
pub mod state;

pub use config::ServerConfig;
pub use error::{ServerError, Result};
pub use state::AppState;
