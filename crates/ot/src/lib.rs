//! ShardLM Oblivious Transfer
//!
//! OT session management and extension interfaces for private embedding retrieval.
//! Implements batched 1-out-of-V OT for vocabulary lookup.

mod error;
mod session;
mod sender;
mod receiver;
mod extension;
mod iknp;

#[cfg(test)]
mod adversarial_tests;

pub use error::{OtError, Result};
pub use session::{OtSession, OtSessionConfig, OtSessionState, SessionId};
pub use sender::OtSender;
pub use receiver::OtReceiver;
pub use extension::{OtExtension, SimpleOtExtension};
pub use iknp::IknpOtExtension;
