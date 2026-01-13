//! ShardLM Test Harness
//!
//! Plaintext reference implementation and integration tests for validating
//! the secure computation produces correct results.

mod error;
mod plaintext;
mod secure;
mod pipeline;

pub use error::{HarnessError, Result};
pub use plaintext::PlaintextInference;
pub use secure::SecureInference;
pub use pipeline::{ShardLmPipeline, InferenceResult};
