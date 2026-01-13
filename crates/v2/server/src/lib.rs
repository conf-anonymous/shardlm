//! ShardLM v2 Server
//!
//! GPU-accelerated Llama 70B inference API server.
//!
//! This server provides:
//! - Distributed inference across multiple GPUs
//! - INT8 quantization for memory efficiency
//! - High-throughput text generation
//! - Compatible API with v1 for easy client migration
//! - Privacy-preserving inference via secret sharing

pub mod config;
pub mod error;
pub mod routes;
pub mod secure_weights;
pub mod state;

pub use config::ServerConfig;
pub use error::{Result, ServerError};
pub use secure_weights::SecureModelWeights;
pub use state::AppState;
