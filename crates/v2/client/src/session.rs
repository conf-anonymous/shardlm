//! Session management for ShardLM V2 client

use serde::{Deserialize, Serialize};

/// Session information returned from server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    /// Unique session ID
    pub session_id: String,
    /// Model name (e.g., "Qwen2.5-1.5B-Instruct")
    pub model_name: String,
    /// Maximum new tokens that can be generated
    pub max_new_tokens: usize,
}

/// Create session request
#[derive(Debug, Serialize)]
pub struct CreateSessionRequest {
    /// Optional client identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

impl Default for CreateSessionRequest {
    fn default() -> Self {
        Self { client_id: None }
    }
}

/// Session status request
#[derive(Debug, Serialize)]
pub struct SessionStatusRequest {
    pub session_id: String,
}

/// Session status response
#[derive(Debug, Deserialize)]
pub struct SessionStatusResponse {
    pub session_id: String,
    pub active: bool,
    pub request_count: u64,
    pub age_secs: u64,
}

/// Model information
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Intermediate dimension (FFN)
    pub intermediate_dim: usize,
}

/// Server info response
#[derive(Debug, Clone, Deserialize)]
pub struct ServerInfo {
    pub version: String,
    pub model_name: String,
    #[serde(default)]
    pub hidden_dim: usize,
    #[serde(default)]
    pub num_layers: usize,
    #[serde(default)]
    pub num_heads: usize,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub num_gpus: usize,
    #[serde(default)]
    pub uptime_secs: u64,
    #[serde(default)]
    pub active_sessions: usize,
    #[serde(default)]
    pub max_new_tokens: usize,
    #[serde(default)]
    pub default_temperature: f32,
    #[serde(default)]
    pub cuda_available: bool,
    #[serde(default)]
    pub gpu_name: Option<String>,
}
