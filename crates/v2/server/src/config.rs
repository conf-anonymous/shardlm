//! Server configuration

use std::path::PathBuf;

use shardlm_v2_core::config::ModelArchitecture;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Path to the model directory (safetensors format)
    pub model_dir: PathBuf,
    /// Model architecture (e.g., Llama3_1_70B, Qwen2_5_7B)
    pub model_architecture: ModelArchitecture,
    /// Host to bind to
    pub host: String,
    /// Port to bind to
    pub port: u16,
    /// Number of GPUs to use
    pub num_gpus: usize,
    /// Maximum tokens to generate per request
    pub max_new_tokens: usize,
    /// Default temperature for sampling
    pub default_temperature: f32,
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Session TTL in seconds
    pub session_ttl_secs: u64,
    /// Maximum sequence length for KV cache (prompt + generation)
    pub max_seq_len: usize,
}

impl ServerConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            model_dir: std::env::var("SHARDLM_V2_MODEL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/data/llama-70b-instruct-weights")),
            model_architecture: std::env::var("SHARDLM_V2_MODEL_ARCH")
                .ok()
                .and_then(|s| ModelArchitecture::from_str(&s))
                .unwrap_or(ModelArchitecture::Llama3_1_70B),
            host: std::env::var("SHARDLM_V2_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("SHARDLM_V2_PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8081), // Default to 8081 to avoid conflict with v1
            num_gpus: std::env::var("SHARDLM_V2_NUM_GPUS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4),
            max_new_tokens: std::env::var("SHARDLM_V2_MAX_NEW_TOKENS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(512),
            default_temperature: std::env::var("SHARDLM_V2_TEMPERATURE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.7),
            max_sessions: std::env::var("SHARDLM_V2_MAX_SESSIONS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
            session_ttl_secs: std::env::var("SHARDLM_V2_SESSION_TTL")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3600), // 1 hour
            max_seq_len: std::env::var("SHARDLM_V2_MAX_SEQ_LEN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8192), // 8K context by default
        }
    }

    /// Get the bind address as a string
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self::from_env()
    }
}
