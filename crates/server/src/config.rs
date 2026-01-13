//! Server configuration

use std::path::PathBuf;
use std::time::Duration;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server bind address
    pub bind_addr: String,

    /// Server port
    pub port: u16,

    /// Path to model weights directory
    pub model_dir: PathBuf,

    /// Maximum concurrent sessions
    pub max_sessions: usize,

    /// Session time-to-live
    pub session_ttl: Duration,

    /// Maximum requests per session
    pub max_requests_per_session: u32,

    /// Maximum prompt length (tokens)
    pub max_prompt_len: u16,

    /// OT security parameter (kappa)
    pub ot_kappa: u8,

    /// CORS allowed origins
    pub cors_origins: Vec<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            model_dir: PathBuf::from("tinyllama-weights"),
            max_sessions: 1000,
            session_ttl: Duration::from_secs(900), // 15 minutes
            max_requests_per_session: 1000,
            max_prompt_len: 1024,
            ot_kappa: 128,
            cors_origins: vec!["http://localhost:3000".to_string()],
        }
    }
}

impl ServerConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(addr) = std::env::var("SHARDLM_BIND_ADDR") {
            config.bind_addr = addr;
        }

        if let Ok(port) = std::env::var("SHARDLM_PORT") {
            if let Ok(p) = port.parse() {
                config.port = p;
            }
        }

        if let Ok(dir) = std::env::var("SHARDLM_MODEL_DIR") {
            config.model_dir = PathBuf::from(dir);
        }

        if let Ok(max) = std::env::var("SHARDLM_MAX_SESSIONS") {
            if let Ok(m) = max.parse() {
                config.max_sessions = m;
            }
        }

        if let Ok(ttl) = std::env::var("SHARDLM_SESSION_TTL_SECS") {
            if let Ok(t) = ttl.parse() {
                config.session_ttl = Duration::from_secs(t);
            }
        }

        if let Ok(origins) = std::env::var("SHARDLM_CORS_ORIGINS") {
            config.cors_origins = origins.split(',').map(|s| s.trim().to_string()).collect();
        }

        if let Ok(len) = std::env::var("SHARDLM_MAX_PROMPT_LEN") {
            if let Ok(l) = len.parse() {
                config.max_prompt_len = l;
            }
        }

        config
    }

    /// Get the full bind address
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.bind_addr, self.port)
    }
}
