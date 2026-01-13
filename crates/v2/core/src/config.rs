//! v2 Configuration for production deployment

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Supported model architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Llama 3.1 8B (for development/testing)
    Llama3_1_8B,
    /// Llama 3.1 70B (production target)
    Llama3_1_70B,
    /// Llama 3.3 70B (latest production target)
    Llama3_3_70B,
    /// Llama 3.1 405B (enterprise target)
    Llama3_1_405B,
    /// Qwen 2.5 0.5B Instruct (tiny, for testing)
    Qwen2_5_0_5B,
    /// Qwen 2.5 1.5B Instruct (small, for testing)
    Qwen2_5_1_5B,
    /// Qwen 2.5 3B Instruct (small)
    Qwen2_5_3B,
    /// Qwen 2.5 7B Instruct
    Qwen2_5_7B,
    /// Qwen 2.5 14B Instruct (future)
    Qwen2_5_14B,
    /// Qwen 2.5 72B Instruct (future)
    Qwen2_5_72B,
}

impl ModelArchitecture {
    /// Get the hidden dimension for this architecture
    pub fn hidden_dim(&self) -> usize {
        match self {
            Self::Llama3_1_8B => 4096,
            Self::Llama3_1_70B | Self::Llama3_3_70B => 8192,
            Self::Llama3_1_405B => 16384,
            Self::Qwen2_5_0_5B => 896,
            Self::Qwen2_5_1_5B => 1536,
            Self::Qwen2_5_3B => 2048,
            Self::Qwen2_5_7B => 3584,
            Self::Qwen2_5_14B => 5120,
            Self::Qwen2_5_72B => 8192,
        }
    }

    /// Get the number of attention heads
    pub fn num_heads(&self) -> usize {
        match self {
            Self::Llama3_1_8B => 32,
            Self::Llama3_1_70B | Self::Llama3_3_70B => 64,
            Self::Llama3_1_405B => 128,
            Self::Qwen2_5_0_5B => 14,
            Self::Qwen2_5_1_5B => 12,
            Self::Qwen2_5_3B => 16,
            Self::Qwen2_5_7B => 28,
            Self::Qwen2_5_14B => 40,
            Self::Qwen2_5_72B => 64,
        }
    }

    /// Get the number of key-value heads (for GQA)
    pub fn num_kv_heads(&self) -> usize {
        match self {
            Self::Llama3_1_8B => 8,
            Self::Llama3_1_70B | Self::Llama3_3_70B => 8,
            Self::Llama3_1_405B => 8,
            Self::Qwen2_5_0_5B => 2,
            Self::Qwen2_5_1_5B => 2,
            Self::Qwen2_5_3B => 2,
            Self::Qwen2_5_7B => 4,
            Self::Qwen2_5_14B => 8,
            Self::Qwen2_5_72B => 8,
        }
    }

    /// Get the number of layers
    pub fn num_layers(&self) -> usize {
        match self {
            Self::Llama3_1_8B => 32,
            Self::Llama3_1_70B | Self::Llama3_3_70B => 80,
            Self::Llama3_1_405B => 126,
            Self::Qwen2_5_0_5B => 24,
            Self::Qwen2_5_1_5B => 28,
            Self::Qwen2_5_3B => 36,
            Self::Qwen2_5_7B => 28,
            Self::Qwen2_5_14B => 48,
            Self::Qwen2_5_72B => 80,
        }
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Qwen2_5_0_5B | Self::Qwen2_5_1_5B | Self::Qwen2_5_3B |
            Self::Qwen2_5_7B | Self::Qwen2_5_14B | Self::Qwen2_5_72B => 152064,
            _ => 128256, // Llama 3.x uses 128K vocabulary
        }
    }

    /// Get the maximum context length
    pub fn max_context_len(&self) -> usize {
        // Both Llama 3.x and Qwen 2.5 support 128K context
        131072
    }

    /// Get the intermediate (FFN) dimension
    pub fn intermediate_dim(&self) -> usize {
        match self {
            Self::Llama3_1_8B => 14336,
            Self::Llama3_1_70B | Self::Llama3_3_70B => 28672,
            Self::Llama3_1_405B => 53248,
            Self::Qwen2_5_0_5B => 4864,
            Self::Qwen2_5_1_5B => 8960,
            Self::Qwen2_5_3B => 11008,
            Self::Qwen2_5_7B => 18944,
            Self::Qwen2_5_14B => 13824,
            Self::Qwen2_5_72B => 29568,
        }
    }

    /// Get the RoPE theta value
    pub fn rope_theta(&self) -> f32 {
        match self {
            Self::Qwen2_5_0_5B | Self::Qwen2_5_1_5B | Self::Qwen2_5_3B |
            Self::Qwen2_5_7B | Self::Qwen2_5_14B | Self::Qwen2_5_72B => 1000000.0,
            _ => 500000.0, // Llama 3.x
        }
    }

    /// Get the RMS normalization epsilon
    pub fn rms_norm_eps(&self) -> f32 {
        match self {
            Self::Qwen2_5_0_5B | Self::Qwen2_5_1_5B | Self::Qwen2_5_3B |
            Self::Qwen2_5_7B | Self::Qwen2_5_14B | Self::Qwen2_5_72B => 1e-6,
            _ => 1e-5, // Llama 3.x
        }
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_dim() / self.num_heads()
    }

    /// Get model family name
    pub fn model_family(&self) -> &'static str {
        match self {
            Self::Qwen2_5_0_5B | Self::Qwen2_5_1_5B | Self::Qwen2_5_3B |
            Self::Qwen2_5_7B | Self::Qwen2_5_14B | Self::Qwen2_5_72B => "qwen2.5",
            _ => "llama3",
        }
    }

    /// Estimate VRAM requirement in GB (BF16)
    pub fn vram_estimate_gb(&self) -> f32 {
        match self {
            Self::Llama3_1_8B => 16.0,
            Self::Llama3_1_70B | Self::Llama3_3_70B => 140.0, // 2x H100 80GB
            Self::Llama3_1_405B => 810.0, // 10+ H100 80GB
            Self::Qwen2_5_0_5B => 1.0,   // Any GPU
            Self::Qwen2_5_1_5B => 3.0,   // Any GPU
            Self::Qwen2_5_3B => 6.0,     // Any GPU
            Self::Qwen2_5_7B => 14.0,    // 1x A10G 24GB or 1x A100 40GB
            Self::Qwen2_5_14B => 28.0,   // 1x A100 40GB
            Self::Qwen2_5_72B => 144.0,  // 2x H100 80GB
        }
    }

    /// Parse architecture from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "llama-3.1-8b" | "llama3.1-8b" | "llama3_1_8b" => Some(Self::Llama3_1_8B),
            "llama-3.1-70b" | "llama3.1-70b" | "llama3_1_70b" => Some(Self::Llama3_1_70B),
            "llama-3.3-70b" | "llama3.3-70b" | "llama3_3_70b" => Some(Self::Llama3_3_70B),
            "llama-3.1-405b" | "llama3.1-405b" | "llama3_1_405b" => Some(Self::Llama3_1_405B),
            "qwen-2.5-0.5b" | "qwen2.5-0.5b" | "qwen2_5_0_5b" => Some(Self::Qwen2_5_0_5B),
            "qwen-2.5-1.5b" | "qwen2.5-1.5b" | "qwen2_5_1_5b" => Some(Self::Qwen2_5_1_5B),
            "qwen-2.5-3b" | "qwen2.5-3b" | "qwen2_5_3b" => Some(Self::Qwen2_5_3B),
            "qwen-2.5-7b" | "qwen2.5-7b" | "qwen2_5_7b" => Some(Self::Qwen2_5_7B),
            "qwen-2.5-14b" | "qwen2.5-14b" | "qwen2_5_14b" => Some(Self::Qwen2_5_14B),
            "qwen-2.5-72b" | "qwen2.5-72b" | "qwen2_5_72b" => Some(Self::Qwen2_5_72B),
            _ => None,
        }
    }
}

/// Model configuration for v2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model architecture
    pub architecture: ModelArchitecture,
    /// Path to model weights
    pub weights_path: PathBuf,
    /// Use tensor parallelism across GPUs
    pub tensor_parallel: bool,
    /// Number of GPUs for tensor parallelism
    pub num_gpus: usize,
    /// Use flash attention
    pub use_flash_attention: bool,
    /// Use FP16 (half precision)
    pub use_fp16: bool,
    /// Use BF16 (brain float) - preferred on H100
    pub use_bf16: bool,
    /// KV cache size (in tokens)
    pub kv_cache_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::Llama3_1_70B,
            weights_path: PathBuf::from("llama-3.1-70b-weights"),
            tensor_parallel: true,
            num_gpus: 2, // 2x H100 for 70B
            use_flash_attention: true,
            use_fp16: false,
            use_bf16: true, // BF16 is better on H100
            kv_cache_size: 131072, // Full 128K context
        }
    }
}

/// Server configuration for v2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server bind address
    pub bind_addr: String,
    /// Server port
    pub port: u16,
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Session time-to-live
    #[serde(with = "humantime_serde")]
    pub session_ttl: Duration,
    /// Maximum requests per session
    pub max_requests_per_session: u32,
    /// Maximum prompt length (tokens)
    pub max_prompt_len: usize,
    /// Maximum generation length (tokens)
    pub max_generation_len: usize,
    /// CORS allowed origins
    pub cors_origins: Vec<String>,
    /// Enable rate limiting
    pub rate_limit_enabled: bool,
    /// Requests per minute per IP
    pub rate_limit_rpm: u32,
    /// Enable request logging
    pub enable_logging: bool,
    /// Prometheus metrics endpoint
    pub metrics_enabled: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0".to_string(),
            port: 8080,
            max_sessions: 10000,
            session_ttl: Duration::from_secs(3600), // 1 hour
            max_requests_per_session: 10000,
            max_prompt_len: 65536,  // 64K input tokens
            max_generation_len: 65536, // 64K output tokens
            cors_origins: vec!["*".to_string()],
            rate_limit_enabled: true,
            rate_limit_rpm: 60,
            enable_logging: true,
            metrics_enabled: true,
        }
    }
}

/// OT (Oblivious Transfer) configuration for v2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtConfig {
    /// Security parameter kappa (bits)
    pub kappa: usize,
    /// Use GPU-accelerated OT
    pub gpu_accelerated: bool,
    /// Batch size for OT operations
    pub batch_size: usize,
    /// Number of base OT instances
    pub num_base_ot: usize,
}

impl Default for OtConfig {
    fn default() -> Self {
        Self {
            kappa: 128,
            gpu_accelerated: true,
            batch_size: 1024,
            num_base_ot: 128,
        }
    }
}

/// Complete v2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2Config {
    /// Model configuration
    pub model: ModelConfig,
    /// Server configuration
    pub server: ServerConfig,
    /// OT configuration
    pub ot: OtConfig,
}

impl Default for V2Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            server: ServerConfig::default(),
            ot: OtConfig::default(),
        }
    }
}

impl V2Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        // Model config
        if let Ok(path) = std::env::var("SHARDLM_V2_MODEL_PATH") {
            config.model.weights_path = PathBuf::from(path);
        }
        if let Ok(arch) = std::env::var("SHARDLM_V2_MODEL_ARCH") {
            config.model.architecture = ModelArchitecture::from_str(&arch)
                .unwrap_or(ModelArchitecture::Llama3_1_70B);
        }
        if let Ok(gpus) = std::env::var("SHARDLM_V2_NUM_GPUS") {
            if let Ok(n) = gpus.parse() {
                config.model.num_gpus = n;
            }
        }

        // Server config
        if let Ok(addr) = std::env::var("SHARDLM_V2_BIND_ADDR") {
            config.server.bind_addr = addr;
        }
        if let Ok(port) = std::env::var("SHARDLM_V2_PORT") {
            if let Ok(p) = port.parse() {
                config.server.port = p;
            }
        }
        if let Ok(max_prompt) = std::env::var("SHARDLM_V2_MAX_PROMPT_LEN") {
            if let Ok(l) = max_prompt.parse() {
                config.server.max_prompt_len = l;
            }
        }

        config
    }

    /// Load configuration from a TOML file
    pub fn from_file(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })
    }

    /// Save configuration to a TOML file
    pub fn to_file(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
        })?;
        std::fs::write(path, content)
    }
}

// Serde helper for Duration
mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}
