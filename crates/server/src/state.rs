//! Application state

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use shardlm_fixed_point::DEFAULT_SCALE;
use shardlm_model::{EmbeddingTable, ModelWeights, SafetensorsLoader};
use shardlm_ot::OtSessionConfig;

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::session::{SessionStore, SharedSessionStore};

/// Server metadata for /info endpoint
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub version: String,
    pub git_commit: Option<String>,
    pub model_name: String,
    pub started_at: Instant,
}

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    /// Server configuration
    pub config: Arc<ServerConfig>,

    /// Session store
    pub sessions: SharedSessionStore,

    /// Embedding table (loaded from model)
    pub embeddings: Arc<Option<EmbeddingTable>>,

    /// Full model weights (for inference)
    pub weights: Arc<Option<ModelWeights>>,

    /// Server info
    pub info: Arc<ServerInfo>,

    /// Whether the server is ready to accept requests
    pub ready: Arc<std::sync::atomic::AtomicBool>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: ServerConfig) -> Self {
        let ot_config = OtSessionConfig {
            vocab_size: 32000, // TinyLlama default, updated when model loads
            hidden_dim: 2048,
            max_prompt_len: config.max_prompt_len,
            scale: DEFAULT_SCALE,
            ..Default::default()
        };

        let sessions = Arc::new(SessionStore::new(
            config.max_sessions,
            config.session_ttl,
            config.max_requests_per_session,
            ot_config,
        ));

        let info = ServerInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("GIT_COMMIT").map(String::from),
            model_name: "tinyllama".to_string(),
            started_at: Instant::now(),
        };

        Self {
            config: Arc::new(config),
            sessions,
            embeddings: Arc::new(None),
            weights: Arc::new(None),
            info: Arc::new(info),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Load model embeddings and weights
    pub async fn load_model(&mut self) -> Result<()> {
        let model_dir = self.config.model_dir.clone();

        tracing::info!("Loading model from {:?}...", model_dir);

        // Load in blocking task
        let (embeddings, weights) = tokio::task::spawn_blocking(move || {
            load_model_sync(&model_dir)
        })
        .await
        .map_err(|e| ServerError::Internal(format!("Task join error: {}", e)))??;

        tracing::info!(
            "Loaded {} embeddings of dim {}, vocab_size={}, lm_head ready",
            embeddings.vocab_size,
            embeddings.embed_dim,
            weights.config.vocab_size
        );

        // Update session config with actual vocab size and hidden dim from loaded model
        self.sessions.update_ot_config(
            weights.config.vocab_size as u32,
            weights.config.hidden_size as u16,
        );
        tracing::info!(
            "Updated OT config: vocab_size={}, hidden_dim={}",
            weights.config.vocab_size,
            weights.config.hidden_size
        );

        self.embeddings = Arc::new(Some(embeddings));
        self.weights = Arc::new(Some(weights));
        self.ready.store(true, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    /// Check if server is ready
    pub fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get embeddings (returns error if not loaded)
    pub fn get_embeddings(&self) -> Result<&EmbeddingTable> {
        self.embeddings.as_ref().as_ref().ok_or(ServerError::ModelNotLoaded)
    }

    /// Get model weights (returns error if not loaded)
    pub fn get_weights(&self) -> Result<&ModelWeights> {
        self.weights.as_ref().as_ref().ok_or(ServerError::ModelNotLoaded)
    }

    /// Get uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.info.started_at.elapsed()
    }
}

/// Synchronous model loading (embeddings + weights)
fn load_model_sync(model_dir: &Path) -> Result<(EmbeddingTable, ModelWeights)> {
    let loader = SafetensorsLoader::from_directory(model_dir, DEFAULT_SCALE)?;
    let embeddings = loader.load_embeddings()?;
    let weights = loader.load_model_weights()?;
    Ok((embeddings, weights))
}
