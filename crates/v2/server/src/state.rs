//! Application state

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use parking_lot::RwLock;
use tokio::sync::RwLock as AsyncRwLock;
use uuid::Uuid;

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_core::gpu::MultiGpuContext;
use shardlm_v2_model::{DistributedEngine, ShardedModelLoader, Tokenizer};

use crate::config::ServerConfig;
use crate::error::{Result, ServerError};
use crate::routes::secure_inference::SecureSession;
use crate::secure_weights::SecureModelWeights;
#[cfg(feature = "cuda")]
use crate::secure_weights::GpuSecureModelWeights;
#[cfg(feature = "cuda")]
use shardlm_v2_core::kernel::KernelContext;

/// Session data for tracking client requests
#[derive(Debug)]
pub struct Session {
    pub id: Uuid,
    pub created_at: Instant,
    pub last_active: RwLock<Instant>,
    pub request_count: std::sync::atomic::AtomicU64,
}

impl Session {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            created_at: now,
            last_active: RwLock::new(now),
            request_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn touch(&self) {
        *self.last_active.write() = Instant::now();
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn is_expired(&self, ttl_secs: u64) -> bool {
        self.last_active.read().elapsed().as_secs() > ttl_secs
    }
}

/// Server metadata
#[derive(Debug, Clone)]
pub struct ServerInfo {
    pub version: String,
    pub model_name: String,
    pub num_gpus: usize,
    pub started_at: Instant,
}

/// Application state shared across all handlers
#[derive(Clone)]
pub struct AppState {
    /// Server configuration
    pub config: Arc<ServerConfig>,
    /// Distributed inference engine (behind RwLock for interior mutability)
    engine: Arc<RwLock<Option<DistributedEngine>>>,
    /// Tokenizer (behind RwLock to allow initialization after construction)
    tokenizer: Arc<RwLock<Option<Tokenizer>>>,
    /// Secure model weights for privacy-preserving inference (CPU-resident)
    secure_weights: Arc<RwLock<Option<SecureModelWeights>>>,
    /// GPU-accelerated secure model weights (optional)
    #[cfg(feature = "cuda")]
    gpu_secure_weights: Arc<RwLock<Option<GpuSecureModelWeights>>>,
    /// GPU kernel contexts for cuBLAS operations (one per GPU)
    #[cfg(feature = "cuda")]
    gpu_kernel_contexts: Arc<RwLock<Vec<KernelContext>>>,
    /// Active sessions (for insecure inference)
    sessions: Arc<DashMap<Uuid, Session>>,
    /// Secure sessions (for privacy-preserving inference)
    pub secure_sessions: Arc<AsyncRwLock<HashMap<Uuid, SecureSession>>>,
    /// Server info
    pub info: Arc<ServerInfo>,
    /// Ready flag
    ready: Arc<std::sync::atomic::AtomicBool>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: ServerConfig) -> Self {
        let model_name = format!("{:?}", config.model_architecture);
        let info = ServerInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            model_name,
            num_gpus: config.num_gpus,
            started_at: Instant::now(),
        };

        Self {
            config: Arc::new(config),
            engine: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            secure_weights: Arc::new(RwLock::new(None)),
            #[cfg(feature = "cuda")]
            gpu_secure_weights: Arc::new(RwLock::new(None)),
            #[cfg(feature = "cuda")]
            gpu_kernel_contexts: Arc::new(RwLock::new(Vec::new())),
            sessions: Arc::new(DashMap::new()),
            secure_sessions: Arc::new(AsyncRwLock::new(HashMap::new())),
            info: Arc::new(info),
            ready: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Load model and prepare for inference
    pub async fn load_model(&mut self) -> Result<()> {
        let model_dir = self.config.model_dir.clone();
        let num_gpus = self.config.num_gpus;
        let architecture = self.config.model_architecture;

        tracing::info!("Loading {:?} from {:?} on {} GPUs...", architecture, model_dir, num_gpus);

        let max_seq_len = self.config.max_seq_len;

        // Load in blocking task
        #[cfg(feature = "cuda")]
        let (engine, tokenizer, secure_weights, gpu_secure_weights, kernel_contexts) =
            tokio::task::spawn_blocking(move || {
                load_model_sync(&model_dir, architecture, num_gpus, max_seq_len)
            })
            .await
            .map_err(|e| ServerError::Internal(format!("Task join error: {}", e)))??;

        #[cfg(not(feature = "cuda"))]
        let (engine, tokenizer, secure_weights) = tokio::task::spawn_blocking(move || {
            load_model_sync(&model_dir, architecture, num_gpus, max_seq_len)
        })
        .await
        .map_err(|e| ServerError::Internal(format!("Task join error: {}", e)))??;

        tracing::info!(
            "Model loaded: {} layers, {} hidden dim",
            engine.config().num_layers,
            engine.config().hidden_dim
        );

        // Update state
        *self.engine.write() = Some(engine);
        *self.tokenizer.write() = Some(tokenizer);
        *self.secure_weights.write() = Some(secure_weights);

        #[cfg(feature = "cuda")]
        {
            *self.gpu_secure_weights.write() = Some(gpu_secure_weights);
            *self.gpu_kernel_contexts.write() = kernel_contexts;
            tracing::info!("GPU-accelerated inference enabled");
        }

        self.ready.store(true, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    /// Check if server is ready
    pub fn is_ready(&self) -> bool {
        self.ready.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get the inference engine
    pub fn get_engine(&self) -> Result<parking_lot::RwLockWriteGuard<Option<DistributedEngine>>> {
        if !self.is_ready() {
            return Err(ServerError::ModelNotLoaded);
        }
        Ok(self.engine.write())
    }

    /// Get the tokenizer (returns a guard that can be dereferenced)
    pub fn get_tokenizer(&self) -> Result<parking_lot::RwLockReadGuard<Option<Tokenizer>>> {
        if !self.is_ready() {
            return Err(ServerError::ModelNotLoaded);
        }
        Ok(self.tokenizer.read())
    }

    /// Get the secure model weights for privacy-preserving inference
    pub fn get_secure_weights(&self) -> Result<parking_lot::RwLockReadGuard<Option<SecureModelWeights>>> {
        if !self.is_ready() {
            return Err(ServerError::ModelNotLoaded);
        }
        Ok(self.secure_weights.read())
    }

    /// Get the GPU-accelerated secure model weights
    #[cfg(feature = "cuda")]
    pub fn get_gpu_secure_weights(&self) -> Result<parking_lot::RwLockReadGuard<Option<GpuSecureModelWeights>>> {
        if !self.is_ready() {
            return Err(ServerError::ModelNotLoaded);
        }
        Ok(self.gpu_secure_weights.read())
    }

    /// Get the GPU kernel contexts for cuBLAS operations (one per GPU)
    #[cfg(feature = "cuda")]
    pub fn get_gpu_kernel_contexts(&self) -> Result<parking_lot::RwLockReadGuard<Vec<KernelContext>>> {
        if !self.is_ready() {
            return Err(ServerError::ModelNotLoaded);
        }
        Ok(self.gpu_kernel_contexts.read())
    }

    /// Get the GPU kernel context for cuBLAS operations (legacy - returns first context)
    /// Prefer get_gpu_kernel_contexts() for multi-GPU support
    #[cfg(feature = "cuda")]
    pub fn get_gpu_kernel_context(&self) -> Result<parking_lot::RwLockReadGuard<Vec<KernelContext>>> {
        self.get_gpu_kernel_contexts()
    }

    /// Set the GPU-accelerated secure weights (for testing/benchmarking)
    #[cfg(feature = "cuda")]
    pub fn set_gpu_secure_weights(&self, weights: GpuSecureModelWeights) {
        *self.gpu_secure_weights.write() = Some(weights);
    }

    /// Set the GPU kernel contexts (for testing/benchmarking)
    #[cfg(feature = "cuda")]
    pub fn set_gpu_kernel_contexts(&self, contexts: Vec<KernelContext>) {
        *self.gpu_kernel_contexts.write() = contexts;
    }

    /// Create a new session
    pub fn create_session(&self) -> Uuid {
        let session = Session::new();
        let id = session.id;
        self.sessions.insert(id, session);
        id
    }

    /// Get a session by ID
    pub fn get_session(&self, id: &Uuid) -> Result<dashmap::mapref::one::Ref<Uuid, Session>> {
        self.sessions
            .get(id)
            .ok_or_else(|| ServerError::SessionNotFound(id.to_string()))
    }

    /// Touch a session to keep it alive
    pub fn touch_session(&self, id: &Uuid) -> Result<()> {
        let session = self.get_session(id)?;
        if session.is_expired(self.config.session_ttl_secs) {
            return Err(ServerError::SessionExpired(id.to_string()));
        }
        session.touch();
        Ok(())
    }

    /// Clean up expired sessions
    pub fn cleanup_sessions(&self) {
        let ttl = self.config.session_ttl_secs;
        self.sessions.retain(|_, session| !session.is_expired(ttl));
    }

    /// Get uptime
    pub fn uptime(&self) -> std::time::Duration {
        self.info.started_at.elapsed()
    }

    /// Get session count
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Synchronous model loading
#[cfg(feature = "cuda")]
fn load_model_sync(
    model_dir: &std::path::Path,
    architecture: ModelArchitecture,
    num_gpus: usize,
    max_seq_len: usize,
) -> Result<(DistributedEngine, Tokenizer, SecureModelWeights, GpuSecureModelWeights, Vec<KernelContext>)> {
    // Load tokenizer
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

    // Create GPU context
    let gpu_ctx = MultiGpuContext::new(num_gpus)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // Load sharded weights
    let mut loader = ShardedModelLoader::new_int8(
        model_dir,
        architecture,
        num_gpus,
    )?;

    // Load from cache if available, otherwise quantize
    if !loader.has_cache() {
        tracing::info!("No cache found, loading and quantizing weights...");
        loader.load_all_files()?;
    }
    let weights = loader.load_all_weights()?;

    // Create engine
    let config = loader.config().clone();
    let mut engine = DistributedEngine::new(config, weights, gpu_ctx)?;

    // Initialize KV cache for fast incremental decoding
    #[cfg(feature = "cuda")]
    {
        tracing::info!("Initializing GPU KV cache with max_seq_len={}...", max_seq_len);
        engine.init_kv_cache(max_seq_len)
            .map_err(|e| ServerError::Internal(format!("Failed to initialize KV cache: {}", e)))?;
        tracing::info!("GPU KV cache initialized successfully");
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = max_seq_len; // Suppress unused warning
        tracing::warn!("CUDA not enabled, KV cache not initialized");
    }

    // Initialize secure weights for privacy-preserving inference
    #[cfg(feature = "cuda")]
    let secure_weights = {
        tracing::info!("Initializing secure weights for privacy-preserving inference...");

        // Create a new GPU context for secure weights initialization
        let gpu_ctx_for_secure = MultiGpuContext::new(num_gpus)
            .map_err(|e| ServerError::GpuError(e.to_string()))?;

        let devices: Vec<_> = (0..num_gpus).map(|i| gpu_ctx_for_secure.device(i)).collect();

        let model_config = engine.config();
        SecureModelWeights::from_gpu_weights(
            engine.weights(),
            &devices,
            model_config.hidden_dim,
            model_config.num_heads,
            model_config.num_kv_heads,
            model_config.head_dim,
            model_config.intermediate_dim,
            model_config.vocab_size,
            model_config.rope_theta,
        )?
    };

    // Initialize GPU-accelerated secure weights for cuBLAS operations
    tracing::info!("Initializing GPU-accelerated secure weights for cuBLAS...");
    let gpu_ctx_for_gpu_secure = MultiGpuContext::new(num_gpus)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;
    let gpu_devices: Vec<_> = (0..num_gpus).map(|i| gpu_ctx_for_gpu_secure.device(i)).collect();

    let model_config = engine.config();
    let gpu_secure_weights = GpuSecureModelWeights::from_gpu_weights(
        engine.weights(),
        &gpu_devices,
        model_config.hidden_dim,
        model_config.num_heads,
        model_config.num_kv_heads,
        model_config.head_dim,
        model_config.intermediate_dim,
        model_config.vocab_size,
        model_config.rope_theta,
    )?;
    tracing::info!("GPU-accelerated secure weights initialized");

    // Initialize kernel contexts for cuBLAS operations (one per GPU)
    tracing::info!("Initializing GPU kernel contexts for {} GPUs...", num_gpus);
    let mut kernel_contexts = Vec::with_capacity(num_gpus);
    for gpu_id in 0..num_gpus {
        let device = shardlm_v2_core::gpu::GpuDevice::new(gpu_id)
            .map_err(|e| ServerError::GpuError(format!("Failed to create device {}: {}", gpu_id, e)))?;
        let ctx = KernelContext::new(device)
            .map_err(|e| ServerError::GpuError(format!("Failed to create kernel context for GPU {}: {}", gpu_id, e)))?;
        kernel_contexts.push(ctx);
        tracing::info!("GPU kernel context {} initialized", gpu_id);
    }

    Ok((engine, tokenizer, secure_weights, gpu_secure_weights, kernel_contexts))
}

/// Synchronous model loading (non-CUDA fallback)
#[cfg(not(feature = "cuda"))]
fn load_model_sync(
    _model_dir: &std::path::Path,
    _architecture: ModelArchitecture,
    _num_gpus: usize,
    _max_seq_len: usize,
) -> Result<(DistributedEngine, Tokenizer, SecureModelWeights)> {
    panic!("CUDA feature required for model loading");
}
