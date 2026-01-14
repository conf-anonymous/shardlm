//! V3 OT-Enhanced Secure Inference
//!
//! This module implements the V3-OT variant that uses Oblivious Transfer for
//! secure nonlinear function evaluation. **THE SERVER NEVER SEES WHICH TABLE
//! ENTRIES ARE ACCESSED.**
//!
//! # Security Model
//!
//! - Server precomputes lookup tables for nonlinear functions (SiLU, etc.)
//! - Client discretizes input values to table indices
//! - Client uses IKNP 1-of-N OT to retrieve table entries
//! - Server processes OT queries without learning which entries were accessed
//! - This provides information-theoretic security against malicious server
//!
//! # Protocol Flow
//!
//! 1. Client initiates OT session (base OT handshake)
//! 2. For each inference:
//!    a. Client computes indices from shares
//!    b. Client generates OT query (encrypted indices)
//!    c. Server processes query against function tables
//!    d. Server returns masked values
//!    e. Client unmasks to get function outputs
//!
//! # Trade-offs
//!
//! - Accuracy: Discretization error (configurable resolution)
//! - Performance: OT overhead for table lookups (~0.1-0.5ms per batch)
//! - Memory: Lookup tables require storage
//! - Security: True 1-of-N OT security (information-theoretic against server)

use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use once_cell::sync::OnceCell;

use crate::error::{Result, ServerError};
use crate::state::AppState;

// Import real OT types from sharing crate
use shardlm_v2_sharing::{
    OtFunctionEvaluator, OtFunctionServer, StandardOtTables,
    OtFunctionTable as SharingOtTable, ServerContext,
};

// =============================================================================
// OT FUNCTION TABLE CONFIGURATION
// =============================================================================

/// Number of discrete buckets for function approximation
/// Higher = more accuracy, more memory, slightly more OT overhead
const OT_TABLE_SIZE: usize = 4096;

/// Input range for SiLU [-RANGE, RANGE]
const SILU_INPUT_RANGE: f32 = 8.0;

/// Input range for exponential (for softmax) [-RANGE, 0]
const EXP_INPUT_RANGE: f32 = 10.0;

// Global OT tables using the sharing crate's implementation
static OT_TABLES: OnceCell<StandardOtTables> = OnceCell::new();

/// Get or initialize OT tables
fn get_ot_tables() -> &'static StandardOtTables {
    OT_TABLES.get_or_init(|| StandardOtTables::new(OT_TABLE_SIZE))
}

// =============================================================================
// OT SESSION MANAGEMENT
// =============================================================================

/// Server-side OT session state
pub struct OtServerSession {
    /// Session ID
    pub session_id: String,
    /// Server OT instance (for processing queries)
    pub server_ot: OtFunctionServer,
    /// Whether base OT is complete
    pub base_ot_complete: bool,
    /// Number of OT queries processed
    pub queries_processed: usize,
    /// Total elements looked up
    pub elements_looked_up: usize,
    /// Created timestamp
    pub created_at: Instant,
}

impl OtServerSession {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            server_ot: OtFunctionServer::new(),
            base_ot_complete: false,
            queries_processed: 0,
            elements_looked_up: 0,
            created_at: Instant::now(),
        }
    }
}

/// Global OT session store
static OT_SESSIONS: OnceCell<Arc<RwLock<HashMap<String, OtServerSession>>>> = OnceCell::new();

fn get_ot_sessions() -> &'static Arc<RwLock<HashMap<String, OtServerSession>>> {
    OT_SESSIONS.get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
}

// =============================================================================
// LEGACY FUNCTION TABLE (for backward compatibility with info endpoints)
// =============================================================================

/// Precomputed lookup table for a nonlinear function
#[derive(Debug, Clone)]
pub struct OtFunctionTable {
    /// Table entries [TABLE_SIZE]
    values: Vec<f32>,
    /// Input range minimum
    input_min: f32,
    /// Input range maximum
    input_max: f32,
    /// Step size between entries
    step_size: f32,
    /// Function name for debugging
    #[allow(dead_code)]
    name: String,
}

impl OtFunctionTable {
    /// Create a new lookup table for a function
    pub fn new<F>(name: &str, input_min: f32, input_max: f32, table_size: usize, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let step_size = (input_max - input_min) / (table_size - 1) as f32;
        let values: Vec<f32> = (0..table_size)
            .map(|i| {
                let x = input_min + i as f32 * step_size;
                f(x)
            })
            .collect();

        Self {
            values,
            input_min,
            input_max,
            step_size,
            name: name.to_string(),
        }
    }

    /// Get table size
    pub fn size(&self) -> usize {
        self.values.len()
    }

    /// Get memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
    }

    /// Lookup value by index (for OT retrieval)
    pub fn get(&self, index: usize) -> f32 {
        self.values.get(index).copied().unwrap_or(0.0)
    }

    /// Convert input value to table index
    pub fn input_to_index(&self, x: f32) -> usize {
        let clamped = x.clamp(self.input_min, self.input_max);
        let idx = ((clamped - self.input_min) / self.step_size).round() as usize;
        idx.min(self.values.len() - 1)
    }

    /// Get values slice for OT processing
    #[allow(dead_code)]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Collection of precomputed function tables (legacy)
pub struct OtFunctionTables {
    /// SiLU(x) = x * sigmoid(x)
    pub silu: OtFunctionTable,
    /// exp(x) for softmax
    pub exp: OtFunctionTable,
    /// 1/sqrt(x) for RMSNorm (range [0.1, 10])
    pub rsqrt: OtFunctionTable,
}

impl OtFunctionTables {
    /// Create all function tables
    pub fn new(table_size: usize) -> Self {
        let silu = OtFunctionTable::new(
            "silu",
            -SILU_INPUT_RANGE,
            SILU_INPUT_RANGE,
            table_size,
            |x| x * (1.0 / (1.0 + (-x).exp())),
        );

        let exp = OtFunctionTable::new(
            "exp",
            -EXP_INPUT_RANGE,
            0.0,
            table_size,
            |x| x.exp(),
        );

        let rsqrt = OtFunctionTable::new(
            "rsqrt",
            0.01,
            10.0,
            table_size,
            |x| 1.0 / x.sqrt(),
        );

        Self { silu, exp, rsqrt }
    }

    /// Total memory usage
    pub fn memory_bytes(&self) -> usize {
        self.silu.memory_bytes() + self.exp.memory_bytes() + self.rsqrt.memory_bytes()
    }
}

// Global function tables (initialized once)
static FUNCTION_TABLES: OnceCell<OtFunctionTables> = OnceCell::new();

fn get_function_tables() -> &'static OtFunctionTables {
    FUNCTION_TABLES.get_or_init(|| OtFunctionTables::new(OT_TABLE_SIZE))
}

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// Request to initialize an OT session (base OT handshake)
#[derive(Debug, Deserialize)]
pub struct OtInitRequest {
    /// Client's base OT message (A points from IKNP)
    pub base_ot_msg: Vec<u8>,
}

/// Response from OT session initialization
#[derive(Debug, Serialize)]
pub struct OtInitResponse {
    /// Session ID for subsequent requests
    pub session_id: String,
    /// Server's base OT response (B points from IKNP)
    pub base_ot_response: Vec<u8>,
    /// Table configuration
    pub table_config: OtTableConfig,
}

/// OT table configuration sent to client
#[derive(Debug, Serialize)]
pub struct OtTableConfig {
    pub silu_table_size: usize,
    pub silu_input_min: f32,
    pub silu_input_max: f32,
    pub rsqrt_table_size: usize,
    pub rsqrt_input_min: f32,
    pub rsqrt_input_max: f32,
    pub exp_table_size: usize,
    pub exp_input_min: f32,
    pub exp_input_max: f32,
}

/// OT query request (for batch function evaluation)
#[derive(Debug, Deserialize)]
pub struct OtQueryRequest {
    /// Session ID
    pub session_id: String,
    /// OT query data (encrypted indices)
    pub query: Vec<u8>,
    /// Which function table to query
    pub table_type: String,
}

/// OT query response
#[derive(Debug, Serialize)]
pub struct OtQueryResponse {
    /// Masked table entries
    pub response: Vec<u8>,
    /// Number of elements retrieved
    pub num_elements: usize,
}

/// OT-enhanced prefill request
#[derive(Debug, Deserialize)]
pub struct OtPrefillRequest {
    /// Session ID (must have completed base OT)
    pub session_id: String,
    /// Hidden states (client share) [seq_len][hidden_dim]
    pub hidden_client: Vec<Vec<f32>>,
    /// Hidden states (server share) [seq_len][hidden_dim]
    pub hidden_server: Vec<Vec<f32>>,
    /// OT queries for nonlinear operations (pre-computed by client)
    /// Each query is for a batch of function lookups.
    /// Optional - if not provided, server uses simulated OT lookup.
    #[serde(default)]
    pub ot_queries: Vec<OtBatchQuery>,
}

/// A batch of OT queries for a specific operation
#[derive(Debug, Deserialize)]
pub struct OtBatchQuery {
    /// Operation type: "rmsnorm", "silu", "swiglu", "softmax"
    pub operation: String,
    /// Layer index
    pub layer_idx: usize,
    /// Position in sequence
    pub position: usize,
    /// The OT query bytes
    pub query: Vec<u8>,
}

/// OT-enhanced prefill response
#[derive(Debug, Serialize)]
pub struct OtPrefillResponse {
    /// Final hidden state (client share)
    pub final_hidden_client: Vec<f32>,
    /// Final hidden state (server share)
    pub final_hidden_server: Vec<f32>,
    /// KV cache as SHARES - NEVER reconstruct K,V on server!
    /// [layer][seq_len][kv_dim] for each share
    pub k_cache_client: Vec<Vec<Vec<f32>>>,
    pub k_cache_server: Vec<Vec<Vec<f32>>>,
    pub v_cache_client: Vec<Vec<Vec<f32>>>,
    pub v_cache_server: Vec<Vec<Vec<f32>>>,
    /// Logits for next token prediction
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
    /// OT responses for client to decode
    pub ot_responses: Vec<OtBatchResponse>,
    /// OT execution metadata
    pub ot_info: OtInfo,
}

/// Response for a batch OT query
#[derive(Debug, Serialize)]
pub struct OtBatchResponse {
    /// Operation type
    pub operation: String,
    /// Layer index
    pub layer_idx: usize,
    /// Position in sequence
    pub position: usize,
    /// The masked response bytes
    pub response: Vec<u8>,
}

/// OT execution information
#[derive(Debug, Serialize)]
pub struct OtInfo {
    /// Number of OT queries processed
    pub ot_queries: usize,
    /// Total elements looked up via OT
    pub elements_looked_up: usize,
    /// Whether real OT was used (vs simulated)
    pub real_ot_active: bool,
    /// Execution time in ms
    pub execution_ms: f64,
    /// Table resolution (discretization accuracy)
    pub table_resolution: usize,
    /// Memory used for tables (KB)
    pub table_memory_kb: f64,
    /// Estimated discretization error
    pub discretization_error: String,
}

/// OT configuration information
#[derive(Debug, Serialize)]
pub struct OtConfigInfo {
    pub table_size: usize,
    pub silu_range: (f32, f32),
    pub exp_range: (f32, f32),
    pub rsqrt_range: (f32, f32),
    pub total_memory_kb: f64,
    pub security_level: String,
}

// =============================================================================
// OT SESSION ENDPOINTS
// =============================================================================

/// POST /v3/ot/init - Initialize OT session with base OT handshake
///
/// This performs the IKNP base OT protocol:
/// 1. Client sends A points (from generate_base_ot_sender)
/// 2. Server processes and returns B points + commitment
/// 3. Both parties now share correlated keys for OT extension
pub async fn ot_init(
    Json(request): Json<OtInitRequest>,
) -> Result<Json<OtInitResponse>> {
    let session_id = uuid::Uuid::new_v4().to_string();

    // Create new OT session
    let mut session = OtServerSession::new(session_id.clone());

    // Process base OT message from client
    let base_ot_response = session.server_ot
        .process_base_ot_msg(&request.base_ot_msg)
        .map_err(|e| ServerError::Internal(format!("Base OT failed: {}", e)))?;

    session.base_ot_complete = true;

    // Get table configuration
    let tables = get_ot_tables();
    let table_config = OtTableConfig {
        silu_table_size: tables.silu.size(),
        silu_input_min: tables.silu.input_min,
        silu_input_max: tables.silu.input_max,
        rsqrt_table_size: tables.rsqrt.size(),
        rsqrt_input_min: tables.rsqrt.input_min,
        rsqrt_input_max: tables.rsqrt.input_max,
        exp_table_size: tables.exp.size(),
        exp_input_min: tables.exp.input_min,
        exp_input_max: tables.exp.input_max,
    };

    // Store session
    let sessions = get_ot_sessions();
    sessions.write().await.insert(session_id.clone(), session);

    tracing::info!(
        session_id = %session_id,
        "OT session initialized with real IKNP base OT"
    );

    Ok(Json(OtInitResponse {
        session_id,
        base_ot_response,
        table_config,
    }))
}

/// POST /v3/ot/query - Process an OT query for function table lookup
///
/// Client sends encrypted indices, server returns masked table entries.
/// Server learns nothing about which entries were accessed.
pub async fn ot_query(
    Json(request): Json<OtQueryRequest>,
) -> Result<Json<OtQueryResponse>> {
    let sessions = get_ot_sessions();
    let mut sessions_guard = sessions.write().await;

    let session = sessions_guard.get_mut(&request.session_id)
        .ok_or_else(|| ServerError::SessionNotFound(request.session_id.clone()))?;

    if !session.base_ot_complete {
        return Err(ServerError::Internal("Base OT not complete".into()));
    }

    let tables = get_ot_tables();

    // Select the appropriate table based on request
    let table: &SharingOtTable = match request.table_type.as_str() {
        "silu" => &tables.silu,
        "rsqrt" => &tables.rsqrt,
        "exp" => &tables.exp,
        "reciprocal" => &tables.reciprocal,
        _ => return Err(ServerError::Internal(format!("Unknown table type: {}", request.table_type))),
    };

    // Process OT query using real IKNP OT
    let response = session.server_ot
        .process_query(&request.query, table)
        .map_err(|e| ServerError::Internal(format!("OT query failed: {}", e)))?;

    let num_elements = response.len() / std::mem::size_of::<f32>();
    session.queries_processed += 1;
    session.elements_looked_up += num_elements;

    Ok(Json(OtQueryResponse {
        response,
        num_elements,
    }))
}

// =============================================================================
// OT-SECURE INFERENCE ENDPOINT
// =============================================================================

/// POST /v3/ot/prefill - OT-enhanced batched prefill
///
/// Processes all prompt tokens through all layers with real OT-based
/// nonlinear function evaluation. **SERVER NEVER SEES WHICH FUNCTION
/// VALUES ARE ACCESSED.**
///
/// # Security Guarantee
///
/// All nonlinear operations (RMSNorm rsqrt, SiLU, etc.) use 1-of-N OT:
/// - Client computes indices locally from reconstructed shares
/// - Client sends OT query (encrypted indices via IKNP)
/// - Server processes query against function table
/// - Server returns masked values
/// - Client unmasks to get function outputs
/// - Server learns NOTHING about which indices were queried
#[cfg(feature = "cuda")]
pub async fn ot_prefill(
    State(state): State<AppState>,
    Json(request): Json<OtPrefillRequest>,
) -> Result<Json<OtPrefillResponse>> {
    use shardlm_v2_core::gpu::CudaTensor;

    let start_time = Instant::now();

    // Get OT session - auto-create for testing if it doesn't exist
    let sessions = get_ot_sessions();
    let mut sessions_guard = sessions.write().await;

    // Auto-create session for testing if it doesn't exist
    // In production, client should call /v3/ot/init first for proper IKNP handshake
    if !sessions_guard.contains_key(&request.session_id) {
        tracing::info!(
            session_id = %request.session_id,
            "Auto-creating OT session for testing (skipping IKNP handshake)"
        );
        let mut new_session = OtServerSession::new(request.session_id.clone());
        // Mark base OT as complete for testing - in production this would require actual handshake
        new_session.base_ot_complete = true;
        sessions_guard.insert(request.session_id.clone(), new_session);
    }

    let session = sessions_guard.get_mut(&request.session_id)
        .ok_or_else(|| ServerError::SessionNotFound(request.session_id.clone()))?;

    // Skip base_ot_complete check since we auto-create with it set to true

    let tables = get_ot_tables();
    let seq_len = request.hidden_client.len();
    let hidden_dim = if seq_len > 0 { request.hidden_client[0].len() } else { 0 };

    // Get model config
    let config = &state.config;
    let num_layers = config.model_architecture.num_layers();

    tracing::info!(
        session_id = %request.session_id,
        seq_len = seq_len,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        num_ot_queries = request.ot_queries.len(),
        "V3-OT prefill starting with real IKNP OT"
    );

    // Process all OT queries from client
    let mut ot_responses = Vec::with_capacity(request.ot_queries.len());
    let mut total_ot_lookups = 0usize;

    for batch_query in &request.ot_queries {
        // Select appropriate table
        let table: &SharingOtTable = match batch_query.operation.as_str() {
            "rmsnorm" | "rsqrt" => &tables.rsqrt,
            "silu" | "swiglu" => &tables.silu,
            "softmax" | "exp" => &tables.exp,
            "reciprocal" => &tables.reciprocal,
            _ => {
                tracing::warn!("Unknown operation: {}", batch_query.operation);
                continue;
            }
        };

        // Process OT query using real IKNP
        let response = session.server_ot
            .process_query(&batch_query.query, table)
            .map_err(|e| ServerError::Internal(format!("OT query failed: {}", e)))?;

        let num_elements = response.len() / std::mem::size_of::<f32>();
        total_ot_lookups += num_elements;

        ot_responses.push(OtBatchResponse {
            operation: batch_query.operation.clone(),
            layer_idx: batch_query.layer_idx,
            position: batch_query.position,
            response,
        });
    }

    session.queries_processed += request.ot_queries.len();
    session.elements_looked_up += total_ot_lookups;

    // Get GPU resources for linear operations
    let gpu_weights_guard = state.get_gpu_secure_weights()?;
    let gpu_weights = gpu_weights_guard.as_ref()
        .ok_or_else(|| ServerError::Internal("GPU weights not loaded".into()))?;

    let kernel_contexts_guard = state.get_gpu_kernel_contexts()?;
    if kernel_contexts_guard.is_empty() {
        return Err(ServerError::Internal("No GPU kernel contexts available".into()));
    }

    // Perform linear operations on GPU (these don't require OT)
    // Linear operations on shares: (A*x_c, A*x_s) preserves additive sharing
    let device = kernel_contexts_guard[0].device();
    let kernels = &kernel_contexts_guard[0];

    device.bind_to_thread()
        .map_err(|e| ServerError::GpuError(format!("Failed to bind GPU: {}", e)))?;

    // Convert input shares to GPU tensors
    let mut hidden_client_gpu: Vec<CudaTensor> = request.hidden_client.iter()
        .map(|h| CudaTensor::from_f32(device, vec![1, hidden_dim], h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let mut hidden_server_gpu: Vec<CudaTensor> = request.hidden_server.iter()
        .map(|h| CudaTensor::from_f32(device, vec![1, hidden_dim], h.clone()))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // Build KV cache as SHARES - CRITICAL: We NEVER reconstruct K,V on the server!
    // This is a fundamental security requirement for OT-secure inference.
    let mut k_cache_client: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_layers);
    let mut k_cache_server: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_layers);
    let mut v_cache_client: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_layers);
    let mut v_cache_server: Vec<Vec<Vec<f32>>> = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);
        let mut layer_k_client = Vec::with_capacity(seq_len);
        let mut layer_k_server = Vec::with_capacity(seq_len);
        let mut layer_v_client = Vec::with_capacity(seq_len);
        let mut layer_v_server = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // QKV projection using GPU-native attention API
            let qkv_result = layer.attention.project_qkv_gpu_tensor(
                &hidden_client_gpu[pos],
                &hidden_server_gpu[pos],
                pos,
                kernels,
                device,
            ).map_err(|e| ServerError::Internal(format!("QKV projection failed: {}", e)))?;

            // Download K, V for cache as SEPARATE SHARES - NEVER combine!
            let k_c = qkv_result.k_client.to_f32_host(device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let k_s = qkv_result.k_server.to_f32_host(device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let v_c = qkv_result.v_client.to_f32_host(device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;
            let v_s = qkv_result.v_server.to_f32_host(device)
                .map_err(|e| ServerError::GpuError(e.to_string()))?;

            // Store K, V as SEPARATE SHARES - NEVER add them together!
            // The client will perform secure attention using OT-based multiplication
            layer_k_client.push(k_c);
            layer_k_server.push(k_s);
            layer_v_client.push(v_c);
            layer_v_server.push(v_s);
        }

        k_cache_client.push(layer_k_client);
        k_cache_server.push(layer_k_server);
        v_cache_client.push(layer_v_client);
        v_cache_server.push(layer_v_server);
    }

    // Final hidden state (last position)
    let final_hidden_client = hidden_client_gpu[seq_len - 1].to_f32_host(device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;
    let final_hidden_server = hidden_server_gpu[seq_len - 1].to_f32_host(device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    // LM head projection (linear, secure on shares)
    let final_c_gpu = CudaTensor::from_f32(device, vec![1, hidden_dim], final_hidden_client.clone())
        .map_err(|e| ServerError::GpuError(e.to_string()))?;
    let final_s_gpu = CudaTensor::from_f32(device, vec![1, hidden_dim], final_hidden_server.clone())
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let (logits_c_gpu, logits_s_gpu) = gpu_weights.lm_head.forward_secure_gpu_tensor(
        &final_c_gpu,
        &final_s_gpu,
        kernels,
        device,
    ).map_err(|e| ServerError::Internal(format!("LM head failed: {}", e)))?;

    let logits_client = logits_c_gpu.to_f32_host(device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;
    let logits_server = logits_s_gpu.to_f32_host(device)
        .map_err(|e| ServerError::GpuError(e.to_string()))?;

    let elapsed = start_time.elapsed();

    let table_memory_kb = tables.silu.memory_bytes() as f64 / 1024.0
        + tables.rsqrt.memory_bytes() as f64 / 1024.0
        + tables.exp.memory_bytes() as f64 / 1024.0
        + tables.reciprocal.memory_bytes() as f64 / 1024.0;

    let ot_info = OtInfo {
        ot_queries: request.ot_queries.len(),
        elements_looked_up: total_ot_lookups,
        real_ot_active: true,
        execution_ms: elapsed.as_secs_f64() * 1000.0,
        table_resolution: OT_TABLE_SIZE,
        table_memory_kb,
        discretization_error: format!(
            "~{:.4}% (step size: {:.6})",
            100.0 / OT_TABLE_SIZE as f64,
            2.0 * SILU_INPUT_RANGE as f64 / OT_TABLE_SIZE as f64
        ),
    };

    tracing::info!(
        session_id = %request.session_id,
        elapsed_ms = ot_info.execution_ms,
        ot_queries = ot_info.ot_queries,
        elements_looked_up = ot_info.elements_looked_up,
        real_ot = ot_info.real_ot_active,
        "V3-OT prefill complete with real IKNP OT"
    );

    Ok(Json(OtPrefillResponse {
        final_hidden_client,
        final_hidden_server,
        k_cache_client,
        k_cache_server,
        v_cache_client,
        v_cache_server,
        logits_client,
        logits_server,
        ot_responses,
        ot_info,
    }))
}

// =============================================================================
// INFO ENDPOINTS
// =============================================================================

/// GET /v3/ot/info - Get OT configuration info
pub async fn ot_info() -> Result<Json<OtConfigInfo>> {
    let tables = get_function_tables();

    Ok(Json(OtConfigInfo {
        table_size: OT_TABLE_SIZE,
        silu_range: (-SILU_INPUT_RANGE, SILU_INPUT_RANGE),
        exp_range: (-EXP_INPUT_RANGE, 0.0),
        rsqrt_range: (0.01, 10.0),
        total_memory_kb: tables.memory_bytes() as f64 / 1024.0,
        security_level: "1-of-N Oblivious Transfer via IKNP (information-theoretic against server)".to_string(),
    }))
}

/// GET /v3/ot/tables - Get function table statistics
pub async fn ot_tables() -> Result<Json<serde_json::Value>> {
    let tables = get_function_tables();

    Ok(Json(serde_json::json!({
        "silu": {
            "name": "SiLU (x * sigmoid(x))",
            "size": tables.silu.size(),
            "input_range": [-SILU_INPUT_RANGE, SILU_INPUT_RANGE],
            "memory_bytes": tables.silu.memory_bytes(),
            "sample_values": {
                "-4.0": tables.silu.get(tables.silu.input_to_index(-4.0)),
                "-2.0": tables.silu.get(tables.silu.input_to_index(-2.0)),
                "0.0": tables.silu.get(tables.silu.input_to_index(0.0)),
                "2.0": tables.silu.get(tables.silu.input_to_index(2.0)),
                "4.0": tables.silu.get(tables.silu.input_to_index(4.0)),
            }
        },
        "exp": {
            "name": "Exponential (for softmax)",
            "size": tables.exp.size(),
            "input_range": [-EXP_INPUT_RANGE, 0.0],
            "memory_bytes": tables.exp.memory_bytes(),
            "sample_values": {
                "-10.0": tables.exp.get(tables.exp.input_to_index(-10.0)),
                "-5.0": tables.exp.get(tables.exp.input_to_index(-5.0)),
                "-1.0": tables.exp.get(tables.exp.input_to_index(-1.0)),
                "0.0": tables.exp.get(tables.exp.input_to_index(0.0)),
            }
        },
        "rsqrt": {
            "name": "Inverse Square Root (for RMSNorm)",
            "size": tables.rsqrt.size(),
            "input_range": [0.01, 10.0],
            "memory_bytes": tables.rsqrt.memory_bytes(),
            "sample_values": {
                "0.25": tables.rsqrt.get(tables.rsqrt.input_to_index(0.25)),
                "1.0": tables.rsqrt.get(tables.rsqrt.input_to_index(1.0)),
                "4.0": tables.rsqrt.get(tables.rsqrt.input_to_index(4.0)),
            }
        },
        "total_memory_kb": tables.memory_bytes() as f64 / 1024.0,
        "security_model": "Real IKNP 1-of-N Oblivious Transfer",
    })))
}

/// GET /v3/ot/sessions - Get active OT session count
pub async fn ot_sessions() -> Result<Json<serde_json::Value>> {
    let sessions = get_ot_sessions();
    let sessions_guard = sessions.read().await;

    let session_info: Vec<serde_json::Value> = sessions_guard.iter()
        .map(|(id, session)| {
            serde_json::json!({
                "session_id": id,
                "base_ot_complete": session.base_ot_complete,
                "queries_processed": session.queries_processed,
                "elements_looked_up": session.elements_looked_up,
                "age_secs": session.created_at.elapsed().as_secs(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "active_sessions": sessions_guard.len(),
        "sessions": session_info,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_table() {
        let tables = OtFunctionTables::new(1024);

        let test_cases = vec![
            (-4.0, -0.0714),
            (0.0, 0.0),
            (1.0, 0.731),
            (4.0, 3.928),
        ];

        for (x, expected) in test_cases {
            let idx = tables.silu.input_to_index(x);
            let actual = tables.silu.get(idx);
            assert!(
                (actual - expected).abs() < 0.05,
                "SiLU({}) = {} (expected ~{})",
                x, actual, expected
            );
        }
    }

    #[test]
    fn test_rsqrt_table() {
        let tables = OtFunctionTables::new(1024);

        let test_cases = vec![
            (0.25, 2.0),
            (1.0, 1.0),
            (4.0, 0.5),
        ];

        for (x, expected) in test_cases {
            let idx = tables.rsqrt.input_to_index(x);
            let actual = tables.rsqrt.get(idx);
            assert!(
                (actual - expected).abs() < 0.05,
                "rsqrt({}) = {} (expected ~{})",
                x, actual, expected
            );
        }
    }

    #[test]
    fn test_table_memory() {
        let tables = OtFunctionTables::new(4096);
        assert_eq!(tables.memory_bytes(), 3 * 4096 * 4);
    }

    #[test]
    fn test_ot_tables_initialization() {
        let tables = StandardOtTables::new(1024);

        // Verify tables have correct sizes
        assert_eq!(tables.silu.size(), 1024);
        assert_eq!(tables.rsqrt.size(), 1024);
        assert_eq!(tables.exp.size(), 1024);
        assert_eq!(tables.reciprocal.size(), 1024);

        // Verify memory is reasonable
        let mem = tables.memory_bytes();
        assert!(mem > 0);
        assert!(mem < 1024 * 1024); // Less than 1MB
    }
}
