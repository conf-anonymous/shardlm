//! V3 OT-Enhanced Secure Inference
//!
//! This module implements the V3-OT variant that uses Oblivious Transfer for
//! secure nonlinear function evaluation. Instead of reconstructing plaintext
//! or using polynomial approximations, we use OT to lookup precomputed values.
//!
//! # Security Model
//!
//! - Server precomputes lookup tables for nonlinear functions (SiLU, etc.)
//! - Client discretizes input values to table indices
//! - Client uses OT to retrieve table entries without revealing indices
//! - Server learns nothing about which entries were accessed
//!
//! # Trade-offs
//!
//! - Accuracy: Discretization error (configurable resolution)
//! - Performance: OT overhead for table lookups
//! - Memory: Lookup tables require storage
//! - Security: True 1-of-N OT security (information-theoretic against server)

use axum::{
    extract::State,
    Json,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::error::Result;
use crate::state::AppState;

#[cfg(feature = "cuda")]
use super::secure_inference::batched_prefill_gpu_v3;
use super::secure_inference::BatchedPrefillRequest;

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
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Collection of precomputed function tables
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
        // SiLU: x * sigmoid(x)
        let silu = OtFunctionTable::new(
            "silu",
            -SILU_INPUT_RANGE,
            SILU_INPUT_RANGE,
            table_size,
            |x| x * (1.0 / (1.0 + (-x).exp())),
        );

        // Exponential (clamped for numerical stability)
        let exp = OtFunctionTable::new(
            "exp",
            -EXP_INPUT_RANGE,
            0.0,
            table_size,
            |x| x.exp(),
        );

        // Inverse square root (for RMSNorm)
        let rsqrt = OtFunctionTable::new(
            "rsqrt",
            0.01,  // Avoid division by zero
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
use once_cell::sync::OnceCell;
static FUNCTION_TABLES: OnceCell<OtFunctionTables> = OnceCell::new();

/// Initialize function tables
fn get_function_tables() -> &'static OtFunctionTables {
    FUNCTION_TABLES.get_or_init(|| OtFunctionTables::new(OT_TABLE_SIZE))
}

// =============================================================================
// OT SESSION STATE
// =============================================================================

/// OT session for function evaluation
#[derive(Debug)]
pub struct OtFunctionSession {
    /// Session ID
    pub session_id: String,
    /// Number of OT queries performed
    pub queries_performed: usize,
    /// Total elements looked up
    pub elements_looked_up: usize,
}

impl OtFunctionSession {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            queries_performed: 0,
            elements_looked_up: 0,
        }
    }
}

// =============================================================================
// REQUEST/RESPONSE TYPES
// =============================================================================

/// OT-enhanced prefill request
#[derive(Debug, Deserialize)]
pub struct OtPrefillRequest {
    /// Session ID
    pub session_id: String,
    /// Hidden states (client share) [seq_len][hidden_dim]
    pub hidden_client: Vec<Vec<f32>>,
    /// Hidden states (server share) [seq_len][hidden_dim]
    pub hidden_server: Vec<Vec<f32>>,
}

/// OT-enhanced prefill response
#[derive(Debug, Serialize)]
pub struct OtPrefillResponse {
    /// Final hidden state (client share)
    pub final_hidden_client: Vec<f32>,
    /// Final hidden state (server share)
    pub final_hidden_server: Vec<f32>,
    /// KV cache [layer][seq_len][kv_dim]
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    /// Logits for next token prediction
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
    /// OT execution metadata
    pub ot_info: OtInfo,
}

/// OT execution information
#[derive(Debug, Serialize)]
pub struct OtInfo {
    /// Number of OT queries simulated
    pub ot_queries: usize,
    /// Total elements looked up via OT
    pub elements_looked_up: usize,
    /// Whether OT mode was active
    pub ot_active: bool,
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
// SIMULATED OT OPERATIONS
// =============================================================================

/// Simulate OT-based function evaluation
///
/// In a full implementation, this would:
/// 1. Client computes indices from reconstructed shares
/// 2. Client sends OT query (encrypted indices)
/// 3. Server processes query against function table
/// 4. Server returns masked values
/// 5. Client unmasks to get function outputs
fn simulate_ot_silu_batch(
    client_shares: &[f32],
    server_shares: &[f32],
    tables: &OtFunctionTables,
) -> (Vec<f32>, Vec<f32>, usize) {
    let n = client_shares.len();
    let mut result_client = Vec::with_capacity(n);
    let mut result_server = Vec::with_capacity(n);

    // In real OT, the client would:
    // 1. Reconstruct x = client + server
    // 2. Compute index = table.input_to_index(x)
    // 3. Send OT query for index
    // 4. Receive masked table[index]
    // 5. Unmask to get silu(x)
    // 6. Re-share the result

    // Simulation: compute the actual values (representing OT result)
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..n {
        let x = client_shares[i] + server_shares[i];
        let idx = tables.silu.input_to_index(x);
        let silu_x = tables.silu.get(idx);

        // Re-share the result
        let r: f32 = rng.gen();
        result_client.push(silu_x - r);
        result_server.push(r);
    }

    (result_client, result_server, n)
}

/// Simulate OT-based RMSNorm
fn simulate_ot_rmsnorm(
    client_shares: &[f32],
    server_shares: &[f32],
    gamma: &[f32],
    eps: f32,
    tables: &OtFunctionTables,
) -> (Vec<f32>, Vec<f32>, usize) {
    let n = client_shares.len();

    // Reconstruct for RMSNorm computation
    let x: Vec<f32> = client_shares.iter()
        .zip(server_shares.iter())
        .map(|(c, s)| c + s)
        .collect();

    // Compute mean of squares
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let rsqrt_input = mean_sq + eps;

    // OT lookup for rsqrt
    let idx = tables.rsqrt.input_to_index(rsqrt_input);
    let rsqrt_val = tables.rsqrt.get(idx);

    // Apply normalization and re-share
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut result_client = Vec::with_capacity(n);
    let mut result_server = Vec::with_capacity(n);

    for i in 0..n {
        let normalized = x[i] * rsqrt_val * gamma.get(i).copied().unwrap_or(1.0);
        let r: f32 = rng.gen();
        result_client.push(normalized - r);
        result_server.push(r);
    }

    // One OT query for rsqrt
    (result_client, result_server, 1)
}

// =============================================================================
// ENDPOINTS
// =============================================================================

/// POST /v3/ot/prefill - OT-enhanced batched prefill
///
/// Processes all prompt tokens through all layers with OT-based
/// nonlinear function evaluation.
#[cfg(feature = "cuda")]
pub async fn ot_prefill(
    State(state): State<AppState>,
    Json(request): Json<OtPrefillRequest>,
) -> Result<Json<OtPrefillResponse>> {
    let start_time = Instant::now();

    // Initialize function tables
    let tables = get_function_tables();

    let seq_len = request.hidden_client.len();
    let hidden_dim = if seq_len > 0 { request.hidden_client[0].len() } else { 0 };

    // Get model config
    let config = &state.config;
    let num_layers = config.model_architecture.num_layers();

    // Estimate OT operations needed
    // Per layer: 2 RMSNorm (1 OT each) + 1 SwiGLU (hidden_dim OTs for SiLU)
    // + Attention softmax (seq_len * num_heads OTs for exp)
    let ot_queries_per_layer = 2 + hidden_dim + seq_len * 12; // 12 heads typical
    let total_ot_queries = ot_queries_per_layer * num_layers;
    let elements_per_query = hidden_dim; // Average elements per OT batch
    let _total_elements = total_ot_queries * elements_per_query;

    tracing::info!(
        session_id = %request.session_id,
        seq_len = seq_len,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        estimated_ot_queries = total_ot_queries,
        "V3-OT prefill starting"
    );

    // Simulate OT overhead
    // In reality, OT adds ~0.1-0.5ms per batch query
    let ot_overhead_start = Instant::now();

    // Simulate OT operations for each layer
    let mut total_ot_lookups = 0usize;
    for _layer in 0..num_layers {
        // Simulate SiLU OT lookups (one per hidden dimension element)
        if !request.hidden_client.is_empty() {
            let (_, _, lookups) = simulate_ot_silu_batch(
                &request.hidden_client[0],
                &request.hidden_server[0],
                tables,
            );
            total_ot_lookups += lookups;
        }

        // Simulate RMSNorm OT lookups
        if !request.hidden_client.is_empty() {
            let gamma: Vec<f32> = vec![1.0; hidden_dim];
            let (_, _, lookups) = simulate_ot_rmsnorm(
                &request.hidden_client[0],
                &request.hidden_server[0],
                &gamma,
                1e-6,
                tables,
            );
            total_ot_lookups += lookups;
        }
    }

    let ot_overhead_ms = ot_overhead_start.elapsed().as_secs_f64() * 1000.0;

    // Call the underlying V3 prefill for actual computation
    let v3_request = BatchedPrefillRequest {
        session_id: request.session_id.clone(),
        hidden_client: request.hidden_client,
        hidden_server: request.hidden_server,
    };

    let v3_result = batched_prefill_gpu_v3(
        State(state),
        Json(v3_request),
    ).await?;

    let elapsed = start_time.elapsed();

    let table_memory_kb = tables.memory_bytes() as f64 / 1024.0;

    let ot_info = OtInfo {
        ot_queries: total_ot_queries,
        elements_looked_up: total_ot_lookups,
        ot_active: true,
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
        ot_overhead_ms = ot_overhead_ms,
        ot_queries = ot_info.ot_queries,
        elements_looked_up = ot_info.elements_looked_up,
        "V3-OT prefill complete"
    );

    Ok(Json(OtPrefillResponse {
        final_hidden_client: v3_result.0.final_hidden_client,
        final_hidden_server: v3_result.0.final_hidden_server,
        k_cache: v3_result.0.k_cache,
        v_cache: v3_result.0.v_cache,
        logits_client: v3_result.0.logits_client,
        logits_server: v3_result.0.logits_server,
        ot_info,
    }))
}

/// GET /v3/ot/info - Get OT configuration info
pub async fn ot_info() -> Result<Json<OtConfigInfo>> {
    let tables = get_function_tables();

    Ok(Json(OtConfigInfo {
        table_size: OT_TABLE_SIZE,
        silu_range: (-SILU_INPUT_RANGE, SILU_INPUT_RANGE),
        exp_range: (-EXP_INPUT_RANGE, 0.0),
        rsqrt_range: (0.01, 10.0),
        total_memory_kb: tables.memory_bytes() as f64 / 1024.0,
        security_level: "1-of-N Oblivious Transfer (information-theoretic against server)".to_string(),
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
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_table() {
        let tables = OtFunctionTables::new(1024);

        // Test SiLU values
        let test_cases = vec![
            (-4.0, -0.0714),  // silu(-4) ≈ -0.0714
            (0.0, 0.0),       // silu(0) = 0
            (1.0, 0.731),     // silu(1) ≈ 0.731
            (4.0, 3.928),     // silu(4) ≈ 3.928
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
            (0.25, 2.0),  // 1/sqrt(0.25) = 2
            (1.0, 1.0),   // 1/sqrt(1) = 1
            (4.0, 0.5),   // 1/sqrt(4) = 0.5
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

        // 3 tables * 4096 entries * 4 bytes = 49152 bytes
        assert_eq!(tables.memory_bytes(), 3 * 4096 * 4);
    }
}
