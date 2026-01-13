//! Inference request/response types for ShardLM V2 client

use serde::{Deserialize, Serialize};

/// Direct embedding lookup request
#[derive(Debug, Serialize)]
pub struct DirectEmbeddingRequest {
    pub session_id: String,
    pub token_ids: Vec<u32>,
}

/// Direct embedding response (returns secret shares)
#[derive(Debug, Deserialize)]
pub struct DirectEmbeddingResponse {
    /// Client shares for each token [num_tokens][hidden_dim]
    pub client_shares: Vec<Vec<f32>>,
    /// Server shares for each token [num_tokens][hidden_dim]
    pub server_shares: Vec<Vec<f32>>,
}

/// Embedding shares for client-side storage
#[derive(Debug, Clone)]
pub struct EmbeddingShares {
    /// Client shares [num_tokens][hidden_dim]
    pub client: Vec<Vec<f32>>,
    /// Server shares [num_tokens][hidden_dim]
    pub server: Vec<Vec<f32>>,
}

/// Batched prefill request (V2/V3)
#[derive(Debug, Serialize)]
pub struct BatchedPrefillRequest {
    pub session_id: String,
    /// Hidden states for ALL prompt tokens: [seq_len][hidden_dim]
    pub hidden_client: Vec<Vec<f32>>,
    pub hidden_server: Vec<Vec<f32>>,
}

/// Response from batched prefill
#[derive(Debug, Deserialize)]
pub struct BatchedPrefillResponse {
    /// Final hidden state for LAST token only (for generation)
    pub final_hidden_client: Vec<f32>,
    pub final_hidden_server: Vec<f32>,
    /// KV cache for all tokens: [layer][seq_len][kv_dim]
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    /// Logits for last token (for first generation step)
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
}

/// Generate token request (for decode phase)
#[derive(Debug, Serialize)]
pub struct GenerateTokenRequest {
    pub session_id: String,
    /// Hidden state for current token
    pub hidden_client: Vec<f32>,
    pub hidden_server: Vec<f32>,
    /// KV cache from prefill
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    /// Current sequence position
    pub position: usize,
}

/// Generate token response
#[derive(Debug, Deserialize)]
pub struct GenerateTokenResponse {
    /// Updated hidden state
    pub hidden_client: Vec<f32>,
    pub hidden_server: Vec<f32>,
    /// Updated KV cache
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    /// Logits for sampling
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
}

/// Result of a complete generation
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Decoded text
    pub text: String,
    /// Timing information
    pub timing: GenerationTiming,
}

/// Timing breakdown for generation
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct GenerationTiming {
    /// Time to fetch embeddings (ms)
    pub embedding_ms: f64,
    /// Time for prefill phase (ms)
    pub prefill_ms: f64,
    /// Time for decode phase (ms)
    pub decode_ms: f64,
    /// Total time (ms)
    pub total_ms: f64,
    /// Tokens generated
    pub tokens_generated: usize,
    /// Tokens per second
    pub tokens_per_second: f64,
}

/// WebSocket streaming message types
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StreamMessage {
    #[serde(rename = "token")]
    Token {
        token_id: u32,
        token: String,
        position: usize,
    },
    #[serde(rename = "done")]
    Done { total_tokens: usize, total_ms: f64 },
    #[serde(rename = "error")]
    Error { message: String },
}

/// WebSocket generate request
#[derive(Debug, Serialize)]
pub struct WsGenerateRequest {
    pub session_id: String,
    pub prompt: String,
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_temperature() -> f32 {
    0.7
}
