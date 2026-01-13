//! ShardLM V2 Headless Client
//!
//! A native Rust client for running ShardLM V2 inference benchmarks.
//! Designed for running on H100 instances for paper artifact evaluation.
//!
//! # Example
//!
//! ```no_run
//! use shardlm_v2_client::ShardLmClient;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut client = ShardLmClient::new("http://localhost:9090");
//!
//!     // Health check
//!     if !client.health_check().await? {
//!         eprintln!("Server not healthy");
//!         return Ok(());
//!     }
//!
//!     // Create session
//!     let session = client.create_session().await?;
//!     println!("Session: {}", session.session_id);
//!
//!     // Generate
//!     let result = client.generate("Hello, how are you?", 50, 0.7).await?;
//!     println!("Generated: {}", result.text);
//!
//!     Ok(())
//! }
//! ```

pub mod benchmark;
pub mod error;
pub mod inference;
pub mod session;

pub use benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkRunner};
pub use error::{ClientError, Result};
pub use inference::{EmbeddingShares, GenerationResult, GenerationTiming};
pub use session::{ModelInfo, ServerInfo, SessionInfo};

use inference::{
    BatchedPrefillRequest, BatchedPrefillResponse, DirectEmbeddingRequest, DirectEmbeddingResponse,
};
use rand::Rng;
use session::CreateSessionRequest;
use std::time::Instant;

/// ShardLM V2 client
pub struct ShardLmClient {
    /// Server base URL
    server_url: String,
    /// HTTP client
    http: reqwest::Client,
    /// Current session ID
    session_id: Option<String>,
    /// Server info (cached)
    server_info: Option<ServerInfo>,
}

impl ShardLmClient {
    /// Create a new client
    pub fn new(server_url: &str) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 min timeout for large requests
            .build()
            .expect("Failed to create HTTP client");

        Self {
            server_url: server_url.trim_end_matches('/').to_string(),
            http,
            session_id: None,
            server_info: None,
        }
    }

    /// Get the server URL
    pub fn server_url(&self) -> &str {
        &self.server_url
    }

    /// Get the current session ID
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.server_url);
        let resp = self.http.get(&url).send().await?;
        Ok(resp.status().is_success())
    }

    /// Get server info
    pub async fn get_info(&mut self) -> Result<ServerInfo> {
        if let Some(ref info) = self.server_info {
            return Ok(info.clone());
        }

        let url = format!("{}/v2/info", self.server_url);
        let resp = self.http.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let info: ServerInfo = resp.json().await?;
        self.server_info = Some(info.clone());
        Ok(info)
    }

    /// Create a new session
    pub async fn create_session(&mut self) -> Result<SessionInfo> {
        let url = format!("{}/v2/session/new", self.server_url);
        let request = CreateSessionRequest::default();

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let session: SessionInfo = resp.json().await?;
        self.session_id = Some(session.session_id.clone());
        Ok(session)
    }

    /// Fetch embeddings for token IDs (direct lookup - reveals tokens to server)
    pub async fn fetch_embeddings(&self, token_ids: &[u32]) -> Result<EmbeddingShares> {
        let session_id = self
            .session_id
            .as_ref()
            .ok_or(ClientError::NoSession)?
            .clone();

        let url = format!("{}/v2/secure/embeddings/direct", self.server_url);
        let request = DirectEmbeddingRequest {
            session_id,
            token_ids: token_ids.to_vec(),
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let response: DirectEmbeddingResponse = resp.json().await?;
        Ok(EmbeddingShares {
            client: response.client_shares,
            server: response.server_shares,
        })
    }

    /// Run batched prefill (V2 - GPU accelerated)
    pub async fn prefill_v2(
        &self,
        hidden_client: &[Vec<f32>],
        hidden_server: &[Vec<f32>],
    ) -> Result<BatchedPrefillResponse> {
        let session_id = self
            .session_id
            .as_ref()
            .ok_or(ClientError::NoSession)?
            .clone();

        let url = format!("{}/v2/secure/gpu/prefill_v2", self.server_url);
        let request = BatchedPrefillRequest {
            session_id,
            hidden_client: hidden_client.to_vec(),
            hidden_server: hidden_server.to_vec(),
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let response: BatchedPrefillResponse = resp.json().await?;
        Ok(response)
    }

    /// Run batched prefill (V3 - minimal GPU transfers)
    pub async fn prefill_v3(
        &self,
        hidden_client: &[Vec<f32>],
        hidden_server: &[Vec<f32>],
    ) -> Result<BatchedPrefillResponse> {
        let session_id = self
            .session_id
            .as_ref()
            .ok_or(ClientError::NoSession)?
            .clone();

        let url = format!("{}/v2/secure/gpu/prefill_v3", self.server_url);
        let request = BatchedPrefillRequest {
            session_id,
            hidden_client: hidden_client.to_vec(),
            hidden_server: hidden_server.to_vec(),
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        let response: BatchedPrefillResponse = resp.json().await?;
        Ok(response)
    }

    /// Run batched prefill (V3-CC - H100 Confidential Computing)
    pub async fn prefill_v3_cc(
        &self,
        hidden_client: &[Vec<f32>],
        hidden_server: &[Vec<f32>],
    ) -> Result<BatchedPrefillResponse> {
        let session_id = self
            .session_id
            .as_ref()
            .ok_or(ClientError::NoSession)?
            .clone();

        let url = format!("{}/v3/cc/prefill", self.server_url);
        let request = BatchedPrefillRequest {
            session_id,
            hidden_client: hidden_client.to_vec(),
            hidden_server: hidden_server.to_vec(),
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        // The CC response has additional fields, but we extract just the base response
        let response: serde_json::Value = resp.json().await?;
        let prefill_response = BatchedPrefillResponse {
            final_hidden_client: serde_json::from_value(response["final_hidden_client"].clone())?,
            final_hidden_server: serde_json::from_value(response["final_hidden_server"].clone())?,
            k_cache: serde_json::from_value(response["k_cache"].clone())?,
            v_cache: serde_json::from_value(response["v_cache"].clone())?,
            logits_client: serde_json::from_value(response["logits_client"].clone())?,
            logits_server: serde_json::from_value(response["logits_server"].clone())?,
        };
        Ok(prefill_response)
    }

    /// Run batched prefill (V3-MPC - True MPC with Beaver triples)
    pub async fn prefill_v3_mpc(
        &self,
        hidden_client: &[Vec<f32>],
        hidden_server: &[Vec<f32>],
    ) -> Result<BatchedPrefillResponse> {
        let session_id = self
            .session_id
            .as_ref()
            .ok_or(ClientError::NoSession)?
            .clone();

        let url = format!("{}/v3/mpc/prefill", self.server_url);
        let request = BatchedPrefillRequest {
            session_id,
            hidden_client: hidden_client.to_vec(),
            hidden_server: hidden_server.to_vec(),
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        // The MPC response has additional fields, but we extract just the base response
        let response: serde_json::Value = resp.json().await?;
        let prefill_response = BatchedPrefillResponse {
            final_hidden_client: serde_json::from_value(response["final_hidden_client"].clone())?,
            final_hidden_server: serde_json::from_value(response["final_hidden_server"].clone())?,
            k_cache: serde_json::from_value(response["k_cache"].clone())?,
            v_cache: serde_json::from_value(response["v_cache"].clone())?,
            logits_client: serde_json::from_value(response["logits_client"].clone())?,
            logits_server: serde_json::from_value(response["logits_server"].clone())?,
        };
        Ok(prefill_response)
    }

    /// Run batched prefill (V3-OT - Oblivious Transfer for secure function evaluation)
    pub async fn prefill_v3_ot(
        &self,
        hidden_client: &[Vec<f32>],
        hidden_server: &[Vec<f32>],
    ) -> Result<BatchedPrefillResponse> {
        let session_id = self
            .session_id
            .as_ref()
            .ok_or(ClientError::NoSession)?
            .clone();

        let url = format!("{}/v3/ot/prefill", self.server_url);
        let request = BatchedPrefillRequest {
            session_id,
            hidden_client: hidden_client.to_vec(),
            hidden_server: hidden_server.to_vec(),
        };

        let resp = self.http.post(&url).json(&request).send().await?;

        if !resp.status().is_success() {
            return Err(ClientError::Server {
                status: resp.status().as_u16(),
                message: resp.text().await.unwrap_or_default(),
            });
        }

        // The OT response has additional fields, but we extract just the base response
        let response: serde_json::Value = resp.json().await?;
        let prefill_response = BatchedPrefillResponse {
            final_hidden_client: serde_json::from_value(response["final_hidden_client"].clone())?,
            final_hidden_server: serde_json::from_value(response["final_hidden_server"].clone())?,
            k_cache: serde_json::from_value(response["k_cache"].clone())?,
            v_cache: serde_json::from_value(response["v_cache"].clone())?,
            logits_client: serde_json::from_value(response["logits_client"].clone())?,
            logits_server: serde_json::from_value(response["logits_server"].clone())?,
        };
        Ok(prefill_response)
    }

    /// Sample token from logits using temperature sampling
    fn sample_token(&self, logits: &[f32], temperature: f32) -> u32 {
        if temperature <= 0.0 {
            // Greedy sampling
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        } else {
            // Temperature sampling with softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = logits
                .iter()
                .map(|x| ((x - max_logit) / temperature).exp())
                .collect();
            let sum: f32 = exp_logits.iter().sum();
            let probs: Vec<f32> = exp_logits.iter().map(|x| x / sum).collect();

            // Sample from distribution
            let mut rng = rand::thread_rng();
            let r: f32 = rng.gen();
            let mut cumsum = 0.0;
            for (i, p) in probs.iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    return i as u32;
                }
            }
            (probs.len() - 1) as u32
        }
    }

    /// Reconstruct logits from shares
    fn reconstruct_logits(&self, client: &[f32], server: &[f32]) -> Vec<f32> {
        client.iter().zip(server).map(|(c, s)| c + s).collect()
    }

    /// Complete generation: embeddings → prefill → decode
    ///
    /// This is the main high-level API for running inference.
    pub async fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<GenerationResult> {
        // Ensure we have a session
        if self.session_id.is_none() {
            self.create_session().await?;
        }

        // Get server info for tokenizer access
        // For now, we use a simple word-based tokenization as placeholder
        // In production, we'd use the actual tokenizer from the server
        let token_ids: Vec<u32> = self.simple_tokenize(prompt);

        let mut timing = GenerationTiming::default();
        let start = Instant::now();

        // Step 1: Fetch embeddings
        let embed_start = Instant::now();
        let embeddings = self.fetch_embeddings(&token_ids).await?;
        timing.embedding_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

        // Step 2: Prefill (process all prompt tokens)
        let prefill_start = Instant::now();
        let prefill_result = self.prefill_v2(&embeddings.client, &embeddings.server).await?;
        timing.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        // Step 3: Decode (generate tokens one at a time)
        let decode_start = Instant::now();
        let mut generated_tokens: Vec<u32> = Vec::new();

        // First token from prefill logits
        let logits = self.reconstruct_logits(&prefill_result.logits_client, &prefill_result.logits_server);
        let first_token = self.sample_token(&logits, temperature);
        generated_tokens.push(first_token);

        // Check for EOS token (for Qwen: 151645)
        const EOS_TOKEN: u32 = 151645;

        if first_token != EOS_TOKEN && generated_tokens.len() < max_tokens {
            // For now, we only do prefill. Full autoregressive decode
            // would require additional server endpoints for KV cache updates.
            // This is sufficient for benchmarking prefill performance.
        }

        timing.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        timing.tokens_generated = generated_tokens.len();
        timing.tokens_per_second = if timing.total_ms > 0.0 {
            (timing.tokens_generated as f64) / (timing.total_ms / 1000.0)
        } else {
            0.0
        };

        // Simple decode (placeholder - in production would use actual tokenizer)
        let text = self.simple_detokenize(&generated_tokens);

        Ok(GenerationResult {
            token_ids: generated_tokens,
            text,
            timing,
        })
    }

    /// Simple word-based tokenization (placeholder)
    /// In production, use the actual model tokenizer
    pub fn simple_tokenize(&self, text: &str) -> Vec<u32> {
        // Use simple character-level tokenization as placeholder
        // Real implementation would load tokenizer from model
        text.chars()
            .filter_map(|c| {
                if c.is_ascii() {
                    Some(c as u32)
                } else {
                    Some(0) // Unknown token
                }
            })
            .collect()
    }

    /// Simple detokenization (placeholder)
    fn simple_detokenize(&self, token_ids: &[u32]) -> String {
        token_ids
            .iter()
            .filter_map(|&id| {
                if id < 128 {
                    Some(id as u8 as char)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Run prefill-only benchmark (most useful for measuring secure inference performance)
    pub async fn benchmark_prefill(
        &mut self,
        prompt: &str,
        runs: usize,
    ) -> Result<Vec<GenerationTiming>> {
        // Ensure we have a session
        if self.session_id.is_none() {
            self.create_session().await?;
        }

        let token_ids: Vec<u32> = self.simple_tokenize(prompt);
        let mut results = Vec::with_capacity(runs);

        for _ in 0..runs {
            let mut timing = GenerationTiming::default();
            let start = Instant::now();

            // Fetch embeddings
            let embed_start = Instant::now();
            let embeddings = self.fetch_embeddings(&token_ids).await?;
            timing.embedding_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

            // Prefill
            let prefill_start = Instant::now();
            let _ = self.prefill_v2(&embeddings.client, &embeddings.server).await?;
            timing.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

            timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
            timing.tokens_generated = token_ids.len();

            results.push(timing);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenize() {
        let client = ShardLmClient::new("http://localhost:9090");
        let tokens = client.simple_tokenize("Hello");
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], 'H' as u32);
    }

    #[test]
    fn test_sample_token_greedy() {
        let client = ShardLmClient::new("http://localhost:9090");
        let logits = vec![0.1, 0.5, 0.2, 0.9, 0.3];
        let token = client.sample_token(&logits, 0.0);
        assert_eq!(token, 3); // Index of max value
    }

    #[test]
    fn test_reconstruct_logits() {
        let client = ShardLmClient::new("http://localhost:9090");
        let client_share = vec![0.5, 1.0, -0.5];
        let server_share = vec![0.5, -0.5, 1.0];
        let logits = client.reconstruct_logits(&client_share, &server_share);
        assert_eq!(logits, vec![1.0, 0.5, 0.5]);
    }
}
