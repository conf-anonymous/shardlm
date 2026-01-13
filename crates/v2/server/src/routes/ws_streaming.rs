//! WebSocket streaming for secure inference
//!
//! This module provides WebSocket-based streaming for token generation,
//! eliminating HTTP round-trip overhead per token.
//!
//! # Protocol
//!
//! 1. Client connects to `/v2/secure/ws/generate`
//! 2. Client sends JSON request with prompt shares
//! 3. Server streams tokens back as JSON messages
//! 4. Server sends final message with `done: true`

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::state::AppState;

/// WebSocket request for secure generation
#[derive(Debug, Clone, Deserialize)]
pub struct WsGenerateRequest {
    /// Session ID
    pub session_id: String,
    /// Client's share of hidden states for all prompt tokens
    pub hidden_client: Vec<Vec<f32>>,
    /// Server's share of hidden states for all prompt tokens
    pub hidden_server: Vec<Vec<f32>>,
    /// Maximum tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_new_tokens: usize,
    /// Temperature for sampling
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_max_tokens() -> usize { 100 }
fn default_temperature() -> f32 { 0.7 }

/// Streaming token message
#[derive(Debug, Clone, Serialize)]
pub struct WsTokenMessage {
    /// Token ID that was generated
    pub token_id: u32,
    /// Client's share of logits for next token
    pub logits_client: Vec<f32>,
    /// Server's share of logits for next token
    pub logits_server: Vec<f32>,
    /// Whether this is the final token
    pub done: bool,
    /// Token position in sequence
    pub position: usize,
    /// Time taken for this token (ms)
    pub token_time_ms: u64,
}

/// Error message
#[derive(Debug, Serialize)]
pub struct WsErrorMessage {
    pub error: String,
    pub code: String,
}

/// Messages sent from GPU thread to WebSocket handler
#[derive(Debug)]
enum GpuMessage {
    Token(WsTokenMessage),
    Error(WsErrorMessage),
    Done,
}

/// WebSocket upgrade handler
pub async fn ws_generate_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(|socket| handle_ws_generate(socket, state))
}

/// Handle WebSocket connection for generation
async fn handle_ws_generate(mut socket: WebSocket, state: AppState) {
    // Wait for initial request
    let request: WsGenerateRequest = match socket.recv().await {
        Some(Ok(Message::Text(text))) => {
            match serde_json::from_str(&text) {
                Ok(req) => req,
                Err(e) => {
                    let _ = socket.send(Message::Text(
                        serde_json::to_string(&WsErrorMessage {
                            error: format!("Invalid request: {}", e),
                            code: "invalid_request".to_string(),
                        }).unwrap()
                    )).await;
                    return;
                }
            }
        }
        _ => {
            let _ = socket.send(Message::Text(
                serde_json::to_string(&WsErrorMessage {
                    error: "Expected JSON request".to_string(),
                    code: "invalid_request".to_string(),
                }).unwrap()
            )).await;
            return;
        }
    };

    tracing::info!(
        session_id = %request.session_id,
        prompt_tokens = request.hidden_client.len(),
        max_new_tokens = request.max_new_tokens,
        "WebSocket generation started"
    );

    // Create channel for GPU thread to send tokens
    let (tx, mut rx) = mpsc::channel::<GpuMessage>(32);

    // Spawn blocking task for GPU work
    let gpu_state = state.clone();
    let gpu_request = request.clone();
    tokio::task::spawn_blocking(move || {
        run_gpu_generation(gpu_state, gpu_request, tx);
    });

    // Forward tokens from GPU thread to WebSocket
    while let Some(msg) = rx.recv().await {
        match msg {
            GpuMessage::Token(token) => {
                let done = token.done;
                if socket.send(Message::Text(serde_json::to_string(&token).unwrap())).await.is_err() {
                    tracing::warn!("WebSocket send failed, client disconnected");
                    return;
                }
                if done {
                    break;
                }
            }
            GpuMessage::Error(err) => {
                let _ = socket.send(Message::Text(serde_json::to_string(&err).unwrap())).await;
                return;
            }
            GpuMessage::Done => {
                break;
            }
        }
    }

    tracing::info!("WebSocket generation complete");
}

/// Run GPU generation in blocking context
#[cfg(feature = "cuda")]
fn run_gpu_generation(
    state: AppState,
    request: WsGenerateRequest,
    tx: mpsc::Sender<GpuMessage>,
) {
    use shardlm_v2_sharing::{
        ServerContext, secure_rms_norm_approx, secure_swiglu_approx,
    };
    use shardlm_v2_core::gpu::GpuDevice;
    use std::time::Instant;

    // Get GPU resources
    let gpu_weights_guard = match state.get_gpu_secure_weights() {
        Ok(guard) => guard,
        Err(e) => {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: format!("Failed to get weights: {}", e),
                code: "internal_error".to_string(),
            }));
            return;
        }
    };
    let gpu_weights = match gpu_weights_guard.as_ref() {
        Some(w) => w,
        None => {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: "GPU weights not initialized".to_string(),
                code: "internal_error".to_string(),
            }));
            return;
        }
    };

    let kernel_guard = match state.get_gpu_kernel_contexts() {
        Ok(guard) => guard,
        Err(e) => {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: format!("Failed to get kernels: {}", e),
                code: "internal_error".to_string(),
            }));
            return;
        }
    };
    let kernels = match kernel_guard.get(0) {
        Some(k) => k,
        None => {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: "Kernel context not initialized".to_string(),
                code: "internal_error".to_string(),
            }));
            return;
        }
    };

    let device = match GpuDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: format!("GPU error: {}", e),
                code: "gpu_error".to_string(),
            }));
            return;
        }
    };

    let ctx = ServerContext::new();
    let num_layers = gpu_weights.num_layers;
    let num_heads = gpu_weights.num_heads;
    let num_kv_heads = gpu_weights.num_kv_heads;
    let head_dim = gpu_weights.head_dim;
    let seq_len = request.hidden_client.len();

    // === PREFILL PHASE ===
    let prefill_start = Instant::now();

    // Initialize KV cache
    let mut k_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len + request.max_new_tokens); num_layers];
    let mut v_cache: Vec<Vec<Vec<f32>>> = vec![Vec::with_capacity(seq_len + request.max_new_tokens); num_layers];

    let mut hidden_client = request.hidden_client;
    let mut hidden_server = request.hidden_server;

    // Process all prompt tokens through all layers
    for layer_idx in 0..num_layers {
        let layer = gpu_weights.layer(layer_idx);
        let mut new_hidden_client = Vec::with_capacity(seq_len);
        let mut new_hidden_server = Vec::with_capacity(seq_len);

        for pos in 0..seq_len {
            // RMSNorm
            let (normed_client, normed_server) = secure_rms_norm_approx(
                &hidden_client[pos], &hidden_server[pos],
                &layer.input_layernorm, 1e-6, &ctx,
            );

            // QKV projection
            let qkv_result = match layer.attention.project_qkv_gpu(
                &ctx, &normed_client, &normed_server, pos, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("QKV failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            // Store KV
            let k_combined: Vec<f32> = qkv_result.k_client.iter()
                .zip(qkv_result.k_server.iter())
                .map(|(c, s)| c + s).collect();
            let v_combined: Vec<f32> = qkv_result.v_client.iter()
                .zip(qkv_result.v_server.iter())
                .map(|(c, s)| c + s).collect();
            k_cache[layer_idx].push(k_combined);
            v_cache[layer_idx].push(v_combined);

            // Attention
            let q_combined: Vec<f32> = qkv_result.q_client.iter()
                .zip(qkv_result.q_server.iter())
                .map(|(c, s)| c + s).collect();
            let k_slice: Vec<Vec<f32>> = k_cache[layer_idx][0..=pos].to_vec();
            let v_slice: Vec<Vec<f32>> = v_cache[layer_idx][0..=pos].to_vec();

            let attn_output = shardlm_v2_sharing::secure_attention_approx(
                &q_combined, &k_slice, &v_slice,
                num_heads, num_kv_heads, head_dim, &ctx,
            );
            let attn_server = vec![0.0; attn_output.len()];

            // O projection + residual
            let (o_client, o_server) = match layer.attention.project_output_gpu(
                &ctx, &attn_output, &attn_server, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("O proj failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            let h_after_attn_c: Vec<f32> = o_client.iter()
                .zip(hidden_client[pos].iter())
                .map(|(o, h)| o + h).collect();
            let h_after_attn_s: Vec<f32> = o_server.iter()
                .zip(hidden_server[pos].iter())
                .map(|(o, h)| o + h).collect();

            // FFN
            let (normed_ffn_c, normed_ffn_s) = secure_rms_norm_approx(
                &h_after_attn_c, &h_after_attn_s,
                &layer.post_attn_layernorm, 1e-6, &ctx,
            );

            let ffn_result = match layer.ffn.project_gate_up_gpu(
                &ctx, &normed_ffn_c, &normed_ffn_s, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("FFN gate/up failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            let (activated_c, activated_s) = secure_swiglu_approx(
                &ffn_result.gate_client, &ffn_result.gate_server,
                &ffn_result.up_client, &ffn_result.up_server, &ctx,
            );

            let (down_c, down_s) = match layer.ffn.project_down_gpu(
                &ctx, &activated_c, &activated_s, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("FFN down failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            let final_c: Vec<f32> = down_c.iter()
                .zip(h_after_attn_c.iter())
                .map(|(d, h)| d + h).collect();
            let final_s: Vec<f32> = down_s.iter()
                .zip(h_after_attn_s.iter())
                .map(|(d, h)| d + h).collect();

            new_hidden_client.push(final_c);
            new_hidden_server.push(final_s);
        }

        hidden_client = new_hidden_client;
        hidden_server = new_hidden_server;
    }

    // Get initial logits from last token
    let last_idx = seq_len - 1;
    let (normed_c, normed_s) = secure_rms_norm_approx(
        &hidden_client[last_idx], &hidden_server[last_idx],
        &gpu_weights.final_norm, 1e-6, &ctx,
    );

    let (mut logits_client, mut logits_server) = match gpu_weights.lm_head.forward_secure_gpu(
        &ctx, &normed_c, &normed_s, kernels, &device
    ) {
        Ok(r) => r,
        Err(e) => {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: format!("LM head failed: {}", e),
                code: "compute_error".to_string(),
            }));
            return;
        }
    };

    let prefill_time = prefill_start.elapsed();
    tracing::info!(
        prefill_ms = prefill_time.as_millis(),
        prompt_tokens = seq_len,
        "Prefill complete, starting generation"
    );

    // === GENERATION PHASE ===
    let mut current_hidden_c = hidden_client.pop().unwrap();
    let mut current_hidden_s = hidden_server.pop().unwrap();
    let mut position = seq_len;
    let eos_token = 151645u32; // Qwen EOS

    for step in 0..request.max_new_tokens {
        let token_start = Instant::now();

        // Sample token (simplified - client would normally do this securely)
        let logits: Vec<f32> = logits_client.iter()
            .zip(logits_server.iter())
            .map(|(c, s)| c + s)
            .collect();

        let token_id = sample_token(&logits, request.temperature);
        let token_time = token_start.elapsed().as_millis() as u64;

        // Send token to client
        let is_done = token_id == eos_token || step == request.max_new_tokens - 1;
        let msg = WsTokenMessage {
            token_id,
            logits_client: logits_client.clone(),
            logits_server: logits_server.clone(),
            done: is_done,
            position,
            token_time_ms: token_time,
        };

        if tx.blocking_send(GpuMessage::Token(msg)).is_err() {
            tracing::warn!("Channel closed, stopping generation");
            return;
        }

        if token_id == eos_token {
            tracing::info!(tokens_generated = step + 1, "Generation complete (EOS)");
            return;
        }

        if step == request.max_new_tokens - 1 {
            return;
        }

        // Get embedding for next token
        let embeddings = &gpu_weights.embeddings;
        let hidden_dim = gpu_weights.hidden_dim;
        let start = token_id as usize * hidden_dim;
        let end = start + hidden_dim;

        if end > embeddings.len() {
            let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                error: format!("Token {} out of vocab range", token_id),
                code: "vocab_error".to_string(),
            }));
            return;
        }

        current_hidden_c = embeddings[start..end].to_vec();
        current_hidden_s = vec![0.0; hidden_dim];

        // Process through all layers
        for layer_idx in 0..num_layers {
            let layer = gpu_weights.layer(layer_idx);

            let (normed_c, normed_s) = secure_rms_norm_approx(
                &current_hidden_c, &current_hidden_s,
                &layer.input_layernorm, 1e-6, &ctx,
            );

            let qkv_result = match layer.attention.project_qkv_gpu(
                &ctx, &normed_c, &normed_s, position, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("QKV failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            let k_combined: Vec<f32> = qkv_result.k_client.iter()
                .zip(qkv_result.k_server.iter())
                .map(|(c, s)| c + s).collect();
            let v_combined: Vec<f32> = qkv_result.v_client.iter()
                .zip(qkv_result.v_server.iter())
                .map(|(c, s)| c + s).collect();
            k_cache[layer_idx].push(k_combined);
            v_cache[layer_idx].push(v_combined);

            let q_combined: Vec<f32> = qkv_result.q_client.iter()
                .zip(qkv_result.q_server.iter())
                .map(|(c, s)| c + s).collect();

            let attn_output = shardlm_v2_sharing::secure_attention_approx(
                &q_combined,
                &k_cache[layer_idx],
                &v_cache[layer_idx],
                num_heads, num_kv_heads, head_dim, &ctx,
            );
            let attn_server = vec![0.0; attn_output.len()];

            let (o_c, o_s) = match layer.attention.project_output_gpu(
                &ctx, &attn_output, &attn_server, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("O proj failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            let h_after_attn_c: Vec<f32> = o_c.iter()
                .zip(current_hidden_c.iter())
                .map(|(o, h)| o + h).collect();
            let h_after_attn_s: Vec<f32> = o_s.iter()
                .zip(current_hidden_s.iter())
                .map(|(o, h)| o + h).collect();

            let (normed_ffn_c, normed_ffn_s) = secure_rms_norm_approx(
                &h_after_attn_c, &h_after_attn_s,
                &layer.post_attn_layernorm, 1e-6, &ctx,
            );

            let ffn_result = match layer.ffn.project_gate_up_gpu(
                &ctx, &normed_ffn_c, &normed_ffn_s, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("FFN failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            let (activated_c, activated_s) = secure_swiglu_approx(
                &ffn_result.gate_client, &ffn_result.gate_server,
                &ffn_result.up_client, &ffn_result.up_server, &ctx,
            );

            let (down_c, down_s) = match layer.ffn.project_down_gpu(
                &ctx, &activated_c, &activated_s, kernels, &device
            ) {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                        error: format!("FFN down failed: {}", e),
                        code: "compute_error".to_string(),
                    }));
                    return;
                }
            };

            current_hidden_c = down_c.iter()
                .zip(h_after_attn_c.iter())
                .map(|(d, h)| d + h).collect();
            current_hidden_s = down_s.iter()
                .zip(h_after_attn_s.iter())
                .map(|(d, h)| d + h).collect();
        }

        // Compute logits for next token
        let (normed_c, normed_s) = secure_rms_norm_approx(
            &current_hidden_c, &current_hidden_s,
            &gpu_weights.final_norm, 1e-6, &ctx,
        );

        (logits_client, logits_server) = match gpu_weights.lm_head.forward_secure_gpu(
            &ctx, &normed_c, &normed_s, kernels, &device
        ) {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
                    error: format!("LM head failed: {}", e),
                    code: "compute_error".to_string(),
                }));
                return;
            }
        };

        position += 1;
    }

    let _ = tx.blocking_send(GpuMessage::Done);
}

/// Simple temperature-based sampling
fn sample_token(logits: &[f32], temperature: f32) -> u32 {
    use rand::Rng;

    if temperature <= 0.0 {
        // Greedy
        return logits.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();

    // Softmax
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

    // Sample
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumsum = 0.0;
    for (i, p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return i as u32;
        }
    }

    (probs.len() - 1) as u32
}

#[cfg(not(feature = "cuda"))]
fn run_gpu_generation(
    _state: AppState,
    _request: WsGenerateRequest,
    tx: mpsc::Sender<GpuMessage>,
) {
    let _ = tx.blocking_send(GpuMessage::Error(WsErrorMessage {
        error: "CUDA not enabled".to_string(),
        code: "not_supported".to_string(),
    }));
}
