//! Inference endpoints
//!
//! These endpoints handle text generation using the distributed Llama 70B model.

#[allow(unused_imports)]
use std::time::Instant;

use axum::{
    extract::State,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream;
use serde::{Deserialize, Serialize};
#[allow(unused_imports)]
use uuid::Uuid;

use crate::error::{Result, ServerError};
use crate::state::AppState;

/// Text generation request
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    /// Session ID (optional - creates new session if not provided)
    #[serde(default)]
    pub session_id: Option<String>,
    /// Input prompt
    pub prompt: String,
    /// Maximum tokens to generate
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    /// Temperature for sampling (0.0 = greedy)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Whether to use chat format
    #[serde(default)]
    pub chat_format: bool,
    /// System message for chat format
    #[serde(default)]
    pub system_message: Option<String>,
}

/// Text generation response
#[derive(Debug, Serialize)]
pub struct GenerateResponse {
    /// Session ID (for continuing conversation)
    pub session_id: String,
    /// Generated text
    pub text: String,
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of generated tokens
    pub generated_tokens: usize,
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    /// Tokens per second
    pub tokens_per_second: f32,
}

/// Streaming generation event
#[derive(Debug, Serialize)]
pub struct StreamEvent {
    /// Event type: "token", "done", or "error"
    pub event: String,
    /// Token text (for "token" events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Full response (for "done" events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<GenerateResponse>,
    /// Error message (for "error" events)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// POST /v2/generate - Generate text
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn generate(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>> {
    use shardlm_v2_model::tokenizer::ChatMessage;

    // Validate request
    if request.prompt.is_empty() {
        return Err(ServerError::InvalidRequest("Prompt cannot be empty".to_string()));
    }

    // Get or create session
    let session_id = match &request.session_id {
        Some(id) => {
            let uuid = Uuid::parse_str(id)
                .map_err(|_| ServerError::InvalidRequest("Invalid session ID".to_string()))?;
            state.touch_session(&uuid)?;
            uuid
        }
        None => state.create_session(),
    };

    // Get generation parameters
    let max_new_tokens = request
        .max_new_tokens
        .unwrap_or(state.config.max_new_tokens);
    let temperature = request
        .temperature
        .unwrap_or(state.config.default_temperature);

    // Get tokenizer and encode prompt in a scope so guard is dropped before await
    let (token_ids, prompt_tokens, eos_tokens) = {
        let tokenizer_guard = state.get_tokenizer()?;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or(ServerError::ModelNotLoaded)?;

        let token_ids = if request.chat_format {
            let mut messages = Vec::new();
            if let Some(system) = &request.system_message {
                messages.push(ChatMessage::system(system));
            }
            messages.push(ChatMessage::user(&request.prompt));
            tokenizer.encode_chat(&messages)?
        } else {
            tokenizer.encode(&request.prompt, true)?
        };

        let prompt_tokens = token_ids.len();
        // Get EOS tokens for proper stopping
        let eos_tokens = vec![tokenizer.eos_token_id, tokenizer.eot_token_id];

        tracing::debug!(
            session_id = %session_id,
            prompt_tokens = prompt_tokens,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            "Starting generation"
        );
        (token_ids, prompt_tokens, eos_tokens)
    }; // tokenizer_guard dropped here before await

    // Run generation in blocking task (CPU-bound with CPU offloading)
    let gen_start = Instant::now();

    let state_clone = state.clone();
    let generated_ids = tokio::task::spawn_blocking(move || -> Result<Vec<u32>> {
        let mut engine_guard = state_clone.get_engine()?;
        let engine = engine_guard
            .as_mut()
            .ok_or(ServerError::ModelNotLoaded)?;
        engine
            .generate_with_eos(&token_ids, max_new_tokens, temperature, Some(&eos_tokens))
            .map_err(|e| ServerError::InferenceError(e.to_string()))
    })
    .await
    .map_err(|e| ServerError::Internal(format!("Task join error: {}", e)))??;

    let gen_time = gen_start.elapsed();

    // Decode output - need to get tokenizer again
    let tokenizer_guard = state.get_tokenizer()?;
    let tokenizer = tokenizer_guard
        .as_ref()
        .ok_or(ServerError::ModelNotLoaded)?;

    let new_tokens = generated_ids.len() - prompt_tokens;
    // Only decode the newly generated tokens, not the prompt
    let output_text = tokenizer.decode(&generated_ids[prompt_tokens..], true)?;

    // Calculate stats
    let generation_time_ms = gen_time.as_millis() as u64;
    let tokens_per_second = if gen_time.as_secs_f32() > 0.0 {
        new_tokens as f32 / gen_time.as_secs_f32()
    } else {
        0.0
    };

    tracing::info!(
        session_id = %session_id,
        new_tokens = new_tokens,
        generation_time_ms = generation_time_ms,
        tokens_per_second = tokens_per_second,
        "Generation complete"
    );

    Ok(Json(GenerateResponse {
        session_id: session_id.to_string(),
        text: output_text,
        prompt_tokens,
        generated_tokens: new_tokens,
        generation_time_ms,
        tokens_per_second,
    }))
}

#[cfg(not(feature = "cuda"))]
#[axum::debug_handler]
pub async fn generate(
    State(_state): State<AppState>,
    Json(_request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>> {
    Err(ServerError::Internal("CUDA feature not enabled".to_string()))
}

/// POST /v2/generate/stream - Generate text with streaming
#[cfg(feature = "cuda")]
#[axum::debug_handler]
pub async fn generate_stream(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Sse<futures::stream::Iter<std::vec::IntoIter<std::result::Result<Event, std::convert::Infallible>>>> {
    use shardlm_v2_model::tokenizer::ChatMessage;

    // For now, just do non-streaming generation and return as a single event
    // Real streaming would require modifying the engine to yield tokens incrementally

    // Helper to create error stream - use stream::iter for type consistency
    fn error_stream(
        msg: &str,
    ) -> Sse<futures::stream::Iter<std::vec::IntoIter<std::result::Result<Event, std::convert::Infallible>>>> {
        let event = StreamEvent {
            event: "error".to_string(),
            token: None,
            response: None,
            error: Some(msg.to_string()),
        };
        let events = vec![Ok(Event::default().data(serde_json::to_string(&event).unwrap()))];
        Sse::new(stream::iter(events))
    }

    // Validate request
    if request.prompt.is_empty() {
        return error_stream("Prompt cannot be empty");
    }

    // Get or create session
    let session_id = match &request.session_id {
        Some(id) => match Uuid::parse_str(id) {
            Ok(uuid) => {
                if let Err(e) = state.touch_session(&uuid) {
                    return error_stream(&e.to_string());
                }
                uuid
            }
            Err(_) => {
                return error_stream("Invalid session ID");
            }
        },
        None => state.create_session(),
    };

    let max_new_tokens = request
        .max_new_tokens
        .unwrap_or(state.config.max_new_tokens);
    let temperature = request
        .temperature
        .unwrap_or(state.config.default_temperature);

    // Get tokenizer and encode prompt in a closure so guard is dropped before await
    let encode_result = (|| -> std::result::Result<(Vec<u32>, usize), String> {
        let tokenizer_guard = state.get_tokenizer().map_err(|e| e.to_string())?;
        let tokenizer = tokenizer_guard
            .as_ref()
            .ok_or_else(|| "Model not loaded".to_string())?;

        let token_ids = if request.chat_format {
            let mut messages = Vec::new();
            if let Some(system) = &request.system_message {
                messages.push(ChatMessage::system(system.clone()));
            }
            messages.push(ChatMessage::user(request.prompt.clone()));
            tokenizer.encode_chat(&messages).map_err(|e| e.to_string())?
        } else {
            tokenizer
                .encode(&request.prompt, true)
                .map_err(|e| e.to_string())?
        };

        let prompt_tokens = token_ids.len();

        // Debug: log the first and last few tokens to verify encoding
        tracing::info!(
            "Encoded {} tokens. First 10: {:?}, Last 10: {:?}",
            prompt_tokens,
            &token_ids[..token_ids.len().min(10)],
            &token_ids[token_ids.len().saturating_sub(10)..]
        );

        Ok((token_ids, prompt_tokens))
    })(); // tokenizer_guard dropped here when closure returns

    let (token_ids, prompt_tokens) = match encode_result {
        Ok(v) => v,
        Err(e) => return error_stream(&e),
    };

    // Get EOS tokens from tokenizer for proper stopping
    let eos_tokens = match state.get_tokenizer() {
        Ok(guard) => match guard.as_ref() {
            Some(t) => vec![t.eos_token_id, t.eot_token_id],
            None => vec![128001, 128009], // Llama 3 defaults
        },
        Err(_) => vec![128001, 128009],
    };

    let gen_start = Instant::now();

    // Run generation in blocking task (CPU-bound with CPU offloading)
    let state_clone = state.clone();
    let result = tokio::task::spawn_blocking(move || -> std::result::Result<Vec<u32>, String> {
        let mut engine_guard = state_clone
            .get_engine()
            .map_err(|e| e.to_string())?;
        let engine = engine_guard
            .as_mut()
            .ok_or_else(|| "Model not loaded".to_string())?;
        engine
            .generate_with_eos(&token_ids, max_new_tokens, temperature, Some(&eos_tokens))
            .map_err(|e| e.to_string())
    })
    .await;

    let (generated_ids, error) = match result {
        Ok(Ok(ids)) => (ids, None),
        Ok(Err(e)) => (vec![], Some(e)),
        Err(e) => (vec![], Some(format!("Task join error: {}", e))),
    };

    let gen_time = gen_start.elapsed();

    // Build stream events
    let events: Vec<std::result::Result<Event, std::convert::Infallible>> = if let Some(err) = error
    {
        let event = StreamEvent {
            event: "error".to_string(),
            token: None,
            response: None,
            error: Some(err),
        };
        vec![Ok(Event::default().data(serde_json::to_string(&event).unwrap()))]
    } else {
        let new_tokens = generated_ids.len() - prompt_tokens;

        // Get tokenizer again for decoding - only decode newly generated tokens
        let output_text = match state.get_tokenizer() {
            Ok(guard) => match guard.as_ref() {
                Some(t) => t.decode(&generated_ids[prompt_tokens..], true).unwrap_or_default(),
                None => String::new(),
            },
            Err(_) => String::new(),
        };
        let generation_time_ms = gen_time.as_millis() as u64;
        let tokens_per_second = if gen_time.as_secs_f32() > 0.0 {
            new_tokens as f32 / gen_time.as_secs_f32()
        } else {
            0.0
        };

        let event = StreamEvent {
            event: "done".to_string(),
            token: None,
            response: Some(GenerateResponse {
                session_id: session_id.to_string(),
                text: output_text,
                prompt_tokens,
                generated_tokens: new_tokens,
                generation_time_ms,
                tokens_per_second,
            }),
            error: None,
        };
        vec![Ok(Event::default().data(serde_json::to_string(&event).unwrap()))]
    };

    Sse::new(stream::iter(events))
}

#[cfg(not(feature = "cuda"))]
#[axum::debug_handler]
pub async fn generate_stream(
    State(_state): State<AppState>,
    Json(_request): Json<GenerateRequest>,
) -> Sse<futures::stream::Iter<std::vec::IntoIter<std::result::Result<Event, std::convert::Infallible>>>> {
    let event = StreamEvent {
        event: "error".to_string(),
        token: None,
        response: None,
        error: Some("CUDA feature not enabled".to_string()),
    };
    let events = vec![Ok(Event::default().data(serde_json::to_string(&event).unwrap()))];
    Sse::new(stream::iter(events))
}
