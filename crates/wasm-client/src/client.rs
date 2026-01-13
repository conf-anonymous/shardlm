//! ShardLM WASM Client
//!
//! Main client interface for browser-side OT operations.

use std::cell::RefCell;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use shardlm_ot::{IknpOtExtension, OtReceiver};
use shardlm_protocol::{Frame, MsgType, Payload};

use crate::error::WasmError;
use crate::timings::Timings;
use crate::transport::HttpClient;

/// Session creation response from server
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionResponse {
    pub session_id: String,
    pub protocol_version: u16,
    pub max_prompt_len: u16,
    pub ot_kappa: u16,
    pub cipher: String,
    pub model: String,
    pub embedding_dim: u32,
    pub vocab_size: u32,
    /// TTL in seconds
    #[serde(default = "default_ttl")]
    pub ttl_secs: i64,
}

fn default_ttl() -> i64 {
    900 // 15 minutes default
}

/// Session status response
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionStatusResponse {
    pub id: String,
    pub ready: bool,
    pub ttl_secs: i64,
    pub request_count: u32,
    pub max_requests: u32,
    pub expected_counter: u64,
}

/// Session refresh response
#[derive(Debug, Serialize, Deserialize)]
pub struct SessionRefreshResponse {
    pub session_id: String,
    pub ttl_secs: i64,
}

/// Internal mutable state
struct ClientState {
    receiver: Option<OtReceiver<IknpOtExtension>>,
    session_id: Option<String>,
    session_id_bytes: Option<[u8; 16]>,
    timings: Timings,
    vocab_size: u32,
    embedding_dim: u32,
    max_prompt_len: u16,
    /// Session TTL in seconds (updated on refresh)
    ttl_secs: i64,
    /// Last time TTL was updated (js timestamp)
    ttl_updated_at: f64,
}

/// ShardLM Client SDK for WASM
#[wasm_bindgen]
pub struct ShardLmClient {
    /// HTTP transport
    http: HttpClient,
    /// Mutable state wrapped in RefCell for interior mutability
    state: RefCell<ClientState>,
}

#[wasm_bindgen]
impl ShardLmClient {
    /// Create a new ShardLM client
    #[wasm_bindgen(constructor)]
    pub fn new(server_url: &str) -> ShardLmClient {
        ShardLmClient {
            http: HttpClient::new(server_url),
            state: RefCell::new(ClientState {
                receiver: None,
                session_id: None,
                session_id_bytes: None,
                timings: Timings::new(),
                vocab_size: 0,
                embedding_dim: 0,
                max_prompt_len: 0,
                ttl_secs: 0,
                ttl_updated_at: 0.0,
            }),
        }
    }

    /// Start a new session with the server
    #[wasm_bindgen]
    pub async fn start_session(&self) -> Result<String, JsValue> {
        let start = js_sys::Date::now();
        web_sys::console::log_1(&"[WASM] start_session: beginning".into());

        // Request new session
        web_sys::console::log_1(&"[WASM] start_session: calling post_json".into());
        let response: SessionResponse = self
            .http
            .post_json("/v1/session/new", &serde_json::json!({}))
            .await?;
        web_sys::console::log_1(&format!("[WASM] start_session: got response, session_id={}", response.session_id).into());

        let session_id = response.session_id.clone();

        // Parse session ID to bytes
        web_sys::console::log_1(&"[WASM] start_session: parsing session_id".into());
        let session_id_bytes = parse_session_id(&response.session_id)?;

        // Create OT receiver
        web_sys::console::log_1(&"[WASM] start_session: creating OT receiver".into());
        let receiver_ext = IknpOtExtension::new_client();
        web_sys::console::log_1(&"[WASM] start_session: created IknpOtExtension".into());
        let mut receiver = OtReceiver::new(receiver_ext);
        web_sys::console::log_1(&"[WASM] start_session: created OtReceiver".into());

        // Initialize receiver with session params
        let params = shardlm_protocol::OtSessionParamsPayload {
            accepted_lmax: response.max_prompt_len,
            accepted_d: response.embedding_dim as u16,
            vocab_size: response.vocab_size,
            value_type: 1, // i32 fixed-point
            fixed_point_scale: shardlm_fixed_point::DEFAULT_SCALE,
            row_layout: 1, // contiguous i32[d]
            selected_ot_suite: 0x0101, // IKNP
            suite_flags: 0,
            session_ttl_sec: 900,
            max_requests: 1000,
        };
        web_sys::console::log_1(&"[WASM] start_session: calling handle_session_params".into());
        receiver
            .handle_session_params(&params, session_id_bytes)
            .map_err(WasmError::from)?;
        web_sys::console::log_1(&"[WASM] start_session: handle_session_params done".into());

        // Update state
        {
            let mut state = self.state.borrow_mut();
            state.session_id = Some(session_id.clone());
            state.session_id_bytes = Some(session_id_bytes);
            state.vocab_size = response.vocab_size;
            state.embedding_dim = response.embedding_dim;
            state.max_prompt_len = response.max_prompt_len;
            state.receiver = Some(receiver);
            state.ttl_secs = response.ttl_secs;
            state.ttl_updated_at = js_sys::Date::now();
            state.timings.session_create_ms = js_sys::Date::now() - start;
        }
        web_sys::console::log_1(&format!("[WASM] start_session: done, ttl={}s", response.ttl_secs).into());

        Ok(session_id)
    }

    /// Perform base OT handshake
    #[wasm_bindgen]
    pub async fn base_ot_handshake(&self) -> Result<(), JsValue> {
        let start = js_sys::Date::now();
        web_sys::console::log_1(&"[WASM] base_ot_handshake: beginning".into());

        // Step 1: Send OtSessionInit to initialize server-side OT session
        let (init_request_bytes, session_id) = {
            web_sys::console::log_1(&"[WASM] base_ot_handshake: generating session init".into());
            let state = self.state.borrow();
            let session_id = state
                .session_id
                .clone()
                .ok_or(WasmError::SessionNotEstablished)?;
            let receiver = state
                .receiver
                .as_ref()
                .ok_or(WasmError::SessionNotEstablished)?;

            // Generate session init message
            let (header, init_payload) = receiver.generate_session_init();
            let payload = init_payload.encode();
            let frame = Frame::new(MsgType::OtSessionInit, header, payload);
            (frame.encode(), session_id)
        };

        web_sys::console::log_1(&format!("[WASM] base_ot_handshake: sending session init, {} bytes", init_request_bytes.len()).into());
        let init_response_bytes = self
            .http
            .post_binary("/v1/ot/base/init", &init_request_bytes, Some(&session_id))
            .await?;
        web_sys::console::log_1(&format!("[WASM] base_ot_handshake: got session params, {} bytes", init_response_bytes.len()).into());

        // Process OtSessionParams response (this updates our session config from server)
        {
            let mut state = self.state.borrow_mut();
            state.timings.bytes_sent += init_request_bytes.len() as u64;
            state.timings.bytes_received += init_response_bytes.len() as u64;

            let response_frame = Frame::decode(&init_response_bytes).map_err(WasmError::from)?;
            if response_frame.msg_type != MsgType::OtSessionParams {
                return Err(WasmError::InvalidResponse(format!(
                    "Expected OtSessionParams, got {:?}",
                    response_frame.msg_type
                ))
                .into());
            }

            // Note: We already have session params from /v1/session/new, but this
            // also initializes the server's OT session state machine
            web_sys::console::log_1(&"[WASM] base_ot_handshake: session init acknowledged".into());
        }

        // Step 2: Generate and send base OT message
        let (base_ot_request_bytes, session_id) = {
            web_sys::console::log_1(&"[WASM] base_ot_handshake: borrowing state for base OT".into());
            let mut state = self.state.borrow_mut();
            let session_id = state
                .session_id
                .clone()
                .ok_or(WasmError::SessionNotEstablished)?;
            web_sys::console::log_1(&"[WASM] base_ot_handshake: got session_id".into());
            let receiver = state
                .receiver
                .as_mut()
                .ok_or(WasmError::SessionNotEstablished)?;
            web_sys::console::log_1(&"[WASM] base_ot_handshake: got receiver".into());

            // Generate initial base OT message
            web_sys::console::log_1(&"[WASM] base_ot_handshake: calling generate_base_ot_init".into());
            let (header, base_ot_msg) = receiver.generate_base_ot_init().map_err(WasmError::from)?;
            web_sys::console::log_1(&"[WASM] base_ot_handshake: generate_base_ot_init done".into());

            // Encode to frame
            let payload = base_ot_msg.encode();
            let frame = Frame::new(MsgType::OtBaseOtMsg, header, payload);
            let request_bytes = frame.encode();
            web_sys::console::log_1(&format!("[WASM] base_ot_handshake: encoded frame, {} bytes", request_bytes.len()).into());

            state.timings.bytes_sent += request_bytes.len() as u64;

            (request_bytes, session_id)
        };

        // Send base OT to server (no borrow held here)
        web_sys::console::log_1(&"[WASM] base_ot_handshake: sending base OT to server".into());
        let response_bytes = self
            .http
            .post_binary("/v1/ot/base/init", &base_ot_request_bytes, Some(&session_id))
            .await?;
        web_sys::console::log_1(&format!("[WASM] base_ot_handshake: got base OT response, {} bytes", response_bytes.len()).into());

        // Process base OT response
        {
            let mut state = self.state.borrow_mut();
            state.timings.bytes_received += response_bytes.len() as u64;

            // Decode response
            web_sys::console::log_1(&"[WASM] base_ot_handshake: decoding response".into());
            let response_frame = Frame::decode(&response_bytes).map_err(WasmError::from)?;
            web_sys::console::log_1(&format!("[WASM] base_ot_handshake: response msg_type={:?}", response_frame.msg_type).into());

            if response_frame.msg_type != MsgType::OtBaseOtMsg {
                return Err(WasmError::InvalidResponse(format!(
                    "Expected OtBaseOtMsg, got {:?}",
                    response_frame.msg_type
                ))
                .into());
            }

            let response_msg =
                shardlm_protocol::OtBaseOtMsgPayload::decode(&response_frame.payload)
                    .map_err(WasmError::from)?;
            web_sys::console::log_1(&"[WASM] base_ot_handshake: decoded response payload".into());

            let receiver = state.receiver.as_mut().unwrap();

            // Process response (may generate more messages in multi-round OT)
            web_sys::console::log_1(&"[WASM] base_ot_handshake: calling handle_base_ot_response".into());
            let next_msg = receiver
                .handle_base_ot_response(&response_msg)
                .map_err(WasmError::from)?;
            web_sys::console::log_1(&"[WASM] base_ot_handshake: handle_base_ot_response done".into());

            // For now, we assume single-round simplified OT
            if next_msg.is_some() {
                web_sys::console::log_1(&"Multi-round base OT not yet implemented".into());
            }
        }

        // Mark session ready - need to send a proper OtSessionReady frame
        web_sys::console::log_1(&"[WASM] base_ot_handshake: marking session ready".into());
        let (ready_request_bytes, ready_session_id) = {
            let state = self.state.borrow();
            let session_id = state
                .session_id
                .clone()
                .ok_or(WasmError::SessionNotEstablished)?;
            let session_id_bytes = state
                .session_id_bytes
                .ok_or(WasmError::SessionNotEstablished)?;

            // Create a session ready request
            let ready_payload = shardlm_protocol::OtSessionReadyPayload::default();
            let header = shardlm_protocol::Header::new(session_id_bytes, 0);
            let frame = Frame::new(MsgType::OtSessionReady, header, ready_payload.encode());
            (frame.encode(), session_id)
        };

        let ready_response_bytes = self
            .http
            .post_binary("/v1/ot/ready", &ready_request_bytes, Some(&ready_session_id))
            .await?;
        web_sys::console::log_1(&format!("[WASM] base_ot_handshake: got ready response, {} bytes", ready_response_bytes.len()).into());

        // Process the ready response
        {
            let mut state = self.state.borrow_mut();
            let response_frame = Frame::decode(&ready_response_bytes).map_err(WasmError::from)?;

            if response_frame.msg_type != MsgType::OtSessionReady {
                return Err(WasmError::InvalidResponse(format!(
                    "Expected OtSessionReady, got {:?}",
                    response_frame.msg_type
                ))
                .into());
            }

            let ready_response =
                shardlm_protocol::OtSessionReadyPayload::decode(&response_frame.payload)
                    .map_err(WasmError::from)?;

            if !ready_response.ok {
                return Err(WasmError::InvalidResponse("Server rejected session ready".to_string()).into());
            }

            // Mark the receiver as ready
            let receiver = state.receiver.as_mut().unwrap();
            receiver.handle_session_ready(&ready_response).map_err(WasmError::from)?;

            state.timings.base_ot_ms = js_sys::Date::now() - start;
            state.timings.request_count += 1;
        }
        web_sys::console::log_1(&"[WASM] base_ot_handshake: done".into());

        Ok(())
    }

    /// Fetch embeddings for the given token IDs
    #[wasm_bindgen]
    pub async fn fetch_embeddings(&self, token_ids: Vec<u32>) -> Result<Vec<i32>, JsValue> {
        let start = js_sys::Date::now();

        // Generate request (minimize borrow scope)
        let (request_bytes, session_id, ctr) = {
            let mut state = self.state.borrow_mut();
            let session_id = state
                .session_id
                .clone()
                .ok_or(WasmError::SessionNotEstablished)?;
            let receiver = state
                .receiver
                .as_mut()
                .ok_or(WasmError::SessionNotEstablished)?;

            if !receiver.is_ready() {
                return Err(WasmError::SessionNotReady.into());
            }

            // Generate fetch request
            let (header, fetch_request) = receiver
                .generate_embed_fetch(&token_ids)
                .map_err(WasmError::from)?;

            let ctr = header.ctr;

            // Encode to frame
            let payload = fetch_request.encode();
            let frame = Frame::new(MsgType::EmbedFetchRequest, header, payload);
            let request_bytes = frame.encode();

            state.timings.bytes_sent += request_bytes.len() as u64;

            (request_bytes, session_id, ctr)
        };

        // Send to server (no borrow held)
        let response_bytes = self
            .http
            .post_binary("/v1/ot/embed/fetch", &request_bytes, Some(&session_id))
            .await?;

        // Process response
        let embeddings = {
            let mut state = self.state.borrow_mut();
            state.timings.bytes_received += response_bytes.len() as u64;

            // Decode response
            let response_frame = Frame::decode(&response_bytes).map_err(WasmError::from)?;

            if response_frame.msg_type != MsgType::EmbedFetchResponse {
                return Err(WasmError::InvalidResponse(format!(
                    "Expected EmbedFetchResponse, got {:?}",
                    response_frame.msg_type
                ))
                .into());
            }

            let fetch_response =
                shardlm_protocol::EmbedFetchResponsePayload::decode(&response_frame.payload)
                    .map_err(WasmError::from)?;

            let receiver = state.receiver.as_mut().unwrap();

            // Decode embeddings
            let embeddings_bytes = receiver
                .handle_embed_fetch_response(&fetch_response, ctr)
                .map_err(WasmError::from)?;

            // Convert bytes to i32 values
            let embeddings: Vec<i32> = embeddings_bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            state.timings.embed_fetch_ms += js_sys::Date::now() - start;
            state.timings.request_count += 1;

            embeddings
        };

        Ok(embeddings)
    }

    /// Get current session ID
    #[wasm_bindgen(getter)]
    pub fn session_id(&self) -> Option<String> {
        self.state.borrow().session_id.clone()
    }

    /// Get vocab size
    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> u32 {
        self.state.borrow().vocab_size
    }

    /// Get embedding dimension
    #[wasm_bindgen(getter)]
    pub fn embedding_dim(&self) -> u32 {
        self.state.borrow().embedding_dim
    }

    /// Get max prompt length
    #[wasm_bindgen(getter)]
    pub fn max_prompt_len(&self) -> u16 {
        self.state.borrow().max_prompt_len
    }

    /// Get current counter
    #[wasm_bindgen(getter)]
    pub fn counter(&self) -> u64 {
        self.state
            .borrow()
            .receiver
            .as_ref()
            .map(|r| r.counter())
            .unwrap_or(0)
    }

    /// Get performance timings
    #[wasm_bindgen(getter)]
    pub fn timings(&self) -> Timings {
        self.state.borrow().timings.clone()
    }

    /// Check if session is ready
    #[wasm_bindgen(getter)]
    pub fn is_ready(&self) -> bool {
        self.state
            .borrow()
            .receiver
            .as_ref()
            .map(|r| r.is_ready())
            .unwrap_or(false)
    }

    /// Check server health
    #[wasm_bindgen]
    pub async fn check_health(&self) -> Result<bool, JsValue> {
        let response: serde_json::Value = self.http.get_json("/health").await?;
        // Server returns {"status":"ok"} when healthy
        Ok(response.get("status").and_then(|s| s.as_str()) == Some("ok"))
    }

    /// Check server readiness
    #[wasm_bindgen]
    pub async fn check_ready(&self) -> Result<bool, JsValue> {
        let response: serde_json::Value = self.http.get_json("/ready").await?;
        Ok(response.get("ready").and_then(|r| r.as_bool()) == Some(true))
    }

    /// Get server info
    #[wasm_bindgen]
    pub async fn server_info(&self) -> Result<JsValue, JsValue> {
        let response: serde_json::Value = self.http.get_json("/v1/info").await?;
        serde_wasm_bindgen::to_value(&response).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Run inference on embeddings to get logits and top predictions
    ///
    /// Run secure inference on hidden state
    ///
    /// Protocol:
    /// 1. Client creates additive shares: H = H_c + H_s (where H is input)
    /// 2. Client sends both shares to server
    /// 3. Server computes logits_c = H_c @ W, logits_s = H_s @ W
    /// 4. Server sends both logit shares back
    /// 5. Client reconstructs: logits = logits_c + logits_s
    #[wasm_bindgen]
    pub async fn infer(&self, hidden_state: Vec<i32>) -> Result<JsValue, JsValue> {
        self.infer_with_position(hidden_state, 0).await
    }

    /// Run secure inference with explicit position for KV cache
    #[wasm_bindgen]
    pub async fn infer_with_position(&self, hidden_state: Vec<i32>, position: usize) -> Result<JsValue, JsValue> {
        let start = js_sys::Date::now();
        web_sys::console::log_1(
            &format!("[WASM] infer_with_position: starting with {} values at position {}", hidden_state.len(), position).into(),
        );

        let session_id = {
            let state = self.state.borrow();
            state.session_id.clone().ok_or(WasmError::SessionNotEstablished)?
        };

        // Create secret shares: H = H_c + H_s
        // Generate random server share, compute client share
        let hidden_server: Vec<i32> = (0..hidden_state.len())
            .map(|_| {
                let mut bytes = [0u8; 4];
                getrandom::getrandom(&mut bytes).unwrap();
                i32::from_le_bytes(bytes)
            })
            .collect();

        let hidden_client: Vec<i32> = hidden_state
            .iter()
            .zip(&hidden_server)
            .map(|(&h, &hs)| h.wrapping_sub(hs))
            .collect();

        web_sys::console::log_1(&"[WASM] infer_with_position: created secret shares".into());

        // Build the inference request with both shares and position
        let request = serde_json::json!({
            "session_id": session_id,
            "hidden_client": hidden_client,
            "hidden_server": hidden_server,
            "position": position,
            "run_transformer": true
        });

        // Call the inference endpoint
        let response: serde_json::Value = self
            .http
            .post_json("/v1/inference/forward", &request)
            .await?;

        web_sys::console::log_1(&"[WASM] infer_with_position: got response".into());

        // Update timings
        {
            let mut state = self.state.borrow_mut();
            state.timings.embed_fetch_ms += js_sys::Date::now() - start;
        }

        // Return the response as a JsValue
        // The response contains logits_client, logits_server, and top_tokens
        // top_tokens are already reconstructed by server for convenience
        serde_wasm_bindgen::to_value(&response).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get remaining TTL in seconds (estimated based on last update)
    #[wasm_bindgen(getter)]
    pub fn ttl_remaining(&self) -> i64 {
        let state = self.state.borrow();
        if state.ttl_secs == 0 {
            return 0;
        }
        let elapsed_ms = js_sys::Date::now() - state.ttl_updated_at;
        let elapsed_secs = (elapsed_ms / 1000.0) as i64;
        (state.ttl_secs - elapsed_secs).max(0)
    }

    /// Refresh the session TTL
    ///
    /// Call this periodically (e.g., every 5-10 minutes) to prevent session expiration.
    /// Returns the new TTL in seconds.
    #[wasm_bindgen]
    pub async fn refresh_session(&self) -> Result<i64, JsValue> {
        let session_id = {
            let state = self.state.borrow();
            state.session_id.clone().ok_or(WasmError::SessionNotEstablished)?
        };

        let request = serde_json::json!({
            "session_id": session_id
        });

        let response: SessionRefreshResponse = self
            .http
            .post_json("/v1/session/refresh", &request)
            .await?;

        // Update local TTL tracking
        {
            let mut state = self.state.borrow_mut();
            state.ttl_secs = response.ttl_secs;
            state.ttl_updated_at = js_sys::Date::now();
        }

        web_sys::console::log_1(&format!("[WASM] refresh_session: new ttl={}s", response.ttl_secs).into());

        Ok(response.ttl_secs)
    }

    /// Get session status without refreshing TTL
    #[wasm_bindgen]
    pub async fn session_status(&self) -> Result<JsValue, JsValue> {
        let session_id = {
            let state = self.state.borrow();
            state.session_id.clone().ok_or(WasmError::SessionNotEstablished)?
        };

        let request = serde_json::json!({
            "session_id": session_id
        });

        let response: SessionStatusResponse = self
            .http
            .post_json("/v1/session/status", &request)
            .await?;

        serde_wasm_bindgen::to_value(&response).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Check if session is about to expire (less than 60 seconds remaining)
    #[wasm_bindgen(getter)]
    pub fn session_expiring_soon(&self) -> bool {
        self.ttl_remaining() < 60
    }
}

/// Parse UUID string to bytes
fn parse_session_id(session_id: &str) -> Result<[u8; 16], WasmError> {
    let uuid = uuid::Uuid::parse_str(session_id)
        .map_err(|e| WasmError::InvalidResponse(format!("Invalid session ID: {}", e)))?;
    Ok(*uuid.as_bytes())
}
