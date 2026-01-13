//! OT protocol endpoints
//!
//! These endpoints handle the binary OT protocol messages:
//! - Base OT initialization (DH key exchange)
//! - Session ready confirmation
//! - Embedding fetch via OT extension

use axum::{
    body::Bytes,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use uuid::Uuid;

use shardlm_protocol::{Frame, Message};

use crate::error::{Result, ServerError};
use crate::state::AppState;

/// Extract session ID from headers
fn extract_session_id(headers: &HeaderMap) -> Result<Uuid> {
    let session_id = headers
        .get("x-session-id")
        .ok_or_else(|| ServerError::InvalidFrame("Missing X-Session-Id header".to_string()))?
        .to_str()
        .map_err(|_| ServerError::InvalidFrame("Invalid X-Session-Id header".to_string()))?;

    Uuid::parse_str(session_id)
        .map_err(|_| ServerError::InvalidFrame("Invalid session ID format".to_string()))
}

/// POST /v1/ot/base/init - Initialize base OT
///
/// Client sends SESSION_INIT, server responds with SESSION_PARAMS.
/// Then client sends BASE_OT_MSG (A points), server responds with BASE_OT_MSG (B points).
pub async fn base_ot_init(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse> {
    let session_id = extract_session_id(&headers)?;

    // Parse incoming frame
    let frame = Frame::decode(&body)
        .map_err(|e| ServerError::InvalidFrame(e.to_string()))?;

    let message = Message::from_frame(frame)
        .map_err(|e| ServerError::InvalidFrame(e.to_string()))?;

    // Get session and process message
    let session = state.sessions.get_mut(&session_id)?;

    let response_bytes = session.with_ot_sender(|sender| {
        match message {
            Message::OtSessionInit { header, payload } => {
                // Handle session init
                let (resp_header, resp_payload) = sender
                    .handle_session_init(&payload, header.client_nonce)?;

                // Set embedding database if loaded
                if let Some(ref embeddings) = *state.embeddings {
                    sender.set_embedding_db(embeddings.to_bytes());
                }

                let response = Message::OtSessionParams {
                    header: resp_header,
                    payload: resp_payload,
                };
                Ok(response.encode())
            }
            Message::OtBaseOtMsg { header, payload } => {
                // Handle base OT message (A points -> B points)
                if let Some((resp_header, resp_payload)) = sender
                    .handle_base_ot_msg(&payload, header.client_nonce)?
                {
                    let response = Message::OtBaseOtMsg {
                        header: resp_header,
                        payload: resp_payload,
                    };
                    Ok(response.encode())
                } else {
                    // No response needed (shouldn't happen in our protocol)
                    Err(ServerError::ProtocolError("Unexpected base OT state".to_string()))
                }
            }
            _ => Err(ServerError::InvalidFrame(format!(
                "Expected SESSION_INIT or BASE_OT_MSG, got {:?}",
                std::mem::discriminant(&message)
            ))),
        }
    })?;

    tracing::debug!(session_id = %session_id, "Base OT step completed");

    Ok((
        StatusCode::OK,
        [("content-type", "application/octet-stream")],
        response_bytes,
    ))
}

/// POST /v1/ot/ready - Mark session as ready
///
/// Client sends SESSION_READY, server confirms and marks session ready.
pub async fn ot_ready(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse> {
    let session_id = extract_session_id(&headers)?;

    // Parse incoming frame
    let frame = Frame::decode(&body)
        .map_err(|e| ServerError::InvalidFrame(e.to_string()))?;

    let message = Message::from_frame(frame)
        .map_err(|e| ServerError::InvalidFrame(e.to_string()))?;

    // Get session
    let mut session = state.sessions.get_mut(&session_id)?;

    let response_bytes = match message {
        Message::OtSessionReady { header: _, payload: _ } => {
            // Generate server's ready response
            let response_bytes = session.with_ot_sender(|sender| {
                let (resp_header, resp_payload) = sender
                    .generate_session_ready([0u8; 8])
                    .map_err(|e| ServerError::ProtocolError(e.to_string()))?;

                let response = Message::OtSessionReady {
                    header: resp_header,
                    payload: resp_payload,
                };
                Ok(response.encode())
            })?;

            // Mark session as ready
            session.mark_ready();

            response_bytes
        }
        _ => {
            return Err(ServerError::InvalidFrame(
                "Expected SESSION_READY message".to_string(),
            ));
        }
    };

    tracing::info!(session_id = %session_id, "Session ready");

    Ok((
        StatusCode::OK,
        [("content-type", "application/octet-stream")],
        response_bytes,
    ))
}

/// POST /v1/ot/embed/fetch - Fetch embeddings via OT
///
/// Client sends EMBED_FETCH_REQUEST with OT-masked indices,
/// server responds with EMBED_FETCH_RESPONSE containing masked embeddings.
pub async fn embed_fetch(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse> {
    let session_id = extract_session_id(&headers)?;

    // Check server is ready
    if !state.is_ready() {
        return Err(ServerError::ModelNotLoaded);
    }

    // Parse incoming frame
    let frame = Frame::decode(&body)
        .map_err(|e| ServerError::InvalidFrame(e.to_string()))?;

    let message = Message::from_frame(frame)
        .map_err(|e| ServerError::InvalidFrame(e.to_string()))?;

    // Get session
    let mut session = state.sessions.get_mut(&session_id)?;

    // Check session is ready
    if !session.ready {
        return Err(ServerError::SessionNotReady(session_id.to_string()));
    }

    let response_bytes = match message {
        Message::EmbedFetchRequest { header, payload } => {
            // Validate counter
            session.validate_counter(header.ctr)?;

            // Process fetch request
            session.with_ot_sender(|sender| {
                let (resp_header, resp_payload) = sender
                    .handle_embed_fetch(&payload, header.ctr, header.client_nonce)?;

                let response = Message::EmbedFetchResponse {
                    header: resp_header,
                    payload: resp_payload,
                };
                Ok(response.encode())
            })?
        }
        _ => {
            return Err(ServerError::InvalidFrame(
                "Expected EMBED_FETCH_REQUEST message".to_string(),
            ));
        }
    };

    tracing::debug!(
        session_id = %session_id,
        counter = session.expected_counter - 1,
        "Embed fetch completed"
    );

    Ok((
        StatusCode::OK,
        [("content-type", "application/octet-stream")],
        response_bytes,
    ))
}
