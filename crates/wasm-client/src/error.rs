//! WASM client errors

use thiserror::Error;
use wasm_bindgen::prelude::*;

#[derive(Debug, Error)]
pub enum WasmError {
    #[error("Session not established")]
    SessionNotEstablished,

    #[error("Session not ready - base OT incomplete")]
    SessionNotReady,

    #[error("Network error: {0}")]
    Network(String),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("OT error: {0}")]
    Ot(String),

    #[error("Counter mismatch: expected {expected}, got {got}")]
    CounterMismatch { expected: u64, got: u64 },
}

impl From<WasmError> for JsValue {
    fn from(err: WasmError) -> Self {
        JsValue::from_str(&err.to_string())
    }
}

impl From<shardlm_ot::OtError> for WasmError {
    fn from(err: shardlm_ot::OtError) -> Self {
        WasmError::Ot(err.to_string())
    }
}

impl From<shardlm_protocol::ProtocolError> for WasmError {
    fn from(err: shardlm_protocol::ProtocolError) -> Self {
        WasmError::Protocol(err.to_string())
    }
}

impl From<serde_json::Error> for WasmError {
    fn from(err: serde_json::Error) -> Self {
        WasmError::Serialization(err.to_string())
    }
}
