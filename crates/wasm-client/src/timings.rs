//! Performance timing utilities

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Performance timings for debugging
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct Timings {
    /// Session creation time (ms)
    pub session_create_ms: f64,

    /// Base OT handshake time (ms)
    pub base_ot_ms: f64,

    /// Embedding fetch time (ms)
    pub embed_fetch_ms: f64,

    /// Total bytes sent
    pub bytes_sent: u64,

    /// Total bytes received
    pub bytes_received: u64,

    /// Number of requests made
    pub request_count: u32,
}

#[wasm_bindgen]
impl Timings {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total time
    #[wasm_bindgen(getter)]
    pub fn total_ms(&self) -> f64 {
        self.session_create_ms + self.base_ot_ms + self.embed_fetch_ms
    }

    /// Reset all timings
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Get as JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}
