//! ShardLM WebAssembly Client SDK
//!
//! Browser-side client for secure LLM inference via Oblivious Transfer.
//! This crate compiles to WASM and is imported by the Next.js frontend.

mod client;
mod error;
mod transport;
mod timings;

pub use client::ShardLmClient;
pub use error::WasmError;
pub use timings::Timings;

use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages in browser console
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get SDK version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
