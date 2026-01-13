//! HTTP transport layer for WASM client

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Headers, Request, RequestInit, RequestMode, Response};

use crate::error::WasmError;

/// HTTP client for communicating with ShardLM server
pub struct HttpClient {
    base_url: String,
}

impl HttpClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// POST JSON and get JSON response
    pub async fn post_json<T: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &T,
    ) -> Result<R, WasmError> {
        let url = format!("{}{}", self.base_url, path);
        let body_str = serde_json::to_string(body)?;

        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&JsValue::from_str(&body_str));

        let headers = Headers::new().map_err(|e| WasmError::Network(format!("{:?}", e)))?;
        headers
            .set("Content-Type", "application/json")
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;
        opts.set_headers(&headers);

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        let window = web_sys::window().ok_or_else(|| WasmError::Network("No window".to_string()))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| WasmError::InvalidResponse("Not a Response".to_string()))?;

        if !resp.ok() {
            return Err(WasmError::Server(format!(
                "HTTP {}: {}",
                resp.status(),
                resp.status_text()
            )));
        }

        let json = JsFuture::from(resp.json().map_err(|e| WasmError::Network(format!("{:?}", e)))?)
            .await
            .map_err(|e| WasmError::InvalidResponse(format!("{:?}", e)))?;

        serde_wasm_bindgen::from_value(json).map_err(|e| WasmError::Serialization(e.to_string()))
    }

    /// POST binary and get binary response
    pub async fn post_binary(
        &self,
        path: &str,
        body: &[u8],
        session_id: Option<&str>,
    ) -> Result<Vec<u8>, WasmError> {
        let url = format!("{}{}", self.base_url, path);

        // Convert body to Uint8Array
        let body_array = js_sys::Uint8Array::new_with_length(body.len() as u32);
        body_array.copy_from(body);

        let body_js: JsValue = body_array.into();
        let opts = RequestInit::new();
        opts.set_method("POST");
        opts.set_mode(RequestMode::Cors);
        opts.set_body(&body_js);

        let headers = Headers::new().map_err(|e| WasmError::Network(format!("{:?}", e)))?;
        headers
            .set("Content-Type", "application/octet-stream")
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        if let Some(sid) = session_id {
            headers
                .set("X-Session-Id", sid)
                .map_err(|e| WasmError::Network(format!("{:?}", e)))?;
        }

        opts.set_headers(&headers);

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        let window = web_sys::window().ok_or_else(|| WasmError::Network("No window".to_string()))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| WasmError::InvalidResponse("Not a Response".to_string()))?;

        if !resp.ok() {
            return Err(WasmError::Server(format!(
                "HTTP {}: {}",
                resp.status(),
                resp.status_text()
            )));
        }

        let buffer = JsFuture::from(
            resp.array_buffer()
                .map_err(|e| WasmError::Network(format!("{:?}", e)))?,
        )
        .await
        .map_err(|e| WasmError::InvalidResponse(format!("{:?}", e)))?;

        let array = js_sys::Uint8Array::new(&buffer);
        Ok(array.to_vec())
    }

    /// GET JSON response
    pub async fn get_json<R: serde::de::DeserializeOwned>(
        &self,
        path: &str,
    ) -> Result<R, WasmError> {
        let url = format!("{}{}", self.base_url, path);

        let opts = RequestInit::new();
        opts.set_method("GET");
        opts.set_mode(RequestMode::Cors);

        let request = Request::new_with_str_and_init(&url, &opts)
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        let window = web_sys::window().ok_or_else(|| WasmError::Network("No window".to_string()))?;
        let resp_value = JsFuture::from(window.fetch_with_request(&request))
            .await
            .map_err(|e| WasmError::Network(format!("{:?}", e)))?;

        let resp: Response = resp_value
            .dyn_into()
            .map_err(|_| WasmError::InvalidResponse("Not a Response".to_string()))?;

        if !resp.ok() {
            return Err(WasmError::Server(format!(
                "HTTP {}: {}",
                resp.status(),
                resp.status_text()
            )));
        }

        let json = JsFuture::from(resp.json().map_err(|e| WasmError::Network(format!("{:?}", e)))?)
            .await
            .map_err(|e| WasmError::InvalidResponse(format!("{:?}", e)))?;

        serde_wasm_bindgen::from_value(json).map_err(|e| WasmError::Serialization(e.to_string()))
    }
}
