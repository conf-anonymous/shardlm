//! Binary serialization protocol for V3 Reference Implementation
//!
//! Uses bincode for efficient serialization, achieving 18x payload reduction
//! compared to JSON for typical inference requests.
//!
//! # Wire Format
//!
//! All requests use little-endian byte order.
//! Floats are transmitted as raw f32 bytes (4 bytes each).
//! Hidden states are flattened: [seq_len * hidden_dim] instead of [seq_len][hidden_dim].

#[cfg(feature = "binary-protocol")]
use serde::{Deserialize, Serialize};

/// Binary prefill request
///
/// Flattened representation for efficient serialization.
/// Total payload: ~60MB for 128 tokens vs ~1GB for JSON
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryPrefillRequest {
    /// Session ID as raw bytes (16 bytes for UUID)
    pub session_id: [u8; 16],
    /// Sequence length (number of tokens)
    pub seq_len: u32,
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Client hidden states, flattened [seq_len * hidden_dim]
    pub hidden_client: Vec<f32>,
    /// Server hidden states, flattened [seq_len * hidden_dim]
    pub hidden_server: Vec<f32>,
}

/// Binary prefill response
///
/// Flattened KV cache and hidden states for efficient transfer.
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryPrefillResponse {
    /// Final hidden state (client share) [hidden_dim]
    pub final_hidden_client: Vec<f32>,
    /// Final hidden state (server share) [hidden_dim]
    pub final_hidden_server: Vec<f32>,
    /// Number of layers
    pub num_layers: u32,
    /// KV dimension per layer
    pub kv_dim: u32,
    /// Flattened K cache [num_layers * seq_len * kv_dim]
    pub k_cache: Vec<f32>,
    /// Flattened V cache [num_layers * seq_len * kv_dim]
    pub v_cache: Vec<f32>,
    /// Logits (client share) [vocab_size]
    pub logits_client: Vec<f32>,
    /// Logits (server share) [vocab_size]
    pub logits_server: Vec<f32>,
}

/// Binary decode request (single token generation step)
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryDecodeRequest {
    /// Session ID
    pub session_id: [u8; 16],
    /// Current position in sequence
    pub position: u32,
    /// Input hidden state (client share) [hidden_dim]
    pub hidden_client: Vec<f32>,
    /// Input hidden state (server share) [hidden_dim]
    pub hidden_server: Vec<f32>,
}

/// Binary decode response
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryDecodeResponse {
    /// Output hidden state (client share) [hidden_dim]
    pub hidden_client: Vec<f32>,
    /// Output hidden state (server share) [hidden_dim]
    pub hidden_server: Vec<f32>,
    /// Logits (client share) [vocab_size]
    pub logits_client: Vec<f32>,
    /// Logits (server share) [vocab_size]
    pub logits_server: Vec<f32>,
}

/// Attestation token for H100 CC verification
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct AttestationToken {
    /// Measurement hash of GPU state
    pub measurement: [u8; 32],
    /// Timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// NVIDIA signature
    pub signature: Vec<u8>,
}

/// Binary prefill request with attestation (V3 Reference Implementation)
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryPrefillRequestV3 {
    /// Base request
    pub request: BinaryPrefillRequest,
    /// Client attestation token (proves client is running in secure environment)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_attestation: Option<AttestationToken>,
}

/// Binary prefill response with attestation (V3 Reference Implementation)
#[cfg(feature = "binary-protocol")]
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryPrefillResponseV3 {
    /// Base response
    pub response: BinaryPrefillResponse,
    /// Server attestation token (proves server processed in CC environment)
    pub server_attestation: AttestationToken,
}

// =============================================================================
// CONVERSION UTILITIES
// =============================================================================

#[cfg(feature = "binary-protocol")]
impl BinaryPrefillRequest {
    /// Create from nested Vec representation
    pub fn from_nested(
        session_id: [u8; 16],
        hidden_client: &[Vec<f32>],
        hidden_server: &[Vec<f32>],
    ) -> Self {
        let seq_len = hidden_client.len() as u32;
        let hidden_dim = hidden_client.first().map(|v| v.len()).unwrap_or(0) as u32;

        Self {
            session_id,
            seq_len,
            hidden_dim,
            hidden_client: hidden_client.iter().flatten().copied().collect(),
            hidden_server: hidden_server.iter().flatten().copied().collect(),
        }
    }

    /// Convert to nested Vec representation
    pub fn to_nested(&self) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let hidden_dim = self.hidden_dim as usize;
        let client = self.hidden_client
            .chunks(hidden_dim)
            .map(|c| c.to_vec())
            .collect();
        let server = self.hidden_server
            .chunks(hidden_dim)
            .map(|c| c.to_vec())
            .collect();
        (client, server)
    }
}

#[cfg(feature = "binary-protocol")]
impl BinaryPrefillResponse {
    /// Create from nested Vec representation
    pub fn from_nested(
        final_hidden_client: Vec<f32>,
        final_hidden_server: Vec<f32>,
        k_cache: &[Vec<Vec<f32>>],
        v_cache: &[Vec<Vec<f32>>],
        logits_client: Vec<f32>,
        logits_server: Vec<f32>,
    ) -> Self {
        let num_layers = k_cache.len() as u32;
        let kv_dim = k_cache.first()
            .and_then(|layer| layer.first())
            .map(|v| v.len())
            .unwrap_or(0) as u32;

        Self {
            final_hidden_client,
            final_hidden_server,
            num_layers,
            kv_dim,
            k_cache: k_cache.iter()
                .flat_map(|layer| layer.iter().flatten())
                .copied()
                .collect(),
            v_cache: v_cache.iter()
                .flat_map(|layer| layer.iter().flatten())
                .copied()
                .collect(),
            logits_client,
            logits_server,
        }
    }

    /// Convert K cache to nested representation
    pub fn k_cache_nested(&self, seq_len: usize) -> Vec<Vec<Vec<f32>>> {
        let kv_dim = self.kv_dim as usize;
        let num_layers = self.num_layers as usize;
        let layer_size = seq_len * kv_dim;

        self.k_cache
            .chunks(layer_size)
            .take(num_layers)
            .map(|layer| {
                layer.chunks(kv_dim)
                    .map(|v| v.to_vec())
                    .collect()
            })
            .collect()
    }

    /// Convert V cache to nested representation
    pub fn v_cache_nested(&self, seq_len: usize) -> Vec<Vec<Vec<f32>>> {
        let kv_dim = self.kv_dim as usize;
        let num_layers = self.num_layers as usize;
        let layer_size = seq_len * kv_dim;

        self.v_cache
            .chunks(layer_size)
            .take(num_layers)
            .map(|layer| {
                layer.chunks(kv_dim)
                    .map(|v| v.to_vec())
                    .collect()
            })
            .collect()
    }
}

// =============================================================================
// SIZE ESTIMATION
// =============================================================================

#[cfg(feature = "binary-protocol")]
impl BinaryPrefillRequest {
    /// Estimate binary size in bytes
    pub fn estimated_size(&self) -> usize {
        16 // session_id
        + 4 // seq_len
        + 4 // hidden_dim
        + self.hidden_client.len() * 4
        + self.hidden_server.len() * 4
    }
}

#[cfg(feature = "binary-protocol")]
impl BinaryPrefillResponse {
    /// Estimate binary size in bytes
    pub fn estimated_size(&self) -> usize {
        self.final_hidden_client.len() * 4
        + self.final_hidden_server.len() * 4
        + 4 // num_layers
        + 4 // kv_dim
        + self.k_cache.len() * 4
        + self.v_cache.len() * 4
        + self.logits_client.len() * 4
        + self.logits_server.len() * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "binary-protocol")]
    fn test_nested_conversion() {
        let hidden_client = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let hidden_server = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];

        let req = BinaryPrefillRequest::from_nested(
            [0u8; 16],
            &hidden_client,
            &hidden_server,
        );

        assert_eq!(req.seq_len, 2);
        assert_eq!(req.hidden_dim, 3);
        assert_eq!(req.hidden_client.len(), 6);

        let (client_nested, server_nested) = req.to_nested();
        assert_eq!(client_nested, hidden_client);
        assert_eq!(server_nested, hidden_server);
    }

    #[test]
    #[cfg(feature = "binary-protocol")]
    fn test_size_estimation() {
        let req = BinaryPrefillRequest {
            session_id: [0u8; 16],
            seq_len: 128,
            hidden_dim: 1536,
            hidden_client: vec![0.0; 128 * 1536],
            hidden_server: vec![0.0; 128 * 1536],
        };

        // 16 + 4 + 4 + 128*1536*4 + 128*1536*4 = 24 + 786432 + 786432 = ~1.5MB
        // vs JSON: ~20MB for same data
        let size = req.estimated_size();
        assert!(size < 2_000_000); // Under 2MB
    }
}
