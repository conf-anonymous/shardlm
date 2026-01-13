//! Secret-shared KV cache for private inference
//!
//! The KV cache stores key and value projections in secret-shared form:
//! - Server holds K_s, V_s shares
//! - Client holds K_c, V_c shares
//! - Neither party can reconstruct full K or V without the other's shares
//!
//! This enables private autoregressive decoding where the server never sees
//! the actual key/value vectors that would reveal token semantics.

use shardlm_fixed_point::FixedVector;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{Result, SharingError};
use crate::share::Share;

/// Client's view of the KV cache (holds client shares)
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KvCacheClient {
    /// K shares per layer: [num_layers][seq_len][num_kv_heads * head_dim]
    k_shares: Vec<Vec<Share>>,
    /// V shares per layer: [num_layers][seq_len][num_kv_heads * head_dim]
    v_shares: Vec<Vec<Share>>,
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Current sequence length
    seq_len: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Fixed-point scale
    scale: u8,
}

/// Server's view of the KV cache (holds server shares)
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KvCacheServer {
    /// K shares per layer: [num_layers][seq_len][num_kv_heads * head_dim]
    k_shares: Vec<Vec<Share>>,
    /// V shares per layer: [num_layers][seq_len][num_kv_heads * head_dim]
    v_shares: Vec<Vec<Share>>,
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Current sequence length
    seq_len: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Fixed-point scale
    scale: u8,
}

impl KvCacheClient {
    /// Create a new empty KV cache for the client
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        scale: u8,
    ) -> Self {
        Self {
            k_shares: (0..num_layers).map(|_| Vec::with_capacity(max_seq_len)).collect(),
            v_shares: (0..num_layers).map(|_| Vec::with_capacity(max_seq_len)).collect(),
            num_layers,
            num_kv_heads,
            head_dim,
            seq_len: 0,
            max_seq_len,
            scale,
        }
    }

    /// Append K and V shares for a new token at a specific layer
    pub fn append(&mut self, layer: usize, k_share: Share, v_share: Share) -> Result<()> {
        if layer >= self.num_layers {
            return Err(SharingError::DimensionMismatch {
                expected: self.num_layers,
                got: layer + 1,
            });
        }
        if self.seq_len >= self.max_seq_len {
            return Err(SharingError::DimensionMismatch {
                expected: self.max_seq_len,
                got: self.seq_len + 1,
            });
        }

        let expected_len = self.num_kv_heads * self.head_dim;
        if k_share.len() != expected_len || v_share.len() != expected_len {
            return Err(SharingError::DimensionMismatch {
                expected: expected_len,
                got: k_share.len(),
            });
        }

        self.k_shares[layer].push(k_share);
        self.v_shares[layer].push(v_share);

        // Update seq_len only after first layer append
        if layer == 0 {
            self.seq_len += 1;
        }

        Ok(())
    }

    /// Fast append without dimension checks - use when dimensions are pre-validated
    ///
    /// This avoids repeated dimension checks in hot loops where the caller has
    /// already validated that layer/seq_len/share dimensions are correct.
    #[inline]
    pub fn append_unchecked(&mut self, layer: usize, k_share: Share, v_share: Share) {
        self.k_shares[layer].push(k_share);
        self.v_shares[layer].push(v_share);
        if layer == 0 {
            self.seq_len += 1;
        }
    }

    /// Get K shares for a layer (all positions)
    #[inline]
    pub fn get_k(&self, layer: usize) -> Option<&[Share]> {
        self.k_shares.get(layer).map(|v| v.as_slice())
    }

    /// Get V shares for a layer (all positions)
    #[inline]
    pub fn get_v(&self, layer: usize) -> Option<&[Share]> {
        self.v_shares.get(layer).map(|v| v.as_slice())
    }

    /// Current sequence length
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for k in &mut self.k_shares {
            k.clear();
        }
        for v in &mut self.v_shares {
            v.clear();
        }
        self.seq_len = 0;
    }
}

impl KvCacheServer {
    /// Create a new empty KV cache for the server
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        scale: u8,
    ) -> Self {
        Self {
            k_shares: (0..num_layers).map(|_| Vec::with_capacity(max_seq_len)).collect(),
            v_shares: (0..num_layers).map(|_| Vec::with_capacity(max_seq_len)).collect(),
            num_layers,
            num_kv_heads,
            head_dim,
            seq_len: 0,
            max_seq_len,
            scale,
        }
    }

    /// Append K and V shares for a new token at a specific layer
    pub fn append(&mut self, layer: usize, k_share: Share, v_share: Share) -> Result<()> {
        if layer >= self.num_layers {
            return Err(SharingError::DimensionMismatch {
                expected: self.num_layers,
                got: layer + 1,
            });
        }
        if self.seq_len >= self.max_seq_len {
            return Err(SharingError::DimensionMismatch {
                expected: self.max_seq_len,
                got: self.seq_len + 1,
            });
        }

        let expected_len = self.num_kv_heads * self.head_dim;
        if k_share.len() != expected_len || v_share.len() != expected_len {
            return Err(SharingError::DimensionMismatch {
                expected: expected_len,
                got: k_share.len(),
            });
        }

        self.k_shares[layer].push(k_share);
        self.v_shares[layer].push(v_share);

        if layer == 0 {
            self.seq_len += 1;
        }

        Ok(())
    }

    /// Fast append without dimension checks - use when dimensions are pre-validated
    #[inline]
    pub fn append_unchecked(&mut self, layer: usize, k_share: Share, v_share: Share) {
        self.k_shares[layer].push(k_share);
        self.v_shares[layer].push(v_share);
        if layer == 0 {
            self.seq_len += 1;
        }
    }

    /// Get K shares for a layer
    #[inline]
    pub fn get_k(&self, layer: usize) -> Option<&[Share]> {
        self.k_shares.get(layer).map(|v| v.as_slice())
    }

    /// Get V shares for a layer
    #[inline]
    pub fn get_v(&self, layer: usize) -> Option<&[Share]> {
        self.v_shares.get(layer).map(|v| v.as_slice())
    }

    /// Current sequence length
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        for k in &mut self.k_shares {
            k.clear();
        }
        for v in &mut self.v_shares {
            v.clear();
        }
        self.seq_len = 0;
    }
}

/// Split a plaintext K or V vector into secret shares for client and server
pub fn share_kv(plaintext: &FixedVector) -> (Share, Share) {
    let server_share = Share::random(plaintext.len(), plaintext.scale);
    let client_data: Vec<i32> = plaintext
        .data
        .iter()
        .zip(&server_share.data)
        .map(|(&x, &s)| x.wrapping_sub(s))
        .collect();
    let client_share = Share::from_raw(client_data, plaintext.scale);
    (client_share, server_share)
}

/// Reconstruct K or V from client and server shares
pub fn reconstruct_kv(client: &Share, server: &Share) -> Result<FixedVector> {
    if client.len() != server.len() {
        return Err(SharingError::DimensionMismatch {
            expected: client.len(),
            got: server.len(),
        });
    }
    if client.scale != server.scale {
        return Err(SharingError::ScaleMismatch {
            expected: client.scale,
            got: server.scale,
        });
    }

    let data: Vec<i32> = client
        .data
        .iter()
        .zip(&server.data)
        .map(|(&c, &s)| c.wrapping_add(s))
        .collect();

    Ok(FixedVector::from_raw(data, client.scale))
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_kv_cache_append() {
        let mut client_cache = KvCacheClient::new(2, 4, 64, 128, DEFAULT_SCALE);
        let mut server_cache = KvCacheServer::new(2, 4, 64, 128, DEFAULT_SCALE);

        // Create dummy K and V shares
        let k = FixedVector::from_raw(vec![1; 256], DEFAULT_SCALE); // 4 heads * 64 dim
        let v = FixedVector::from_raw(vec![2; 256], DEFAULT_SCALE);

        let (k_client, k_server) = share_kv(&k);
        let (v_client, v_server) = share_kv(&v);

        // Append to layer 0
        client_cache.append(0, k_client.clone(), v_client.clone()).unwrap();
        server_cache.append(0, k_server.clone(), v_server.clone()).unwrap();

        assert_eq!(client_cache.seq_len(), 1);
        assert_eq!(server_cache.seq_len(), 1);

        // Verify reconstruction
        let k_client_shares = client_cache.get_k(0).unwrap();
        let k_server_shares = server_cache.get_k(0).unwrap();
        let reconstructed = reconstruct_kv(&k_client_shares[0], &k_server_shares[0]).unwrap();
        assert_eq!(reconstructed.data, k.data);
    }

    #[test]
    fn test_share_kv_roundtrip() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, -0.5, 2.5, 3.14]).unwrap();
        let (client, server) = share_kv(&plaintext);

        // Neither share equals plaintext
        assert_ne!(client.data, plaintext.data);
        assert_ne!(server.data, plaintext.data);

        // Reconstruction matches
        let reconstructed = reconstruct_kv(&client, &server).unwrap();
        assert_eq!(reconstructed.data, plaintext.data);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut client_cache = KvCacheClient::new(2, 4, 64, 128, DEFAULT_SCALE);

        let k = FixedVector::from_raw(vec![1; 256], DEFAULT_SCALE);
        let v = FixedVector::from_raw(vec![2; 256], DEFAULT_SCALE);
        let (k_c, _) = share_kv(&k);
        let (v_c, _) = share_kv(&v);

        client_cache.append(0, k_c, v_c).unwrap();
        assert_eq!(client_cache.seq_len(), 1);

        client_cache.clear();
        assert_eq!(client_cache.seq_len(), 0);
    }
}
