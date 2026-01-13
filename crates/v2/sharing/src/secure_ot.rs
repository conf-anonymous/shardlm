//! Secure OT integration for embedding retrieval
//!
//! This module provides integration with the v1 OT crate for privacy-preserving
//! embedding retrieval. The client retrieves embeddings without revealing
//! which token IDs were requested.
//!
//! # Security Guarantee
//!
//! **The server NEVER learns which embeddings were requested.**
//! - Client sends encrypted indices via OT extension
//! - Server masks all responses with per-row keys
//! - Only client can unmask the requested rows
//!
//! # Usage Flow
//!
//! 1. Session establishment (base OT handshake)
//! 2. Client generates OT query for token IDs
//! 3. Server processes query and returns masked embeddings
//! 4. Client decodes to get embeddings as shares

use crate::error::{Result, SharingError};
use crate::secure::{ClientShare, SecureSharePair, ServerShare};

/// Result of OT embedding fetch (client side)
///
/// Contains the retrieved embeddings split into shares for secure computation.
pub struct OtEmbeddingResult {
    /// Client's share of embeddings [num_tokens × embedding_dim]
    pub client_shares: Vec<ClientShare<f32>>,
    /// Server's share of embeddings (to be sent to server via secure channel)
    pub server_shares: Vec<ServerShare<f32>>,
    /// Embedding dimension
    pub embedding_dim: usize,
}

impl OtEmbeddingResult {
    /// Create embedding result from raw bytes (after OT decode)
    ///
    /// The raw embeddings are split into shares for secure computation.
    pub fn from_raw_embeddings<R: rand::Rng>(
        raw_embeddings: &[u8],
        num_tokens: usize,
        embedding_dim: usize,
        rng: &mut R,
    ) -> Result<Self>
    where
        rand::distributions::Standard: rand::distributions::Distribution<f32>,
    {
        let bytes_per_embedding = embedding_dim * 4; // f32 = 4 bytes
        if raw_embeddings.len() != num_tokens * bytes_per_embedding {
            return Err(SharingError::ShapeMismatch {
                expected: vec![num_tokens * bytes_per_embedding],
                got: vec![raw_embeddings.len()],
            });
        }

        let mut client_shares = Vec::with_capacity(num_tokens);
        let mut server_shares = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let start = t * bytes_per_embedding;
            let end = start + bytes_per_embedding;
            let embedding_bytes = &raw_embeddings[start..end];

            // Convert bytes to f32 vector
            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(bytes)
                })
                .collect();

            // Split into shares
            let share_pair = SecureSharePair::from_plaintext(embedding, vec![embedding_dim], rng);
            let (client, server) = share_pair.take_server_share();

            client_shares.push(client);
            server_shares.push(server);
        }

        Ok(Self {
            client_shares,
            server_shares,
            embedding_dim,
        })
    }

    /// Sum embeddings to get initial hidden state (client side)
    ///
    /// Returns share pair of the summed embeddings.
    pub fn sum_embeddings(&self) -> Result<SecureSharePair<f32>> {
        if self.client_shares.is_empty() {
            return Err(SharingError::InvalidShare("No embeddings to sum".to_string()));
        }

        let dim = self.embedding_dim;

        // Sum client shares
        let mut client_sum = vec![0.0f32; dim];
        for share in &self.client_shares {
            for (i, &val) in share.data().iter().enumerate() {
                client_sum[i] += val;
            }
        }

        // Sum server shares
        let mut server_sum = vec![0.0f32; dim];
        for share in &self.server_shares {
            for (i, &val) in share.data().iter().enumerate() {
                server_sum[i] += val;
            }
        }

        // Create share pair
        // Note: We're constructing this directly since we already have separated shares
        Ok(SecureSharePair::from_existing_shares(
            client_sum,
            server_sum,
            vec![dim],
        ))
    }
}

/// Extension trait for SecureSharePair to allow construction from existing shares
impl SecureSharePair<f32> {
    /// Create from pre-computed shares (used after OT)
    ///
    /// # Security
    ///
    /// This is only safe when called on the CLIENT side with shares that
    /// were properly generated (e.g., from OT protocol).
    pub fn from_existing_shares(
        client_data: Vec<f32>,
        server_data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Self {
        // Use crate-internal constructors
        let client = ClientShare::new(client_data, shape.clone());
        let server = ServerShare::new(server_data, shape);

        Self::from_shares(client, server)
    }
}

/// OT session state for server
///
/// Wraps the v1 OT sender to manage embedding database and queries.
pub struct SecureOtServer {
    /// Embedding database [vocab_size × embedding_dim] as bytes
    embedding_db: Vec<u8>,
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embedding_dim: usize,
    /// Row bytes (embedding_dim * 4 for f32)
    row_bytes: usize,
}

impl SecureOtServer {
    /// Create OT server with embedding database
    pub fn new(embedding_db: Vec<u8>, vocab_size: usize, embedding_dim: usize) -> Result<Self> {
        let row_bytes = embedding_dim * 4;
        if embedding_db.len() != vocab_size * row_bytes {
            return Err(SharingError::ShapeMismatch {
                expected: vec![vocab_size * row_bytes],
                got: vec![embedding_db.len()],
            });
        }

        Ok(Self {
            embedding_db,
            vocab_size,
            embedding_dim,
            row_bytes,
        })
    }

    /// Get embedding database reference (for OT sender)
    pub fn embedding_db(&self) -> &[u8] {
        &self.embedding_db
    }

    /// Get row bytes
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// Get vocab size
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_embedding_to_shares() {
        let embedding_dim = 4;
        let num_tokens = 2;

        // Create fake embeddings (2 tokens × 4 dims)
        let embeddings: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // Token 0
            5.0, 6.0, 7.0, 8.0, // Token 1
        ];

        // Convert to bytes
        let raw_bytes: Vec<u8> = embeddings
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let result = OtEmbeddingResult::from_raw_embeddings(
            &raw_bytes,
            num_tokens,
            embedding_dim,
            &mut rng,
        ).unwrap();

        assert_eq!(result.client_shares.len(), num_tokens);
        assert_eq!(result.server_shares.len(), num_tokens);

        // Verify reconstruction works
        for t in 0..num_tokens {
            let client = result.client_shares[t].data();
            let server = result.server_shares[t].data();

            let reconstructed: Vec<f32> = client.iter()
                .zip(server.iter())
                .map(|(c, s)| c + s)
                .collect();

            let expected = &embeddings[t * embedding_dim..(t + 1) * embedding_dim];
            for (r, e) in reconstructed.iter().zip(expected.iter()) {
                assert!((r - e).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_sum_embeddings() {
        let embedding_dim = 4;
        let num_tokens = 2;

        let embeddings: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ];

        let raw_bytes: Vec<u8> = embeddings
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let result = OtEmbeddingResult::from_raw_embeddings(
            &raw_bytes,
            num_tokens,
            embedding_dim,
            &mut rng,
        ).unwrap();

        let sum_pair = result.sum_embeddings().unwrap();
        let sum = sum_pair.reconstruct();

        // Expected: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        for (s, e) in sum.iter().zip(expected.iter()) {
            assert!((s - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_shares_are_different() {
        let embedding_dim = 4;
        let embeddings: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let raw_bytes: Vec<u8> = embeddings.iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let result = OtEmbeddingResult::from_raw_embeddings(
            &raw_bytes,
            1,
            embedding_dim,
            &mut rng,
        ).unwrap();

        // Shares should not equal plaintext
        let client = result.client_shares[0].data();
        let server = result.server_shares[0].data();

        assert_ne!(client, &embeddings[..]);
        assert_ne!(server, &embeddings[..]);
    }
}
