//! Embedding table management

use shardlm_fixed_point::FixedVector;

use crate::error::{ModelError, Result};

/// Embedding table (V × d) stored as fixed-point i32
#[derive(Debug, Clone)]
pub struct EmbeddingTable {
    /// Raw embedding data (V × d) in row-major order
    pub data: Vec<i32>,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Fixed-point scale
    pub scale: u8,
}

impl EmbeddingTable {
    /// Create from f32 embeddings (row-major)
    pub fn from_f32(embeddings: &[f32], vocab_size: usize, embed_dim: usize, scale: u8) -> Result<Self> {
        if embeddings.len() != vocab_size * embed_dim {
            return Err(ModelError::InvalidShape {
                expected: vec![vocab_size, embed_dim],
                got: vec![embeddings.len()],
            });
        }

        let scale_factor = (1u64 << scale) as f64;
        let data: Vec<i32> = embeddings
            .iter()
            .map(|&x| (x as f64 * scale_factor).round() as i32)
            .collect();

        Ok(Self {
            data,
            vocab_size,
            embed_dim,
            scale,
        })
    }

    /// Create from raw i32 data
    pub fn from_raw(data: Vec<i32>, vocab_size: usize, embed_dim: usize, scale: u8) -> Result<Self> {
        if data.len() != vocab_size * embed_dim {
            return Err(ModelError::InvalidShape {
                expected: vec![vocab_size, embed_dim],
                got: vec![data.len()],
            });
        }

        Ok(Self {
            data,
            vocab_size,
            embed_dim,
            scale,
        })
    }

    /// Create random embeddings (for testing)
    pub fn random(vocab_size: usize, embed_dim: usize, scale: u8) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize with small random values
        let std_dev = 0.02;
        let scale_factor = (1u64 << scale) as f64;
        let data: Vec<i32> = (0..vocab_size * embed_dim)
            .map(|_| {
                let val: f64 = rng.gen_range(-std_dev..std_dev);
                (val * scale_factor).round() as i32
            })
            .collect();

        Self {
            data,
            vocab_size,
            embed_dim,
            scale,
        }
    }

    /// Get embedding for a single token
    pub fn get(&self, token_id: usize) -> Option<FixedVector> {
        if token_id >= self.vocab_size {
            return None;
        }
        let start = token_id * self.embed_dim;
        let end = start + self.embed_dim;
        Some(FixedVector::from_raw(
            self.data[start..end].to_vec(),
            self.scale,
        ))
    }

    /// Get embeddings for multiple tokens
    pub fn get_batch(&self, token_ids: &[usize]) -> Vec<FixedVector> {
        token_ids.iter().filter_map(|&id| self.get(id)).collect()
    }

    /// Get row size in bytes (for OT)
    pub fn row_bytes(&self) -> usize {
        self.embed_dim * 4
    }

    /// Export as bytes for OT database
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Import from bytes
    pub fn from_bytes(bytes: &[u8], vocab_size: usize, embed_dim: usize, scale: u8) -> Result<Self> {
        if bytes.len() != vocab_size * embed_dim * 4 {
            return Err(ModelError::InvalidShape {
                expected: vec![vocab_size * embed_dim * 4],
                got: vec![bytes.len()],
            });
        }

        let data: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(Self {
            data,
            vocab_size,
            embed_dim,
            scale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_embedding_get() {
        let embeddings = EmbeddingTable::random(100, 64, DEFAULT_SCALE);

        let emb = embeddings.get(0).unwrap();
        assert_eq!(emb.len(), 64);

        let emb = embeddings.get(99).unwrap();
        assert_eq!(emb.len(), 64);

        assert!(embeddings.get(100).is_none());
    }

    #[test]
    fn test_embedding_batch() {
        let embeddings = EmbeddingTable::random(100, 64, DEFAULT_SCALE);
        let batch = embeddings.get_batch(&[0, 5, 10, 50]);
        assert_eq!(batch.len(), 4);
    }

    #[test]
    fn test_embedding_bytes_roundtrip() {
        let embeddings = EmbeddingTable::random(100, 64, DEFAULT_SCALE);
        let bytes = embeddings.to_bytes();
        let restored = EmbeddingTable::from_bytes(&bytes, 100, 64, DEFAULT_SCALE).unwrap();

        assert_eq!(embeddings.data, restored.data);
    }

    #[test]
    fn test_embedding_from_f32() {
        let data: Vec<f32> = (0..100 * 64).map(|i| i as f32 * 0.001).collect();
        let embeddings = EmbeddingTable::from_f32(&data, 100, 64, DEFAULT_SCALE).unwrap();

        assert_eq!(embeddings.vocab_size, 100);
        assert_eq!(embeddings.embed_dim, 64);
    }
}
