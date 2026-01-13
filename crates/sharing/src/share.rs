//! Basic secret share types
//!
//! Secret shares are zeroized on drop to protect against memory disclosure.

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shardlm_fixed_point::{Fixed, FixedVector};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::Result;

/// A single share (either client or server share)
///
/// Implements ZeroizeOnDrop to securely erase share data from memory
/// when the share is no longer needed.
#[derive(Debug, Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct Share {
    /// The share values (zeroized on drop)
    pub data: Vec<i32>,
    /// Scale factor
    pub scale: u8,
}

impl Share {
    /// Create a new share from raw data
    pub fn from_raw(data: Vec<i32>, scale: u8) -> Self {
        Self { data, scale }
    }

    /// Create a zero share
    pub fn zeros(len: usize, scale: u8) -> Self {
        Self {
            data: vec![0; len],
            scale,
        }
    }

    /// Create a random share using thread RNG
    pub fn random(len: usize, scale: u8) -> Self {
        let mut rng = rand::thread_rng();
        Self::random_with_rng(len, scale, &mut rng)
    }

    /// Create a random share using a seeded RNG (for determinism)
    pub fn random_seeded(len: usize, scale: u8, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        Self::random_with_rng(len, scale, &mut rng)
    }

    /// Create a random share with a given RNG
    pub fn random_with_rng<R: RngCore>(len: usize, scale: u8, rng: &mut R) -> Self {
        let data: Vec<i32> = (0..len).map(|_| rng.next_u32() as i32).collect();
        Self { data, scale }
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Convert to FixedVector
    pub fn to_fixed_vector(&self) -> FixedVector {
        FixedVector::from_raw(self.data.clone(), self.scale)
    }

    /// Create from FixedVector
    pub fn from_fixed_vector(vec: &FixedVector) -> Self {
        Self {
            data: vec.data.clone(),
            scale: vec.scale,
        }
    }

    /// Encode to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.data.len() * 4);
        for &val in &self.data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        bytes
    }

    /// Decode from bytes
    pub fn from_bytes(bytes: &[u8], scale: u8) -> Self {
        let data: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self { data, scale }
    }
}

/// A pair of shares (client + server) that reconstruct to a value
#[derive(Debug, Clone)]
pub struct SharePair {
    /// Client's share
    pub client: Share,
    /// Server's share
    pub server: Share,
}

impl SharePair {
    /// Create shares from a plaintext value: X = X_c + X_s
    /// Client keeps X_c, Server gets X_s
    pub fn from_plaintext(plaintext: &FixedVector) -> Self {
        let server = Share::random(plaintext.len(), plaintext.scale);
        let client = Self::compute_client_share(plaintext, &server);
        Self { client, server }
    }

    /// Create shares from plaintext with a seeded RNG (for determinism)
    pub fn from_plaintext_seeded(plaintext: &FixedVector, seed: u64) -> Self {
        let server = Share::random_seeded(plaintext.len(), plaintext.scale, seed);
        let client = Self::compute_client_share(plaintext, &server);
        Self { client, server }
    }

    /// Compute client share: X_c = X - X_s
    fn compute_client_share(plaintext: &FixedVector, server: &Share) -> Share {
        let data: Vec<i32> = plaintext
            .data
            .iter()
            .zip(&server.data)
            .map(|(&x, &xs)| x.wrapping_sub(xs))
            .collect();
        Share {
            data,
            scale: plaintext.scale,
        }
    }

    /// Reconstruct the plaintext: X = X_c + X_s
    pub fn reconstruct(&self) -> Result<FixedVector> {
        if self.client.scale != self.server.scale {
            return Err(crate::error::SharingError::ScaleMismatch {
                expected: self.client.scale,
                got: self.server.scale,
            });
        }
        if self.client.len() != self.server.len() {
            return Err(crate::error::SharingError::DimensionMismatch {
                expected: self.client.len(),
                got: self.server.len(),
            });
        }

        let data: Vec<i32> = self
            .client
            .data
            .iter()
            .zip(&self.server.data)
            .map(|(&xc, &xs)| xc.wrapping_add(xs))
            .collect();

        Ok(FixedVector::from_raw(data, self.client.scale))
    }

    /// Get the scale
    pub fn scale(&self) -> u8 {
        self.client.scale
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.client.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.client.is_empty()
    }
}

/// Share a single fixed-point value
#[allow(dead_code)]
pub fn share_fixed(value: Fixed) -> (Fixed, Fixed) {
    let server_raw = rand::random::<i32>();
    let client_raw = value.raw.wrapping_sub(server_raw);
    (
        Fixed {
            raw: client_raw,
            scale: value.scale,
        },
        Fixed {
            raw: server_raw,
            scale: value.scale,
        },
    )
}

/// Share a single fixed-point value with seed
#[allow(dead_code)]
pub fn share_fixed_seeded(value: Fixed, seed: u64) -> (Fixed, Fixed) {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let server_raw = rng.next_u32() as i32;
    let client_raw = value.raw.wrapping_sub(server_raw);
    (
        Fixed {
            raw: client_raw,
            scale: value.scale,
        },
        Fixed {
            raw: server_raw,
            scale: value.scale,
        },
    )
}

/// Reconstruct from two shares
#[allow(dead_code)]
pub fn reconstruct_fixed(client: Fixed, server: Fixed) -> Fixed {
    Fixed {
        raw: client.raw.wrapping_add(server.raw),
        scale: client.scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

    #[test]
    fn test_share_reconstruct_vector() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, -1.0]).unwrap();
        let shares = SharePair::from_plaintext(&plaintext);
        let reconstructed = shares.reconstruct().unwrap();

        assert_eq!(plaintext, reconstructed);
    }

    #[test]
    fn test_share_reconstruct_seeded() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let shares1 = SharePair::from_plaintext_seeded(&plaintext, 42);
        let shares2 = SharePair::from_plaintext_seeded(&plaintext, 42);

        // Same seed should produce same server shares
        assert_eq!(shares1.server.data, shares2.server.data);
        assert_eq!(shares1.client.data, shares2.client.data);
    }

    #[test]
    fn test_share_fixed() {
        let value = Fixed::from_f64_default(3.14159).unwrap();
        let (client, server) = share_fixed(value);
        let reconstructed = reconstruct_fixed(client, server);

        assert_eq!(value.raw, reconstructed.raw);
    }

    #[test]
    fn test_server_share_reveals_nothing() {
        // The server share alone should look random and reveal nothing about the plaintext
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let shares = SharePair::from_plaintext(&plaintext);

        // Server share alone doesn't equal plaintext
        assert_ne!(shares.server.data, plaintext.data);

        // Server share should be uniformly distributed (statistical test)
        // Here we just check it's not all zeros
        assert!(shares.server.data.iter().any(|&x| x != 0));
    }

    #[test]
    fn test_share_bytes_roundtrip() {
        let share = Share::random(10, DEFAULT_SCALE);
        let bytes = share.to_bytes();
        let decoded = Share::from_bytes(&bytes, DEFAULT_SCALE);
        assert_eq!(share.data, decoded.data);
    }
}
