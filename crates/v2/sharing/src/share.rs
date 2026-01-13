//! Secret sharing types for v2

use crate::error::{Result, SharingError};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use shardlm_v2_core::tensor::{DType, Device, GpuTensor};
use zeroize::Zeroize;

/// A secret share (one half of XOR sharing)
#[derive(Debug)]
pub struct Share {
    /// The share data
    pub tensor: GpuTensor,
    /// Share index (0 = client, 1 = server)
    pub index: u8,
}

impl Share {
    /// Create a new share from tensor
    pub fn new(tensor: GpuTensor, index: u8) -> Self {
        Self { tensor, index }
    }

    /// Get the shape of the share
    pub fn shape(&self) -> &[usize] {
        &self.tensor.shape
    }

    /// Check if share is on GPU
    pub fn is_cuda(&self) -> bool {
        self.tensor.is_cuda()
    }
}

/// A pair of shares that can be recombined
#[derive(Debug)]
pub struct SharePair {
    /// Client share
    pub client: Share,
    /// Server share
    pub server: Share,
}

impl SharePair {
    /// Create shares from a secret value (CPU implementation)
    pub fn from_secret_cpu(secret: &[u8], shape: Vec<usize>, seed: u64) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random mask for client share
        let mut client_data = vec![0u8; secret.len()];
        rng.fill(&mut client_data[..]);

        // Server share = secret XOR client_share
        let server_data: Vec<u8> = secret
            .iter()
            .zip(client_data.iter())
            .map(|(s, c)| s ^ c)
            .collect();

        let client_tensor = GpuTensor::new_cpu(shape.clone(), DType::I8, client_data);
        let server_tensor = GpuTensor::new_cpu(shape, DType::I8, server_data);

        Self {
            client: Share::new(client_tensor, 0),
            server: Share::new(server_tensor, 1),
        }
    }

    /// Reconstruct the secret from both shares (CPU implementation)
    pub fn reconstruct_cpu(&self) -> Result<Vec<u8>> {
        let client_bytes = self
            .client
            .tensor
            .as_bytes()
            .ok_or_else(|| SharingError::InvalidShare("Client share not on CPU".to_string()))?;

        let server_bytes = self
            .server
            .tensor
            .as_bytes()
            .ok_or_else(|| SharingError::InvalidShare("Server share not on CPU".to_string()))?;

        if client_bytes.len() != server_bytes.len() {
            return Err(SharingError::ShapeMismatch {
                expected: self.client.shape().to_vec(),
                got: self.server.shape().to_vec(),
            });
        }

        let secret: Vec<u8> = client_bytes
            .iter()
            .zip(server_bytes.iter())
            .map(|(c, s)| c ^ s)
            .collect();

        Ok(secret)
    }
}

/// Sharing context for batched operations
pub struct SharingContext {
    /// RNG for generating shares
    rng: ChaCha20Rng,
    /// Device to use
    device: Device,
}

impl SharingContext {
    /// Create a new sharing context
    pub fn new(seed: u64, device: Device) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            device,
        }
    }

    /// Generate random bytes
    pub fn random_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; len];
        self.rng.fill(&mut bytes[..]);
        bytes
    }

    /// Share a batch of values
    pub fn share_batch(&mut self, secrets: &[Vec<u8>]) -> Vec<SharePair> {
        secrets
            .iter()
            .map(|secret| {
                let seed = self.rng.gen();
                SharePair::from_secret_cpu(secret, vec![secret.len()], seed)
            })
            .collect()
    }
}

impl Drop for SharingContext {
    fn drop(&mut self) {
        // Zeroize internal state
        let mut seed_bytes = [0u8; 32];
        self.rng.fill(&mut seed_bytes);
        seed_bytes.zeroize();
    }
}

/// Half-precision (BF16) sharing for GPU operations
pub mod bf16 {
    use super::*;
    use half::bf16;

    /// Share BF16 values
    pub fn share_bf16(values: &[bf16], seed: u64) -> (Vec<bf16>, Vec<bf16>) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let client: Vec<bf16> = values.iter().map(|_| bf16::from_f32(rng.gen())).collect();

        let server: Vec<bf16> = values
            .iter()
            .zip(client.iter())
            .map(|(v, c)| bf16::from_f32(v.to_f32() - c.to_f32()))
            .collect();

        (client, server)
    }

    /// Reconstruct BF16 values
    pub fn reconstruct_bf16(client: &[bf16], server: &[bf16]) -> Vec<bf16> {
        client
            .iter()
            .zip(server.iter())
            .map(|(c, s)| bf16::from_f32(c.to_f32() + s.to_f32()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_share_and_reconstruct() {
        let secret = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let pair = SharePair::from_secret_cpu(&secret, vec![8], 42);

        let reconstructed = pair.reconstruct_cpu().unwrap();
        assert_eq!(secret, reconstructed);
    }

    #[test]
    fn test_bf16_sharing() {
        use super::bf16 as bf16_sharing;
        use half::bf16;

        let values: Vec<bf16> = vec![1.0, 2.5, -3.0, 0.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect();

        let (client, server) = bf16_sharing::share_bf16(&values, 123);
        let reconstructed = bf16_sharing::reconstruct_bf16(&client, &server);

        for (orig, rec) in values.iter().zip(reconstructed.iter()) {
            let diff = (orig.to_f32() - rec.to_f32()).abs();
            assert!(diff < 0.01, "BF16 reconstruction error too large");
        }
    }
}
