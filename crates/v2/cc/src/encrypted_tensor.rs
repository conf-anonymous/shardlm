//! Encrypted tensor support for H100 Confidential Computing
//!
//! Provides encrypted storage for tensor data with GPU-accelerated
//! encryption/decryption.

use crate::error::{CcError, Result};
use serde::{Deserialize, Serialize};

/// Encrypted buffer for secure data transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedBuffer {
    /// Encrypted data bytes
    pub data: Vec<u8>,

    /// AES-GCM nonce (12 bytes)
    pub nonce: [u8; 12],

    /// AES-GCM authentication tag (16 bytes)
    pub tag: [u8; 16],

    /// Original number of f32 elements
    pub original_len: usize,

    /// Whether data is actually encrypted
    pub encrypted: bool,
}

impl EncryptedBuffer {
    /// Create a new unencrypted buffer (for passthrough mode)
    pub fn unencrypted(data: Vec<u8>, original_len: usize) -> Self {
        Self {
            data,
            nonce: [0u8; 12],
            tag: [0u8; 16],
            original_len,
            encrypted: false,
        }
    }

    /// Create a new encrypted buffer
    pub fn encrypted(
        data: Vec<u8>,
        nonce: [u8; 12],
        tag: [u8; 16],
        original_len: usize,
    ) -> Self {
        Self {
            data,
            nonce,
            tag,
            original_len,
            encrypted: true,
        }
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Verify authentication tag (for encrypted data)
    pub fn verify(&self) -> bool {
        if !self.encrypted {
            return true; // Unencrypted data doesn't need verification
        }

        // Tag verification happens during decryption in AES-GCM
        // This is a placeholder for pre-verification checks
        self.tag != [0u8; 16]
    }
}

/// Encrypted tensor wrapper for GPU tensors
///
/// Wraps encrypted data with shape information for tensor operations.
#[derive(Debug, Clone)]
pub struct EncryptedTensor {
    /// Encrypted buffer containing tensor data
    pub buffer: EncryptedBuffer,

    /// Tensor shape
    pub shape: Vec<usize>,
}

impl EncryptedTensor {
    /// Create from encrypted buffer with shape
    pub fn new(buffer: EncryptedBuffer, shape: Vec<usize>) -> Result<Self> {
        // Verify shape matches buffer size
        let expected_elements: usize = shape.iter().product();
        if expected_elements != buffer.original_len {
            return Err(CcError::InvalidState(format!(
                "Shape {:?} expects {} elements but buffer has {}",
                shape, expected_elements, buffer.original_len
            )));
        }

        Ok(Self { buffer, shape })
    }

    /// Create unencrypted tensor (for passthrough)
    pub fn unencrypted(data: &[f32], shape: Vec<usize>) -> Result<Self> {
        let expected_elements: usize = shape.iter().product();
        if expected_elements != data.len() {
            return Err(CcError::InvalidState(format!(
                "Shape {:?} expects {} elements but got {}",
                shape, expected_elements, data.len()
            )));
        }

        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        Ok(Self {
            buffer: EncryptedBuffer::unencrypted(bytes, data.len()),
            shape,
        })
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if tensor is encrypted
    pub fn is_encrypted(&self) -> bool {
        self.buffer.encrypted
    }

    /// Reshape tensor (must have same number of elements)
    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<()> {
        let expected: usize = new_shape.iter().product();
        if expected != self.numel() {
            return Err(CcError::InvalidState(format!(
                "Cannot reshape {} elements to {:?}",
                self.numel(), new_shape
            )));
        }

        self.shape = new_shape;
        Ok(())
    }
}

/// Key derivation for per-session encryption
pub fn derive_session_key(master_key: &[u8; 32], session_id: &[u8]) -> [u8; 32] {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(master_key);
    hasher.update(session_id);
    hasher.update(b"shardlm-v3-session-key");

    let result = hasher.finalize();
    let mut key = [0u8; 32];
    key.copy_from_slice(&result);
    key
}

/// Generate random nonce for AES-GCM
pub fn generate_nonce() -> [u8; 12] {
    use rand::RngCore;
    let mut nonce = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce);
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypted_buffer() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let buffer = EncryptedBuffer::unencrypted(data.clone(), 2);

        assert!(!buffer.encrypted);
        assert_eq!(buffer.original_len, 2);
        assert!(buffer.verify());
    }

    #[test]
    fn test_encrypted_tensor() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = EncryptedTensor::unencrypted(&data, vec![2, 3]).unwrap();

        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert!(!tensor.is_encrypted());
    }

    #[test]
    fn test_tensor_reshape() {
        let data = vec![1.0f32; 12];
        let mut tensor = EncryptedTensor::unencrypted(&data, vec![3, 4]).unwrap();

        tensor.reshape(vec![2, 6]).unwrap();
        assert_eq!(tensor.shape, vec![2, 6]);

        tensor.reshape(vec![12]).unwrap();
        assert_eq!(tensor.shape, vec![12]);

        // Invalid reshape should fail
        assert!(tensor.reshape(vec![5, 5]).is_err());
    }

    #[test]
    fn test_key_derivation() {
        let master = [0u8; 32];
        let session_id = b"session-123";

        let key1 = derive_session_key(&master, session_id);
        let key2 = derive_session_key(&master, session_id);

        // Same inputs should give same key
        assert_eq!(key1, key2);

        // Different session should give different key
        let key3 = derive_session_key(&master, b"session-456");
        assert_ne!(key1, key3);
    }
}
