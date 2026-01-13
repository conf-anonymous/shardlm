//! Software Confidential Computing Fallback
//!
//! Provides AES-GCM encryption for systems without hardware CC support.
//! This is a development/testing fallback, not as secure as hardware CC.

#[cfg(feature = "software-cc")]
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};

use crate::{
    attestation::AttestationToken,
    encrypted_tensor::{derive_session_key, generate_nonce, EncryptedBuffer},
    error::{CcError, Result},
    ConfidentialCompute,
};
use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Software-based CC provider using AES-GCM
#[cfg(feature = "software-cc")]
pub struct SoftwareCcProvider {
    device: Arc<CudaDevice>,
    session_key: [u8; 32],
    cipher: Aes256Gcm,
}

#[cfg(feature = "software-cc")]
impl SoftwareCcProvider {
    /// Create a new software CC provider
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        // Generate random session key
        let mut session_key = [0u8; 32];
        use rand::RngCore;
        rand::thread_rng().fill_bytes(&mut session_key);

        let cipher = Aes256Gcm::new_from_slice(&session_key)
            .map_err(|_| CcError::EncryptionError("Failed to create cipher".to_string()))?;

        Ok(Self {
            device,
            session_key,
            cipher,
        })
    }

    /// Encrypt data using AES-GCM
    fn aes_encrypt(&self, plaintext: &[u8]) -> Result<(Vec<u8>, [u8; 12], [u8; 16])> {
        let nonce = generate_nonce();
        let nonce_ref = Nonce::from_slice(&nonce);

        let ciphertext = self.cipher
            .encrypt(nonce_ref, plaintext)
            .map_err(|e| CcError::EncryptionError(format!("AES-GCM encrypt failed: {:?}", e)))?;

        // AES-GCM appends the tag to ciphertext
        let tag_start = ciphertext.len() - 16;
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&ciphertext[tag_start..]);

        let data = ciphertext[..tag_start].to_vec();

        Ok((data, nonce, tag))
    }

    /// Decrypt data using AES-GCM
    fn aes_decrypt(&self, ciphertext: &[u8], nonce: &[u8; 12], tag: &[u8; 16]) -> Result<Vec<u8>> {
        let nonce_ref = Nonce::from_slice(nonce);

        // Reconstruct ciphertext with tag
        let mut data_with_tag = ciphertext.to_vec();
        data_with_tag.extend_from_slice(tag);

        self.cipher
            .decrypt(nonce_ref, data_with_tag.as_ref())
            .map_err(|e| CcError::DecryptionError(format!("AES-GCM decrypt failed: {:?}", e)))
    }
}

#[cfg(feature = "software-cc")]
impl ConfidentialCompute for SoftwareCcProvider {
    fn is_available(&self) -> bool {
        true // Software CC is always available
    }

    fn provider_name(&self) -> &str {
        "Software CC (AES-GCM)"
    }

    fn get_attestation(&self) -> Result<AttestationToken> {
        // Software attestation: hash of session key and timestamp
        use sha2::{Sha256, Digest};

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut hasher = Sha256::new();
        hasher.update(&self.session_key);
        hasher.update(timestamp.to_le_bytes());
        hasher.update(b"software-cc-attestation");

        let measurement: [u8; 32] = hasher.finalize().into();

        // Sign with session key (simplified; real impl would use PKI)
        let mut sign_hasher = Sha256::new();
        sign_hasher.update(&measurement);
        sign_hasher.update(&self.session_key);
        let signature: Vec<u8> = sign_hasher.finalize().to_vec();

        Ok(AttestationToken {
            measurement,
            timestamp,
            signature,
            provider: self.provider_name().to_string(),
            gpu_id: self.device.name().unwrap_or_else(|_| "unknown".to_string()),
        })
    }

    fn verify_attestation(&self, token: &AttestationToken) -> Result<bool> {
        // Check provider
        if token.provider != self.provider_name() {
            return Ok(false);
        }

        // Check timestamp (1 hour validity)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if now > token.timestamp + 3600 {
            return Err(CcError::AttestationFailed("Token expired".to_string()));
        }

        // Verify signature
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&token.measurement);
        hasher.update(&self.session_key);
        let expected_sig: Vec<u8> = hasher.finalize().to_vec();

        Ok(token.signature == expected_sig)
    }

    fn allocate_secure(&self, size: usize) -> Result<EncryptedBuffer> {
        // Allocate encrypted buffer
        let data = vec![0u8; size * 4]; // f32 = 4 bytes
        let (encrypted_data, nonce, tag) = self.aes_encrypt(&data)?;

        Ok(EncryptedBuffer::encrypted(encrypted_data, nonce, tag, size))
    }

    fn encrypt_buffer(&self, data: &[f32]) -> Result<EncryptedBuffer> {
        // Convert f32 to bytes
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        // Encrypt
        let (encrypted_data, nonce, tag) = self.aes_encrypt(&bytes)?;

        Ok(EncryptedBuffer::encrypted(encrypted_data, nonce, tag, data.len()))
    }

    fn decrypt_buffer(&self, buffer: &EncryptedBuffer) -> Result<Vec<f32>> {
        if !buffer.encrypted {
            // Passthrough for unencrypted data
            let floats: Vec<f32> = buffer.data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            return Ok(floats);
        }

        // Decrypt
        let decrypted = self.aes_decrypt(&buffer.data, &buffer.nonce, &buffer.tag)?;

        // Convert bytes to f32
        let floats: Vec<f32> = decrypted
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    fn get_session_key(&self) -> Result<[u8; 32]> {
        Ok(self.session_key)
    }

    fn set_session_key(&mut self, key: [u8; 32]) -> Result<()> {
        self.session_key = key;
        self.cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| CcError::KeyExchangeError("Failed to set session key".to_string()))?;
        Ok(())
    }
}

#[cfg(all(test, feature = "software-cc"))]
mod tests {
    use super::*;

    #[test]
    fn test_software_cc_encrypt_decrypt() {
        let device = CudaDevice::new(0).expect("CUDA device required");
        let provider = SoftwareCcProvider::new(Arc::new(device)).unwrap();

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let encrypted = provider.encrypt_buffer(&data).unwrap();

        assert!(encrypted.encrypted);
        assert_ne!(encrypted.data.len(), data.len() * 4); // Should be encrypted (different size due to padding)

        let decrypted = provider.decrypt_buffer(&encrypted).unwrap();
        assert_eq!(data, decrypted);
    }

    #[test]
    fn test_software_cc_attestation() {
        let device = CudaDevice::new(0).expect("CUDA device required");
        let provider = SoftwareCcProvider::new(Arc::new(device)).unwrap();

        let token = provider.get_attestation().unwrap();
        assert!(provider.verify_attestation(&token).unwrap());
    }
}
