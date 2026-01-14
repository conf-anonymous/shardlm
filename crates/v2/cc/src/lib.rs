//! H100 Confidential Computing Module
//!
//! This module provides hardware-based memory encryption and attestation for
//! secure inference on NVIDIA H100 GPUs with Confidential Computing support.
//!
//! # Architecture
//!
//! The module uses a trait-based abstraction to support:
//! - **NVIDIA CC**: Hardware-enforced memory encryption via NVIDIA SDK
//! - **Software CC**: AES-GCM fallback for development/testing
//!
//! # Security Model
//!
//! H100 Confidential Computing provides:
//! - **Memory Encryption**: GPU memory encrypted at rest and in transit
//! - **Attestation**: Cryptographic proof that code runs in secure environment
//! - **Isolation**: Protected memory regions inaccessible to host OS
//!
//! # Usage
//!
//! ```ignore
//! use shardlm_v2_cc::{ConfidentialCompute, get_cc_provider};
//! use cudarc::driver::CudaDevice;
//!
//! // Get CUDA device and create CC provider
//! let device = CudaDevice::new(0).unwrap();
//! let cc = get_cc_provider(device).unwrap();
//!
//! // Verify attestation
//! let token = cc.get_attestation().unwrap();
//! assert!(cc.verify_attestation(&token).unwrap());
//!
//! // Create encrypted tensor
//! let data = vec![1.0f32; 1024];
//! let encrypted = cc.encrypt_buffer(&data).unwrap();
//!
//! // Decrypt for computation
//! let decrypted = cc.decrypt_buffer(&encrypted).unwrap();
//! ```

pub mod attestation;
pub mod encrypted_tensor;
pub mod error;

#[cfg(feature = "nvidia-cc")]
pub mod nvidia_cc;

#[cfg(feature = "software-cc")]
pub mod software_cc;

pub use attestation::{AttestationToken, AttestationVerifier};
pub use encrypted_tensor::{EncryptedBuffer, EncryptedTensor};
pub use error::{CcError, Result};

use cudarc::driver::CudaDevice;
use std::sync::Arc;

/// Trait for Confidential Computing providers
///
/// Implementations provide hardware or software-based memory encryption
/// and attestation capabilities.
pub trait ConfidentialCompute: Send + Sync {
    /// Check if CC is available on this system
    fn is_available(&self) -> bool;

    /// Get the CC provider name
    fn provider_name(&self) -> &str;

    /// Generate attestation token proving secure execution environment
    fn get_attestation(&self) -> Result<AttestationToken>;

    /// Verify an attestation token
    fn verify_attestation(&self, token: &AttestationToken) -> Result<bool>;

    /// Allocate encrypted GPU memory
    fn allocate_secure(&self, size: usize) -> Result<EncryptedBuffer>;

    /// Encrypt data for GPU transfer
    fn encrypt_buffer(&self, data: &[f32]) -> Result<EncryptedBuffer>;

    /// Decrypt buffer (returns f32 data)
    fn decrypt_buffer(&self, buffer: &EncryptedBuffer) -> Result<Vec<f32>>;

    /// Get encryption key (for session key exchange)
    fn get_session_key(&self) -> Result<[u8; 32]>;

    /// Set encryption key (received from client)
    fn set_session_key(&mut self, key: [u8; 32]) -> Result<()>;
}

/// Default (no-op) CC provider for systems without CC support
///
/// This provider passes data through without encryption.
/// Used for development and baseline benchmarking.
pub struct NoOpCcProvider {
    device: Arc<CudaDevice>,
    session_key: Option<[u8; 32]>,
}

impl NoOpCcProvider {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            device,
            session_key: None,
        }
    }
}

impl ConfidentialCompute for NoOpCcProvider {
    fn is_available(&self) -> bool {
        false // No-op provider, CC not available
    }

    fn provider_name(&self) -> &str {
        "NoOp (Insecure)"
    }

    fn get_attestation(&self) -> Result<AttestationToken> {
        // Return a dummy attestation for testing
        Ok(AttestationToken {
            measurement: [0u8; 32],
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: vec![0u8; 64],
            provider: "NoOp".to_string(),
            gpu_id: "unknown".to_string(),
        })
    }

    fn verify_attestation(&self, _token: &AttestationToken) -> Result<bool> {
        // No-op always returns true for testing
        tracing::warn!("NoOp CC provider: attestation verification skipped (INSECURE)");
        Ok(true)
    }

    fn allocate_secure(&self, size: usize) -> Result<EncryptedBuffer> {
        // Allocate regular GPU memory (not encrypted)
        Ok(EncryptedBuffer {
            data: vec![0u8; size * 4], // f32 = 4 bytes
            nonce: [0u8; 12],
            tag: [0u8; 16],
            original_len: size,
            encrypted: false,
        })
    }

    fn encrypt_buffer(&self, data: &[f32]) -> Result<EncryptedBuffer> {
        // Pass through without encryption
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        Ok(EncryptedBuffer {
            data: bytes,
            nonce: [0u8; 12],
            tag: [0u8; 16],
            original_len: data.len(),
            encrypted: false,
        })
    }

    fn decrypt_buffer(&self, buffer: &EncryptedBuffer) -> Result<Vec<f32>> {
        // Pass through without decryption
        let floats: Vec<f32> = buffer.data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    fn get_session_key(&self) -> Result<[u8; 32]> {
        Ok(self.session_key.unwrap_or([0u8; 32]))
    }

    fn set_session_key(&mut self, key: [u8; 32]) -> Result<()> {
        self.session_key = Some(key);
        Ok(())
    }
}

/// Get the appropriate CC provider for the current system
///
/// Tries providers in order of preference:
/// 1. NVIDIA H100 CC (hardware or software fallback, if feature enabled)
/// 2. Software CC (AES-GCM, if feature enabled)
/// 3. NoOp (passthrough, for development)
///
/// Note: NVIDIA CC provider handles both hardware CC (when ON) and
/// software fallback (when CC is OFF but GPU is capable).
pub fn get_cc_provider(device: Arc<CudaDevice>) -> Result<Box<dyn ConfidentialCompute>> {
    #[cfg(feature = "nvidia-cc")]
    {
        match nvidia_cc::NvidiaCcProvider::new(device.clone()) {
            Ok(provider) => {
                if provider.is_available() {
                    let status = provider.get_cc_status();
                    match status {
                        nvidia_cc::CcStatus::On => {
                            tracing::info!("Using NVIDIA H100 Confidential Computing (Hardware CC ON)");
                        }
                        nvidia_cc::CcStatus::OffCapable => {
                            tracing::warn!(
                                "H100 CC capable but OFF. Using software fallback. \
                                 Enable hardware CC with: sudo nvidia-smi conf-compute -srs 1"
                            );
                        }
                        _ => {
                            tracing::info!("Using NVIDIA H100 CC with software fallback");
                        }
                    }
                    return Ok(Box::new(provider));
                }
            }
            Err(e) => {
                tracing::debug!("NVIDIA CC not available: {}", e);
            }
        }
    }

    #[cfg(feature = "software-cc")]
    {
        tracing::info!("Using software CC (AES-GCM)");
        return Ok(Box::new(software_cc::SoftwareCcProvider::new(device)?));
    }

    tracing::warn!("No CC provider available, using NoOp (INSECURE)");
    Ok(Box::new(NoOpCcProvider::new(device)))
}

/// Check if H100 CC is available on this system
pub fn is_h100_cc_available() -> bool {
    // Check for H100 GPU with CC support
    // This is a simplified check; real implementation would query NVIDIA API

    if let Ok(device) = CudaDevice::new(0) {
        let name = device.name().unwrap_or_default();

        // H100 variants with CC support
        if name.contains("H100") {
            tracing::info!("Detected H100 GPU: {}", name);

            // Check for CC capability (would use NVIDIA API in production)
            #[cfg(feature = "nvidia-cc")]
            {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop_provider() {
        // CudaDevice::new() returns Arc<CudaDevice> in cudarc 0.12+
        let device = CudaDevice::new(0).expect("CUDA device required");
        let provider = NoOpCcProvider::new(device);

        assert!(!provider.is_available());
        assert_eq!(provider.provider_name(), "NoOp (Insecure)");

        // Test encryption/decryption passthrough
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let encrypted = provider.encrypt_buffer(&data).unwrap();
        let decrypted = provider.decrypt_buffer(&encrypted).unwrap();

        assert_eq!(data, decrypted);
    }

    #[test]
    fn test_attestation() {
        // CudaDevice::new() returns Arc<CudaDevice> in cudarc 0.12+
        let device = CudaDevice::new(0).expect("CUDA device required");
        let provider = NoOpCcProvider::new(device);

        let token = provider.get_attestation().unwrap();
        assert!(provider.verify_attestation(&token).unwrap());
    }
}
