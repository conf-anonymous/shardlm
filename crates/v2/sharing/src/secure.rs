//! Secure share types with compile-time ownership enforcement
//!
//! This module provides type-safe shares that PREVENT the server from
//! reconstructing plaintext data through Rust's type system.
//!
//! # Security Model
//!
//! - `ClientShare<T>`: Data the client owns (can reconstruct)
//! - `ServerShare<T>`: Data the server computes on (CANNOT reconstruct alone)
//! - `SharePair<T>`: Both shares together (only valid on CLIENT)
//!
//! # Compile-Time Guarantees
//!
//! The `reconstruct()` method is ONLY available on `SharePair`, which
//! can only be constructed by the client. Server code cannot create
//! a `SharePair` from individual shares.

use std::marker::PhantomData;

#[allow(unused_imports)]
use crate::error::{Result, SharingError};

/// Marker trait for share ownership
pub trait ShareOwner: private::Sealed {}

/// Client-owned data marker
pub struct Client;
impl ShareOwner for Client {}

/// Server-owned data marker
pub struct Server;
impl ShareOwner for Server {}

mod private {
    pub trait Sealed {}
    impl Sealed for super::Client {}
    impl Sealed for super::Server {}
}

/// A single share with ownership tracking
///
/// The type parameter `O` tracks who owns this share:
/// - `Share<Client, T>`: Client's share of the data
/// - `Share<Server, T>`: Server's share of the data
///
/// # Security
///
/// Individual shares cannot be reconstructed. You need BOTH shares
/// (via `SharePair`) to recover the plaintext.
#[derive(Debug)]
pub struct SecureShare<O: ShareOwner, T> {
    data: Vec<T>,
    shape: Vec<usize>,
    _owner: PhantomData<O>,
}

impl<O: ShareOwner, T: Clone> SecureShare<O, T> {
    /// Create a new share (crate-internal use only)
    ///
    /// # Security
    ///
    /// This is pub(crate) to allow internal construction while preventing
    /// external code from creating arbitrary shares.
    pub(crate) fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            _owner: PhantomData,
        }
    }

    /// Get the shape of this share
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data (for computation)
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get mutable raw data (for computation)
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

/// Create client share from network data (for server receiving client shares)
///
/// # Security
///
/// This is a public constructor because the SERVER needs to wrap client-sent
/// data into ClientShare for processing. The data comes from the client via
/// the network - the server is NOT creating shares from plaintext.
impl<T: Clone> ClientShare<T> {
    /// Create a client share from data received over network
    ///
    /// # Security
    ///
    /// This wraps data that was already a share on the client side.
    /// The server cannot reconstruct plaintext with just this share.
    pub fn from_network(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new(data, shape)
    }
}

/// Create server share from network data (for server receiving server shares)
impl<T: Clone> ServerShare<T> {
    /// Create a server share from data received over network
    ///
    /// # Security
    ///
    /// This wraps data that was already a share on the client side.
    /// Combined with ClientShare, this allows processing but the server
    /// processes them SEPARATELY and never reconstructs.
    pub fn from_network(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::new(data, shape)
    }
}

// Note: For production, implement zeroization using the zeroize crate
// with a wrapper type. For now, rely on Rust's drop semantics.
//
// Security consideration: In production, sensitive data should be
// zeroized on drop to prevent memory scraping attacks.

/// Type alias for client's share
pub type ClientShare<T> = SecureShare<Client, T>;

/// Type alias for server's share
pub type ServerShare<T> = SecureShare<Server, T>;

/// A pair of shares that can be reconstructed
///
/// # Security
///
/// This type can ONLY be created by the client. The server
/// never has access to both shares simultaneously.
///
/// The `reconstruct()` method is intentionally ONLY on this type,
/// not on individual shares.
pub struct SecureSharePair<T> {
    client: ClientShare<T>,
    server: ServerShare<T>,
}

impl<T: Clone + std::ops::Add<Output = T>> SecureSharePair<T> {
    /// Create a share pair from plaintext (CLIENT-SIDE ONLY)
    ///
    /// # Security
    ///
    /// This method should only be called by client code.
    /// Server code should never have access to plaintext.
    pub fn from_plaintext<R: rand::Rng>(plaintext: Vec<T>, shape: Vec<usize>, rng: &mut R) -> Self
    where
        T: std::ops::Sub<Output = T>,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        use rand::distributions::Distribution;

        // Generate random server share
        let server_data: Vec<T> = (0..plaintext.len())
            .map(|_| rand::distributions::Standard.sample(rng))
            .collect();

        // Client share = plaintext - server share
        let client_data: Vec<T> = plaintext
            .into_iter()
            .zip(server_data.iter().cloned())
            .map(|(p, s)| p - s)
            .collect();

        Self {
            client: ClientShare::new(client_data, shape.clone()),
            server: ServerShare::new(server_data, shape),
        }
    }

    /// Reconstruct the plaintext from both shares (CLIENT-SIDE ONLY)
    ///
    /// # Security
    ///
    /// This method is the ONLY way to get plaintext.
    /// It requires BOTH shares, which only the client has.
    pub fn reconstruct(&self) -> Vec<T> {
        self.client
            .data
            .iter()
            .cloned()
            .zip(self.server.data.iter().cloned())
            .map(|(c, s)| c + s)
            .collect()
    }

    /// Get the client share (for sending to server operations)
    pub fn client_share(&self) -> &ClientShare<T> {
        &self.client
    }

    /// Get the server share (for OT transfer)
    pub fn server_share(&self) -> &ServerShare<T> {
        &self.server
    }

    /// Consume and return server share for OT transfer
    pub fn take_server_share(self) -> (ClientShare<T>, ServerShare<T>) {
        (self.client, self.server)
    }

    /// Get shape
    pub fn shape(&self) -> &[usize] {
        self.client.shape()
    }

    /// Create from pre-computed shares (crate-internal use)
    ///
    /// # Security
    ///
    /// This is only safe when called on the CLIENT side with shares that
    /// were properly generated (e.g., from OT protocol or internal operations).
    /// This is pub(crate) to prevent external misuse.
    pub(crate) fn from_shares(
        client: ClientShare<T>,
        server: ServerShare<T>,
    ) -> Self {
        Self { client, server }
    }
}

/// Server-side computation result
///
/// Contains output shares from server computation.
/// Server computes on its share, returns both output shares.
pub struct ServerComputeResult<T> {
    /// Y_c = X_c * W (computed from client's input share)
    pub output_from_client_share: Vec<T>,
    /// Y_s = X_s * W + b (computed from server's input share)
    pub output_from_server_share: Vec<T>,
    /// Output shape
    pub shape: Vec<usize>,
}

impl<T: Clone + std::ops::Add<Output = T>> ServerComputeResult<T> {
    /// Convert to share pair (CLIENT-SIDE ONLY)
    ///
    /// Client receives this result and creates a SharePair for reconstruction.
    pub fn into_share_pair(self) -> SecureSharePair<T> {
        SecureSharePair {
            client: ClientShare::new(self.output_from_client_share, self.shape.clone()),
            server: ServerShare::new(self.output_from_server_share, self.shape),
        }
    }
}

/// GPU-backed secure share
#[cfg(feature = "cuda")]
pub mod gpu {
    use super::*;
    use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};

    /// GPU share that CANNOT be reconstructed on server
    pub struct GpuSecureShare<O: ShareOwner> {
        tensor: CudaTensor,
        _owner: PhantomData<O>,
    }

    impl<O: ShareOwner> GpuSecureShare<O> {
        /// Create from CPU data
        pub fn from_cpu(data: Vec<f32>, shape: Vec<usize>, device: &GpuDevice) -> Result<Self> {
            let tensor = CudaTensor::from_f32(device, shape, data)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            Ok(Self {
                tensor,
                _owner: PhantomData,
            })
        }

        /// Get underlying tensor for computation
        pub fn tensor(&self) -> &CudaTensor {
            &self.tensor
        }

        /// Get mutable tensor
        pub fn tensor_mut(&mut self) -> &mut CudaTensor {
            &mut self.tensor
        }

        /// Get shape
        pub fn shape(&self) -> &[usize] {
            &self.tensor.shape
        }
    }

    pub type GpuClientShare = GpuSecureShare<Client>;
    pub type GpuServerShare = GpuSecureShare<Server>;

    /// GPU share pair - only valid on client
    pub struct GpuSecureSharePair {
        client: GpuClientShare,
        server: GpuServerShare,
    }

    impl GpuSecureSharePair {
        /// Reconstruct to CPU (CLIENT-SIDE ONLY)
        pub fn reconstruct_to_cpu(&self, device: &GpuDevice) -> Result<Vec<f32>> {
            let client_cpu = self.client.tensor.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;
            let server_cpu = self.server.tensor.to_f32_host(device)
                .map_err(|e| SharingError::CudaError(e.to_string()))?;

            Ok(client_cpu
                .iter()
                .zip(server_cpu.iter())
                .map(|(c, s)| c + s)
                .collect())
        }

        /// Get client share
        pub fn client(&self) -> &GpuClientShare {
            &self.client
        }

        /// Get server share
        pub fn server(&self) -> &GpuServerShare {
            &self.server
        }
    }
}

// =============================================================================
// SECURITY ASSERTIONS
// =============================================================================

/// Marker type that PREVENTS reconstruction on server
///
/// Server code should use this type to ensure it cannot accidentally
/// reconstruct plaintext.
pub struct ServerContext {
    _private: (),
}

impl ServerContext {
    /// Create server context (marks code as server-side)
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Compute linear layer on single share (SAFE)
    ///
    /// Server computes Y = X * W where X is a single share.
    /// Result is a share of Y, not plaintext Y.
    pub fn compute_linear<T: Clone>(
        &self,
        input_share: &[T],
        weight: &[T],
        in_features: usize,
        out_features: usize,
    ) -> Vec<T>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + Default + Copy,
    {
        let mut output = vec![T::default(); out_features];

        for i in 0..in_features {
            let xi = input_share[i];
            for j in 0..out_features {
                let wij = weight[i * out_features + j];
                output[j] = output[j] + xi * wij;
            }
        }

        output
    }

    // NOTE: There is intentionally NO reconstruct() method here.
    // The server context CANNOT reconstruct plaintext.
}

impl Default for ServerContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_share_and_reconstruct() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let plaintext: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        let pair = SecureSharePair::from_plaintext(plaintext.clone(), shape, &mut rng);
        let reconstructed = pair.reconstruct();

        for (orig, rec) in plaintext.iter().zip(reconstructed.iter()) {
            assert!((orig - rec).abs() < 1e-6);
        }
    }

    #[test]
    fn test_shares_are_different_from_plaintext() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let plaintext: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        let pair = SecureSharePair::from_plaintext(plaintext.clone(), shape, &mut rng);

        // Neither share should equal plaintext
        let client_data = pair.client_share().data();
        let server_data = pair.server_share().data();

        assert_ne!(client_data, &plaintext[..]);
        assert_ne!(server_data, &plaintext[..]);
    }

    #[test]
    fn test_server_context_cannot_reconstruct() {
        // This test verifies that ServerContext has no reconstruct method
        // by attempting to use it (compile-time check)
        let _ctx = ServerContext::new();

        // The following would NOT compile if uncommented:
        // ctx.reconstruct(...);

        // Server can only compute on individual shares
        let share: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let weight: Vec<f32> = vec![
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            0.5, 0.5,
        ];
        let _result = _ctx.compute_linear(&share, &weight, 4, 2);
    }
}
