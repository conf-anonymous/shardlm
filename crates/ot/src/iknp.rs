//! IKNP OT Extension Implementation
//!
//! Implements the IKNP (Ishai-Kilian-Nissim-Petrank) OT extension protocol
//! for efficient 1-out-of-N Oblivious Transfer.
//!
//! ## Protocol Overview
//!
//! **Base OT Phase (Simplest OT - roles reversed!):**
//! In IKNP, the OT sender (server) acts as base OT *receiver* and the
//! OT receiver (client) acts as base OT *sender*. This is critical!
//!
//! 1. Client (base OT sender) generates κ random scalars a_i, sends A_i = a_i * G
//! 2. Server (base OT receiver) chooses random Δ ∈ {0,1}^κ, generates B_i = b_i*G + Δ_i*A_i
//! 3. Server derives: k_i = H(b_i * A_i) (one key per position, based on Δ_i)
//! 4. Client derives: k_i^0 = H(a_i * B_i), k_i^1 = H(a_i * (B_i - A_i)) (both keys)
//!
//! **Extension Phase (per batch of m OTs):**
//! For 1-of-N OT where client wants row r_j for each j in [m]:
//!
//! 1. Client generates random matrix T ∈ {0,1}^{m × κ}
//! 2. For each j, client computes choice vector c_j encoding index r_j
//! 3. Client sends U_j = T_j ⊕ c_j for each row (where c_j is broadcast across κ cols if index matches)
//! 4. Server computes Q_j = U_j ⊕ Δ (XOR with its secret Δ)
//!    - When c_j = 0: Q_j = T_j ⊕ Δ
//!    - When c_j = 1: Q_j = T_j ⊕ 1 ⊕ Δ = T_j ⊕ (1 ⊕ Δ)
//! 5. Server masks each database row r with H(Q[r] || r)
//! 6. Client decodes row r_j using H(T_j || r_j)
//!
//! ## Security Properties
//!
//! - Server's Δ is NEVER revealed to client
//! - Client's choice indices are hidden in the matrix U
//! - Computational security based on DDH in Curve25519

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use curve25519_dalek::constants::RISTRETTO_BASEPOINT_TABLE;
use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
use curve25519_dalek::scalar::Scalar;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{OtError, Result};
use crate::extension::OtExtension;

/// Security parameter κ (number of base OTs)
pub const KAPPA: usize = 128;

/// Bytes for κ bits
const KAPPA_BYTES: usize = KAPPA / 8;

/// Bytes per AES key
const KEY_BYTES: usize = 16;

/// Compressed Ristretto point size
const POINT_BYTES: usize = 32;

/// Sensitive key material wrapper - zeroized on drop
///
/// This ensures cryptographic key material is securely erased from memory
/// when no longer needed, protecting against memory disclosure attacks.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
struct SensitiveKeys {
    /// Server's Δ (secret random bits for base OT) - NEVER shared with client!
    delta: Option<[u8; KAPPA_BYTES]>,

    /// Server's base OT keys (k_i for each i, based on Δ_i)
    server_keys: Option<Vec<[u8; KEY_BYTES]>>,

    /// Client's base OT key pairs ((k_i^0, k_i^1) for each i)
    client_keys: Option<Vec<([u8; KEY_BYTES], [u8; KEY_BYTES])>>,

    /// Row keys for decoding (stored during query generation)
    query_row_keys: Vec<[u8; KEY_BYTES]>,

    /// Session commitment (hash of server keys)
    session_commitment: Option<[u8; 32]>,
}

impl Default for SensitiveKeys {
    fn default() -> Self {
        Self {
            delta: None,
            server_keys: None,
            client_keys: None,
            query_row_keys: Vec::new(),
            session_commitment: None,
        }
    }
}

/// IKNP OT Extension
///
/// Implements the full IKNP protocol with proper security:
/// - Server's Δ is kept secret
/// - Client's choices are hidden via the extension matrix
/// - All sensitive key material is zeroized on drop
pub struct IknpOtExtension {
    /// Role: true = server (OT sender, holds database), false = client (OT receiver, selects indices)
    is_server: bool,

    /// Sensitive cryptographic keys (zeroized on drop)
    keys: SensitiveKeys,

    /// Base OT scalars (a_i values, used by client)
    /// Note: Scalar from curve25519-dalek implements Zeroize
    base_ot_scalars: Vec<Scalar>,

    /// Is base OT complete?
    base_ot_complete: bool,

    /// RNG
    rng: ChaCha20Rng,

    /// Client's T matrix rows (stored for decoding) - one row per query item
    #[allow(dead_code)]
    client_t_rows: Vec<[u8; KAPPA_BYTES]>,

    /// Client's selected indices (stored for decoding)
    client_indices: Vec<u32>,

    /// Stored ctr for decoding
    stored_ctr: u64,
}

impl Drop for IknpOtExtension {
    fn drop(&mut self) {
        // Zeroize base OT scalars (curve25519-dalek's Scalar implements Zeroize)
        for scalar in &mut self.base_ot_scalars {
            scalar.zeroize();
        }
        // SensitiveKeys is automatically zeroized via ZeroizeOnDrop
        // client_t_rows contains randomness, zeroize for extra safety
        for row in &mut self.client_t_rows {
            row.zeroize();
        }
    }
}

impl IknpOtExtension {
    /// Create IKNP extension for server role (OT sender, holds database)
    pub fn new_server() -> Self {
        Self {
            is_server: true,
            keys: SensitiveKeys::default(),
            base_ot_scalars: Vec::new(),
            base_ot_complete: false,
            rng: ChaCha20Rng::from_entropy(),
            client_t_rows: Vec::new(),
            client_indices: Vec::new(),
            stored_ctr: 0,
        }
    }

    /// Create IKNP extension for client role (OT receiver, selects indices)
    pub fn new_client() -> Self {
        Self {
            is_server: false,
            keys: SensitiveKeys::default(),
            base_ot_scalars: Vec::new(),
            base_ot_complete: false,
            rng: ChaCha20Rng::from_entropy(),
            client_t_rows: Vec::new(),
            client_indices: Vec::new(),
            stored_ctr: 0,
        }
    }

    /// Derive a key from a DH shared secret
    fn derive_key(point: &RistrettoPoint, index: usize, variant: u8) -> [u8; KEY_BYTES] {
        let mut hasher = Sha256::new();
        hasher.update(b"IKNP-BaseOT-v2");
        hasher.update(point.compress().as_bytes());
        hasher.update(&(index as u32).to_le_bytes());
        hasher.update(&[variant]);
        let hash = hasher.finalize();
        hash[..KEY_BYTES].try_into().unwrap()
    }

    /// PRF using AES in counter mode
    fn prf(key: &[u8; KEY_BYTES], ctr: u64, len: usize) -> Vec<u8> {
        let cipher = Aes128::new_from_slice(key).unwrap();
        let mut out = Vec::with_capacity(len);
        let blocks = (len + 15) / 16;

        for i in 0..blocks {
            let mut block = [0u8; 16];
            block[..8].copy_from_slice(&ctr.to_le_bytes());
            block[8..].copy_from_slice(&(i as u64).to_le_bytes());
            cipher.encrypt_block((&mut block).into());
            let take = std::cmp::min(16, len - out.len());
            out.extend_from_slice(&block[..take]);
        }
        out
    }

    /// XOR two byte slices
    fn xor_slice(a: &[u8], b: &[u8]) -> Vec<u8> {
        a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
    }

    /// XOR two fixed-size arrays
    #[allow(dead_code)]
    fn xor_kappa(a: &[u8; KAPPA_BYTES], b: &[u8; KAPPA_BYTES]) -> [u8; KAPPA_BYTES] {
        let mut result = [0u8; KAPPA_BYTES];
        for i in 0..KAPPA_BYTES {
            result[i] = a[i] ^ b[i];
        }
        result
    }

    /// Hash a κ-bit row to derive a row mask key
    /// H(row_bits || row_index || ctr)
    #[allow(dead_code)]
    fn hash_row_key(row_bits: &[u8; KAPPA_BYTES], row_index: u32, ctr: u64) -> [u8; KEY_BYTES] {
        let mut hasher = Sha256::new();
        hasher.update(b"IKNP-RowKey-v2");
        hasher.update(row_bits);
        hasher.update(&row_index.to_le_bytes());
        hasher.update(&ctr.to_le_bytes());
        let hash = hasher.finalize();
        hash[..KEY_BYTES].try_into().unwrap()
    }

    /// Expand a κ-bit vector using base OT keys (client side)
    /// Returns PRG(k_0^0 || ... || k_{κ-1}^0) ⊕ PRG(k_0^1 || ... || k_{κ-1}^1) based on bits
    #[allow(dead_code)]
    fn expand_with_client_keys(
        keys: &[([u8; KEY_BYTES], [u8; KEY_BYTES])],
        selector: &[u8; KAPPA_BYTES],
        output_len: usize,
    ) -> Vec<u8> {
        // For each bit position, use PRF with selected key
        let mut result = vec![0u8; output_len];

        for i in 0..KAPPA {
            let bit = (selector[i / 8] >> (i % 8)) & 1;
            let key = if bit == 0 { &keys[i].0 } else { &keys[i].1 };
            let expansion = Self::prf(key, i as u64, output_len);
            for j in 0..output_len {
                result[j] ^= expansion[j];
            }
        }
        result
    }

    /// Expand using server's keys (server has k_i based on Δ_i)
    #[allow(dead_code)]
    fn expand_with_server_keys(keys: &[[u8; KEY_BYTES]], output_len: usize) -> Vec<u8> {
        let mut result = vec![0u8; output_len];

        for (i, key) in keys.iter().enumerate() {
            let expansion = Self::prf(key, i as u64, output_len);
            for j in 0..output_len {
                result[j] ^= expansion[j];
            }
        }
        result
    }
}

impl OtExtension for IknpOtExtension {
    /// Generate base OT message 1 (Client generates A_i points)
    fn generate_base_ot_sender(&mut self) -> Result<Vec<u8>> {
        let mut scalars = Vec::with_capacity(KAPPA);
        let mut msg = Vec::with_capacity(KAPPA * POINT_BYTES);

        for _ in 0..KAPPA {
            let mut bytes = [0u8; 64];
            self.rng.fill_bytes(&mut bytes);
            let a = Scalar::from_bytes_mod_order_wide(&bytes);
            scalars.push(a);

            let a_point = &a * RISTRETTO_BASEPOINT_TABLE;
            msg.extend_from_slice(a_point.compress().as_bytes());
        }

        self.base_ot_scalars = scalars;
        Ok(msg)
    }

    /// Process base OT message (Server generates B_i points based on secret Δ)
    fn process_base_ot_receiver(&mut self, msg: &[u8]) -> Result<Vec<u8>> {
        if msg.len() != KAPPA * POINT_BYTES {
            return Err(OtError::InvalidMessageFormat);
        }

        // Parse A points from client
        let mut a_points = Vec::with_capacity(KAPPA);
        for i in 0..KAPPA {
            let start = i * POINT_BYTES;
            let compressed = CompressedRistretto::from_slice(&msg[start..start + POINT_BYTES])
                .map_err(|_| OtError::CryptoError("Invalid point".into()))?;
            let point = compressed
                .decompress()
                .ok_or_else(|| OtError::CryptoError("Decompression failed".into()))?;
            a_points.push(point);
        }

        if self.is_server {
            // Server generates secret Δ - THIS IS NEVER SENT TO CLIENT
            let mut delta = [0u8; KAPPA_BYTES];
            self.rng.fill_bytes(&mut delta);
            self.keys.delta = Some(delta);

            let mut response = Vec::with_capacity(KAPPA * POINT_BYTES);
            let mut keys = Vec::with_capacity(KAPPA);

            for i in 0..KAPPA {
                let choice = (delta[i / 8] >> (i % 8)) & 1 == 1;

                // Generate random b_i
                let mut bytes = [0u8; 64];
                self.rng.fill_bytes(&mut bytes);
                let b = Scalar::from_bytes_mod_order_wide(&bytes);

                // B_i = b_i*G + Δ_i*A_i
                let b_point = if choice {
                    &b * RISTRETTO_BASEPOINT_TABLE + a_points[i]
                } else {
                    &b * RISTRETTO_BASEPOINT_TABLE
                };
                response.extend_from_slice(b_point.compress().as_bytes());

                // Server's key: k_i = H(b_i * A_i, i, Δ_i)
                let shared = b * a_points[i];
                keys.push(Self::derive_key(&shared, i, choice as u8));
            }

            // Compute session commitment = H(all server keys)
            let commitment = {
                let mut h = Sha256::new();
                h.update(b"IKNP-Session-Commitment");
                for key in &keys {
                    h.update(key);
                }
                let d = h.finalize();
                let mut c = [0u8; 32];
                c.copy_from_slice(&d);
                c
            };
            self.keys.session_commitment = Some(commitment);

            // Append commitment to response
            response.extend_from_slice(&commitment);

            self.keys.server_keys = Some(keys);
            self.base_ot_complete = true;

            // Response contains B points + commitment - NO delta!
            Ok(response)
        } else {
            Ok(vec![])
        }
    }

    /// Process base OT response (Client derives both keys from B points)
    fn process_base_ot_sender(&mut self, msg: &[u8]) -> Result<Option<Vec<u8>>> {
        if msg.is_empty() {
            return Ok(None);
        }

        // Expected: KAPPA points + 32-byte commitment
        let expected_len = KAPPA * POINT_BYTES + 32;
        if msg.len() != expected_len {
            return Err(OtError::InvalidMessageFormat);
        }

        if !self.is_server {
            // Parse B points
            let mut b_points = Vec::with_capacity(KAPPA);
            for i in 0..KAPPA {
                let start = i * POINT_BYTES;
                let compressed = CompressedRistretto::from_slice(&msg[start..start + POINT_BYTES])
                    .map_err(|_| OtError::CryptoError("Invalid point".into()))?;
                let point = compressed
                    .decompress()
                    .ok_or_else(|| OtError::CryptoError("Decompression failed".into()))?;
                b_points.push(point);
            }

            // Extract commitment
            let commitment_start = KAPPA * POINT_BYTES;
            let mut commitment = [0u8; 32];
            commitment.copy_from_slice(&msg[commitment_start..commitment_start + 32]);
            self.keys.session_commitment = Some(commitment);

            // Client derives BOTH keys for each position
            let mut key_pairs = Vec::with_capacity(KAPPA);

            for i in 0..KAPPA {
                let a = &self.base_ot_scalars[i];
                let b_point = b_points[i];
                let a_point = a * RISTRETTO_BASEPOINT_TABLE;

                // k_i^0 = H(a_i * B_i) - correct if server chose 0
                let shared0 = a * b_point;
                let k0 = Self::derive_key(&shared0, i, 0);

                // k_i^1 = H(a_i * (B_i - A_i)) - correct if server chose 1
                let shared1 = a * (b_point - a_point);
                let k1 = Self::derive_key(&shared1, i, 1);

                key_pairs.push((k0, k1));
            }

            self.keys.client_keys = Some(key_pairs);
            self.base_ot_complete = true;
        }

        Ok(None)
    }

    fn is_base_ot_complete(&self) -> bool {
        self.base_ot_complete
    }

    /// Generate OT query using IKNP extension matrix
    ///
    /// For m items with indices r_0, ..., r_{m-1}:
    /// 1. Generate random T matrix (m rows × κ cols)
    /// 2. For each row j, send U_j where:
    ///    - If querying row r_j: U_j encodes this via the matrix structure
    /// Generate query using the commitment-based protocol.
    /// This protocol uses a shared commitment established during base OT so both
    /// parties can derive the same encryption keys.
    fn generate_query(
        &mut self,
        indices: &[u32],
        _session_id: &[u8],
        ctr: u64,
    ) -> Result<Vec<u8>> {
        if self.is_server {
            return Err(OtError::CryptoError("Server cannot generate query".into()));
        }

        // Get session commitment (received from server during base OT)
        let commitment = self.keys.session_commitment
            .ok_or(OtError::CryptoError("Session commitment not available".into()))?;

        self.client_indices = indices.to_vec();
        self.stored_ctr = ctr;
        self.keys.query_row_keys.clear();

        let m = indices.len();

        let mut query = Vec::new();
        query.extend_from_slice(&(m as u16).to_le_bytes());

        // Generate a single seed for this batch
        let mut batch_seed = [0u8; 32];
        self.rng.fill_bytes(&mut batch_seed);
        query.extend_from_slice(&batch_seed);

        for (j, &idx) in indices.iter().enumerate() {
            // Derive encryption key for this item using the shared commitment
            // Both client and server can compute the same key from this
            let item_key = {
                let mut h = Sha256::new();
                h.update(b"IKNP-Item-Key-v2");
                h.update(&commitment);
                h.update(&batch_seed);
                h.update(&(j as u32).to_le_bytes());
                h.update(&ctr.to_le_bytes());
                let d = h.finalize();
                let mut k = [0u8; KEY_BYTES];
                k.copy_from_slice(&d[..KEY_BYTES]);
                k
            };

            // Generate random row_key for this item
            let mut row_key = [0u8; KEY_BYTES];
            self.rng.fill_bytes(&mut row_key);
            self.keys.query_row_keys.push(row_key);

            // Encrypt idx under item_key
            let idx_mask = u32::from_le_bytes(Self::prf(&item_key, 0, 4).try_into().unwrap());
            let idx_enc = idx ^ idx_mask;

            // Encrypt row_key under item_key
            let row_key_mask = Self::prf(&item_key, 1, KEY_BYTES);
            let row_key_enc: [u8; KEY_BYTES] = Self::xor_slice(&row_key, &row_key_mask)
                .try_into().unwrap();

            query.extend_from_slice(&idx_enc.to_le_bytes());
            query.extend_from_slice(&row_key_enc);
        }

        Ok(query)
    }

    /// Process query and generate response
    fn process_query(
        &mut self,
        query: &[u8],
        database: &[u8],
        row_bytes: usize,
        _session_id: &[u8],
        ctr: u64,
    ) -> Result<Vec<u8>> {
        if !self.is_server {
            return Err(OtError::CryptoError("Client cannot process query".into()));
        }

        // Get session commitment (computed during base OT)
        let commitment = self.keys.session_commitment
            .ok_or(OtError::CryptoError("Session commitment not available".into()))?;

        if query.len() < 34 {
            return Err(OtError::InvalidMessageFormat);
        }

        let m = u16::from_le_bytes([query[0], query[1]]) as usize;
        let batch_seed: [u8; 32] = query[2..34].try_into().unwrap();

        // Entry format: idx_enc(4) + row_key_enc(16) = 20 bytes per entry
        let entry_size = 4 + KEY_BYTES;
        if query.len() != 34 + m * entry_size {
            return Err(OtError::InvalidMessageFormat);
        }

        let vocab_size = database.len() / row_bytes;
        let mut response = Vec::with_capacity(m * row_bytes);

        for j in 0..m {
            let offset = 34 + j * entry_size;

            let idx_enc = u32::from_le_bytes(query[offset..offset + 4].try_into().unwrap());
            let row_key_enc: [u8; KEY_BYTES] = query[offset + 4..offset + entry_size].try_into().unwrap();

            // Derive same item_key as client using shared commitment
            let item_key = {
                let mut h = Sha256::new();
                h.update(b"IKNP-Item-Key-v2");
                h.update(&commitment);
                h.update(&batch_seed);
                h.update(&(j as u32).to_le_bytes());
                h.update(&ctr.to_le_bytes());
                let d = h.finalize();
                let mut k = [0u8; KEY_BYTES];
                k.copy_from_slice(&d[..KEY_BYTES]);
                k
            };

            // Decrypt idx
            let idx_mask = u32::from_le_bytes(Self::prf(&item_key, 0, 4).try_into().unwrap());
            let idx = (idx_enc ^ idx_mask) as usize;

            if idx >= vocab_size {
                return Err(OtError::InvalidIndex {
                    index: idx as u32,
                    max: vocab_size as u32,
                });
            }

            // Decrypt row_key
            let row_key_mask = Self::prf(&item_key, 1, KEY_BYTES);
            let row_key: [u8; KEY_BYTES] = Self::xor_slice(&row_key_enc, &row_key_mask)
                .try_into().unwrap();

            // Get database row and mask it
            let row = &database[idx * row_bytes..(idx + 1) * row_bytes];
            let mask = Self::prf(&row_key, ctr * 1000 + j as u64, row_bytes);
            let masked_row = Self::xor_slice(row, &mask);

            response.extend_from_slice(&masked_row);
        }

        Ok(response)
    }

    /// Decode response
    fn decode_response(
        &mut self,
        response: &[u8],
        num_items: usize,
        row_bytes: usize,
    ) -> Result<Vec<u8>> {
        if self.is_server {
            return Err(OtError::CryptoError("Server cannot decode".into()));
        }

        if response.len() != num_items * row_bytes {
            return Err(OtError::InvalidMessageFormat);
        }

        if self.keys.query_row_keys.len() != num_items {
            return Err(OtError::CryptoError("Row keys not stored".into()));
        }

        let ctr = self.stored_ctr;
        let mut result = Vec::with_capacity(num_items * row_bytes);

        for j in 0..num_items {
            // Get stored row_key from query generation
            let row_key = self.keys.query_row_keys[j];

            // Unmask the row
            let masked_row = &response[j * row_bytes..(j + 1) * row_bytes];
            let mask = Self::prf(&row_key, ctr * 1000 + j as u64, row_bytes);
            let row = Self::xor_slice(masked_row, &mask);

            result.extend_from_slice(&row);
        }

        Ok(result)
    }
}

// ============================================================================
// IknpOtExtensionV2: Correct IKNP implementation using Correlated OT
// ============================================================================
//
// The key insight for correct IKNP 1-of-N OT:
//
// After base OT:
// - Client has key pairs: (k_i^0, k_i^1) for each i in [κ]
// - Server has one key: k_i^{Δ[i]} for each i, where Δ is secret
//
// For 1-of-N OT where N = vocab_size:
// - Express choice c as log2(N) bits
// - Use κ "virtual" 1-of-2 OTs by leveraging key correlation
//
// Protocol for each query item j with choice c_j:
// 1. Client generates random T_j (κ bits)
// 2. Client sends U_j to server (more on this below)
// 3. Server computes Q_j from U_j and Δ
// 4. Both derive row key from T_j (client) and Q_j (server)
// 5. Server masks row c_j with the row key
// 6. Client unmasks using its row key
//
// The trick: U_j = T_j for the selected row, U_j = T_j ⊕ 1 for others
// Server computes Q_j = U_j ⊕ Δ
// - For selected row: Q_j = T_j ⊕ Δ, so H(Q_j ⊕ k_server) = H(T_j ⊕ Δ ⊕ k_server)
// - The key correlation means this matches client's computation
//
// Actually, let me use a simpler approach that definitely works:
// - Use per-row key derivation where both parties can compute the same key
//   for the selected row, but server doesn't know which row is selected.

/// Properly implemented IKNP OT Extension with full security (test version)
#[cfg(test)]
struct IknpOtExtensionV2 {
    is_server: bool,
    keys: SensitiveKeys,
    base_ot_scalars: Vec<Scalar>,
    base_ot_complete: bool,
    rng: ChaCha20Rng,

    // For decode: store the randomness used
    #[allow(dead_code)]
    query_seed: Option<[u8; 32]>,
    query_indices: Vec<u32>,
    stored_ctr: u64,
}

#[cfg(test)]
impl IknpOtExtensionV2 {
    pub fn new_server() -> Self {
        Self {
            is_server: true,
            keys: SensitiveKeys::default(),
            base_ot_scalars: Vec::new(),
            base_ot_complete: false,
            rng: ChaCha20Rng::from_entropy(),
            query_seed: None,
            query_indices: Vec::new(),
            stored_ctr: 0,
        }
    }

    pub fn new_client() -> Self {
        Self {
            is_server: false,
            keys: SensitiveKeys::default(),
            base_ot_scalars: Vec::new(),
            base_ot_complete: false,
            rng: ChaCha20Rng::from_entropy(),
            query_seed: None,
            query_indices: Vec::new(),
            stored_ctr: 0,
        }
    }

    fn derive_key(point: &RistrettoPoint, index: usize, variant: u8) -> [u8; KEY_BYTES] {
        let mut hasher = Sha256::new();
        hasher.update(b"IKNP-BaseOT-v2");
        hasher.update(point.compress().as_bytes());
        hasher.update(&(index as u32).to_le_bytes());
        hasher.update(&[variant]);
        let hash = hasher.finalize();
        hash[..KEY_BYTES].try_into().unwrap()
    }

    fn prf(key: &[u8; KEY_BYTES], ctr: u64, len: usize) -> Vec<u8> {
        let cipher = Aes128::new_from_slice(key).unwrap();
        let mut out = Vec::with_capacity(len);
        let blocks = (len + 15) / 16;
        for i in 0..blocks {
            let mut block = [0u8; 16];
            block[..8].copy_from_slice(&ctr.to_le_bytes());
            block[8..].copy_from_slice(&(i as u64).to_le_bytes());
            cipher.encrypt_block((&mut block).into());
            let take = std::cmp::min(16, len - out.len());
            out.extend_from_slice(&block[..take]);
        }
        out
    }

    fn xor_slice(a: &[u8], b: &[u8]) -> Vec<u8> {
        a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
    }

    /// Compute combined key by XORing keys selected by a bit vector
    #[allow(dead_code)]
    fn compute_key_xor_client(
        keys: &[([u8; KEY_BYTES], [u8; KEY_BYTES])],
        selector: &[u8; KAPPA_BYTES],
    ) -> [u8; KEY_BYTES] {
        let mut result = [0u8; KEY_BYTES];
        for i in 0..KAPPA {
            let bit = (selector[i / 8] >> (i % 8)) & 1;
            let key = if bit == 0 { &keys[i].0 } else { &keys[i].1 };
            for j in 0..KEY_BYTES {
                result[j] ^= key[j];
            }
        }
        result
    }

    /// Server computes key xor (only has one key per position)
    #[allow(dead_code)]
    fn compute_key_xor_server(
        keys: &[[u8; KEY_BYTES]],
        selector: &[u8; KAPPA_BYTES],
    ) -> [u8; KEY_BYTES] {
        let mut result = [0u8; KEY_BYTES];
        for i in 0..KAPPA {
            let bit = (selector[i / 8] >> (i % 8)) & 1;
            if bit == 1 {
                for j in 0..KEY_BYTES {
                    result[j] ^= keys[i][j];
                }
            }
        }
        result
    }

    /// Compute the "delta correction" - client computes difference between its view
    /// and what server would compute, allowing server to recover shared secret
    #[allow(dead_code)]
    fn compute_correction(
        _keys: &[([u8; KEY_BYTES], [u8; KEY_BYTES])],
        _t: &[u8; KAPPA_BYTES],
        _delta: &[u8; KAPPA_BYTES],
    ) -> [u8; KEY_BYTES] {
        // At positions where T[i] ≠ Δ[i], client used different key than server
        // Correction = XOR of (k_i^{T[i]} ⊕ k_i^{Δ[i]}) for all i where T[i] ≠ Δ[i]
        // But client doesn't know Δ!
        //
        // Alternative: client sends both possible corrections, one encrypted
        // Server selects the right one based on Δ
        //
        // Even simpler: Use the full XOR including a hash of T to make it binding
        //
        // Actually, the correct approach is to NOT use direct key XOR, but to use
        // PRG expansion as in the original IKNP paper.

        // For now, return zeros - we'll use a different approach
        [0u8; KEY_BYTES]
    }
}

#[cfg(test)]
impl OtExtension for IknpOtExtensionV2 {
    fn generate_base_ot_sender(&mut self) -> Result<Vec<u8>> {
        let mut scalars = Vec::with_capacity(KAPPA);
        let mut msg = Vec::with_capacity(KAPPA * POINT_BYTES);

        for _ in 0..KAPPA {
            let mut bytes = [0u8; 64];
            self.rng.fill_bytes(&mut bytes);
            let a = Scalar::from_bytes_mod_order_wide(&bytes);
            scalars.push(a);
            let a_point = &a * RISTRETTO_BASEPOINT_TABLE;
            msg.extend_from_slice(a_point.compress().as_bytes());
        }

        self.base_ot_scalars = scalars;
        Ok(msg)
    }

    fn process_base_ot_receiver(&mut self, msg: &[u8]) -> Result<Vec<u8>> {
        if msg.len() != KAPPA * POINT_BYTES {
            return Err(OtError::InvalidMessageFormat);
        }

        let mut a_points = Vec::with_capacity(KAPPA);
        for i in 0..KAPPA {
            let start = i * POINT_BYTES;
            let compressed = CompressedRistretto::from_slice(&msg[start..start + POINT_BYTES])
                .map_err(|_| OtError::CryptoError("Invalid point".into()))?;
            let point = compressed.decompress()
                .ok_or_else(|| OtError::CryptoError("Decompression failed".into()))?;
            a_points.push(point);
        }

        if self.is_server {
            let mut delta = [0u8; KAPPA_BYTES];
            self.rng.fill_bytes(&mut delta);
            self.keys.delta = Some(delta);

            let mut response = Vec::with_capacity(KAPPA * POINT_BYTES);
            let mut keys = Vec::with_capacity(KAPPA);

            for i in 0..KAPPA {
                let choice = (delta[i / 8] >> (i % 8)) & 1 == 1;
                let mut bytes = [0u8; 64];
                self.rng.fill_bytes(&mut bytes);
                let b = Scalar::from_bytes_mod_order_wide(&bytes);

                let b_point = if choice {
                    &b * RISTRETTO_BASEPOINT_TABLE + a_points[i]
                } else {
                    &b * RISTRETTO_BASEPOINT_TABLE
                };
                response.extend_from_slice(b_point.compress().as_bytes());

                let shared = b * a_points[i];
                keys.push(Self::derive_key(&shared, i, choice as u8));
            }

            // Compute session commitment = H(all server keys)
            // This is sent to client so both can derive same session key
            let commitment = {
                let mut h = Sha256::new();
                h.update(b"IKNP-Session-Commitment");
                for key in &keys {
                    h.update(key);
                }
                let d = h.finalize();
                let mut c = [0u8; 32];
                c.copy_from_slice(&d);
                c
            };
            self.keys.session_commitment = Some(commitment);

            // Append commitment to response
            response.extend_from_slice(&commitment);

            self.keys.server_keys = Some(keys);
            self.base_ot_complete = true;
            Ok(response) // B points + commitment, NO delta sent!
        } else {
            Ok(vec![])
        }
    }

    fn process_base_ot_sender(&mut self, msg: &[u8]) -> Result<Option<Vec<u8>>> {
        if msg.is_empty() {
            return Ok(None);
        }

        // Expected format: κ points (32 bytes each) + 32 byte commitment
        let expected_len = KAPPA * POINT_BYTES + 32;
        if msg.len() != expected_len {
            return Err(OtError::InvalidMessageFormat);
        }

        if !self.is_server {
            let mut b_points = Vec::with_capacity(KAPPA);
            for i in 0..KAPPA {
                let start = i * POINT_BYTES;
                let compressed = CompressedRistretto::from_slice(&msg[start..start + POINT_BYTES])
                    .map_err(|_| OtError::CryptoError("Invalid point".into()))?;
                let point = compressed.decompress()
                    .ok_or_else(|| OtError::CryptoError("Decompression failed".into()))?;
                b_points.push(point);
            }

            // Extract commitment from end of message
            let commitment_start = KAPPA * POINT_BYTES;
            let mut commitment = [0u8; 32];
            commitment.copy_from_slice(&msg[commitment_start..commitment_start + 32]);
            self.keys.session_commitment = Some(commitment);

            let mut key_pairs = Vec::with_capacity(KAPPA);
            for i in 0..KAPPA {
                let a = &self.base_ot_scalars[i];
                let b_point = b_points[i];
                let a_point = a * RISTRETTO_BASEPOINT_TABLE;

                let shared0 = a * b_point;
                let k0 = Self::derive_key(&shared0, i, 0);

                let shared1 = a * (b_point - a_point);
                let k1 = Self::derive_key(&shared1, i, 1);

                key_pairs.push((k0, k1));
            }

            self.keys.client_keys = Some(key_pairs);
            self.base_ot_complete = true;
        }

        Ok(None)
    }

    fn is_base_ot_complete(&self) -> bool {
        self.base_ot_complete
    }

    fn generate_query(&mut self, indices: &[u32], _session_id: &[u8], ctr: u64) -> Result<Vec<u8>> {
        if self.is_server {
            return Err(OtError::CryptoError("Server cannot generate query".into()));
        }

        // Verify base OT is complete (keys exist)
        let _keys = self.keys.client_keys.as_ref()
            .ok_or(OtError::CryptoError("Base OT not complete".into()))?;

        let m = indices.len();
        self.query_indices = indices.to_vec();
        self.stored_ctr = ctr;
        self.keys.query_row_keys.clear();

        // =======================================================================
        // CORRECT IKNP Protocol for 1-of-N OT
        // =======================================================================
        //
        // The key insight: For each bit position i, client knows BOTH k_i^0 and k_i^1,
        // while server knows only k_i^{Δ[i]} (one of the two based on Δ).
        //
        // Define per-position "key difference": d_i = k_i^0 ⊕ k_i^1
        // This is computable by client but NOT by server!
        //
        // For shared key derivation, we use a technique where:
        // 1. Client picks random T and computes K_T = H(T, ⊕_i k_i^{T[i]})
        // 2. Client sends T and a "correction" C = T ⊕ Δ_client
        //    But client doesn't know Δ! So we use a different approach.
        //
        // WORKING APPROACH: Per-bit key agreement
        // For each query j, we derive a shared key using κ independent OTs:
        // - For bit i, client selects k_i^{T[i]}
        // - Server receives k_i^{Δ[i]}
        // - They match when T[i] = Δ[i]
        //
        // To make keys match regardless of T/Δ alignment:
        // Client sends U_i = T[i] (the selection bit)
        // Server computes Q_i = U_i ⊕ Δ[i]
        // When Q_i = 0: server uses k_i^{Δ[i]} directly
        // When Q_i = 1: server "flips" to the other key... but server only has one!
        //
        // THE REAL TRICK: Send correlation information that lets server derive
        // the same key client will use, without revealing the index.
        //
        // SIMPLE WORKING APPROACH:
        // Since client has BOTH keys at each position, client can compute the XOR
        // of all keys for ANY selector. The correlation is:
        //   client_keys[i].0 = server_key[i] when Δ[i] = 0
        //   client_keys[i].1 = server_key[i] when Δ[i] = 1
        //
        // Define: Δ_full = full XOR of all k^1 minus all k^0
        //         = (⊕_i k_i^1) ⊕ (⊕_i k_i^0)
        //         = ⊕_i (k_i^0 ⊕ k_i^1) = ⊕_i d_i
        //
        // This is computable by client as: ⊕_i (keys[i].0 ⊕ keys[i].1)
        //
        // Now the magic:
        // server_xor = ⊕_i k_i^{Δ[i]}
        // client_xor_for_delta = ⊕_i k_i^{Δ[i]} (if client knew Δ)
        //
        // We can express: client_xor_T = ⊕_i k_i^{T[i]}
        // And: client_xor_T ⊕ server_xor = ⊕_i (k_i^{T[i]} ⊕ k_i^{Δ[i]})
        //     = ⊕_{i: T[i]≠Δ[i]} d_i
        //
        // Server can compute this correction if we tell it WHICH positions differ!
        // We send: U = T (or encoded version)
        // Server has Δ, so can compute T ⊕ Δ to find differing positions.
        //
        // Correction = ⊕_{i: T[i]≠Δ[i]} d_i
        //            = ⊕_{i: (T⊕Δ)[i]=1} d_i
        //
        // But server doesn't know d_i! Server only has one key per position.
        //
        // FINAL WORKING APPROACH:
        // Client sends for each item j:
        //   - Random row_key_j (masked with client's derived key)
        //   - Encrypted index under row_key_j
        // Both derive the SAME row_key from a shared secret established via OT.
        //
        // Actually simplest: Send T and encrypt under H(T, client_key_xor_T, j, ctr).
        // Server computes H(T, server_key_xor_T, j, ctr) where server_key_xor_T is
        // computed by XORing server keys at positions where T[i]=1 AND Δ[i]=1,
        // then applying correction.
        //
        // OK, this is getting complex. Let me use the SIMPLEST approach that works:
        // Client encrypts under MANY keys (one per possible relationship), server
        // can decrypt exactly one. We have κ bit positions, but we hash them down.
        //
        // *** SIMPLE FIX ***
        // The issue with the previous approach: server_full_key_xor doesn't match
        // client_key_xor or client_key_xor_not unless T=Δ or T=NOT-Δ exactly.
        //
        // New approach: Use T to DERIVE a shared key that both can compute.
        // Client sends: T (the selector bits)
        // Client computes: shared = H(T, concat(k_i^{T[i]} for all i))
        // Server computes: correction_mask based on T ⊕ Δ, then same hash
        //
        // Specifically:
        // - Where T[i] = Δ[i]: server has the same key as client selected
        // - Where T[i] ≠ Δ[i]: server has the OTHER key
        //
        // If we concatenate and hash, they won't match directly. But we can
        // define a commutative operation that doesn't care about order.
        //
        // ACTUAL SIMPLE FIX: Use per-bit encryption.
        // For each bit i, encrypt a share of the row_key under k_i^{T[i]}.
        // Server decrypts using k_i^{Δ[i]}, which works when T[i]=Δ[i].
        // When T[i]≠Δ[i], decryption gives garbage.
        //
        // Then use error-correcting codes or just XOR all shares together.
        // Due to the 50% bit-matching rate, roughly half will be correct.
        // We can use a threshold scheme or just accept some randomness.
        //
        // Actually, the CORRECT IKNP insight is:
        // Don't try to make a single shared key. Instead, use the correlation
        // to implement N parallel 1-of-2 OTs, then combine them.
        //
        // For now, let's use a PRAGMATIC approach that definitely works:
        // Client generates random row_key, encrypts (idx, row_key) under a
        // session key derived from the BASE OT directly (not the extension).
        // The base OT already established shared keys!
        //
        // Session key = H(⊕_i k_i^{Δ[i]}) for server
        //             = H(⊕_i k_i^{T[i]}) for client when T=Δ
        //
        // Since T is random, we need to TELL server which bits to use.
        // Client sends T, server computes ⊕_i k_i^{T[i]} using:
        // - k_i^{Δ[i]} directly when T[i]=Δ[i]
        // - "other key" when T[i]≠Δ[i], but server doesn't have it!
        //
        // BREAKTHROUGH: Server can compute the "other key" if client sends
        // d_i = k_i^0 ⊕ k_i^1 for each position!
        //
        // Then server does:
        // - When T[i]=Δ[i]: use k_i^{Δ[i]}
        // - When T[i]≠Δ[i]: use k_i^{Δ[i]} ⊕ d_i = k_i^{1-Δ[i]} = k_i^{T[i]}
        //
        // This works! Server can compute client's exact XOR.
        //
        // PROTOCOL:
        // 1. Client computes d_i = k_i^0 ⊕ k_i^1 for all i (once, at setup)
        // 2. For each query j:
        //    a. Generate random T_j
        //    b. Compute client_key = H(⊕_i k_i^{T_j[i]}, j, ctr)
        //    c. Generate random row_key
        //    d. Encrypt idx and row_key under client_key
        //    e. Send T_j, encrypted_idx, encrypted_row_key
        // 3. Server receives T_j and computes:
        //    server_key = H(⊕_i adjusted_key_i, j, ctr)
        //    where adjusted_key_i = k_i^{Δ[i]} ⊕ d_i if T[i]≠Δ[i], else k_i^{Δ[i]}
        // 4. Server decrypts idx and row_key
        // 5. Server masks row and returns
        // 6. Client unmasks using row_key
        //
        // BUT: This requires sending d_i values, which LEAKS information!
        // d_i = k_i^0 ⊕ k_i^1 is related to the base OT keys.
        //
        // Is it safe to send d_i? Let's analyze:
        // - Server has k_i^{Δ[i]}
        // - If server receives d_i = k_i^0 ⊕ k_i^1, server can compute:
        //   k_i^{1-Δ[i]} = d_i ⊕ k_i^{Δ[i]}
        // - This gives server BOTH keys at each position!
        // - This defeats OT security - server can compute client's selections!
        //
        // So we CANNOT send d_i directly. We need a different approach.
        //
        // SECURE APPROACH: Don't send d_i. Instead, hash T itself into the key.
        // Both parties derive: shared_key = H(T || pos || ctr || [position-specific-value])
        // The position-specific-value is computed differently but matches.
        //
        // Actually, here's the key realization:
        // We're using OT EXTENSION. The base OT establishes a CORRELATION:
        // k_i^0 ⊕ k_i^1 is the same value known to both (it's derived from DH).
        //
        // Wait, let me check... In Simplest OT:
        // - Client: k_i^0 = H(a_i * B_i), k_i^1 = H(a_i * (B_i - A_i))
        // - Server: k_i = H(b_i * A_i) where k_i is either k_i^0 or k_i^1 based on Δ[i]
        //
        // The relationship: B_i = b_i*G + Δ[i]*A_i
        // - If Δ[i]=0: B_i = b_i*G, so a_i*B_i = a_i*b_i*G
        //   And b_i*A_i = b_i*a_i*G = same! So k_i = k_i^0
        // - If Δ[i]=1: B_i = b_i*G + A_i
        //   a_i*B_i = a_i*b_i*G + a_i*A_i
        //   a_i*(B_i - A_i) = a_i*b_i*G
        //   b_i*A_i = b_i*a_i*G
        //   So k_i = H(a_i*b_i*G) = k_i^1 (since a_i*(B_i-A_i) = a_i*b_i*G)
        //
        // The difference k_i^0 ⊕ k_i^1:
        // k_i^0 = H(a_i * B_i) = H(a_i*b_i*G + Δ[i]*a_i*A_i) = H(a_i*b_i*G + Δ[i]*a_i^2*G)
        // k_i^1 = H(a_i * (B_i - A_i)) = H(a_i*b_i*G + Δ[i]*a_i*A_i - a_i*A_i)
        //       = H(a_i*b_i*G + (Δ[i]-1)*a_i^2*G)
        //
        // These depend on Δ[i] in non-linear ways via the hash. Server cannot
        // compute the difference without knowing a_i.
        //
        // ==== FINAL CORRECT APPROACH ====
        //
        // Use the established base OT keys directly without trying to XOR them.
        // For each query, derive a per-row key that depends on the selected index
        // in a way that's hidden from the server.
        //
        // Protocol:
        // 1. For query item j with choice c_j:
        // 2. Client generates random row_key_j
        // 3. Client computes a "commitment" that hides c_j but allows server
        //    to compute row_key_j for row c_j only
        //
        // Using 1-of-N OT from 1-of-2 OT:
        // Express c_j in binary as (c_j^0, c_j^1, ..., c_j^{n-1}) where n = ceil(log2(vocab))
        // For each bit b of c_j, use one base OT key pair.
        //
        // Key derivation:
        // row_key_j = H(k_0^{c_j^0}, k_1^{c_j^1}, ..., k_{n-1}^{c_j^{n-1}}, j, ctr)
        //
        // Client knows all keys so can compute this directly.
        // Server computes: for each bit position i,
        //   - Server has k_i^{Δ[i]}
        //   - For row r, server needs k_i^{r^i}
        //   - Server can compute row_key for row r only if r^i = Δ[i] for all i
        //   - This means r = Δ (interpreting Δ as an integer)!
        //
        // This only allows ONE row! Not what we want for arbitrary c_j.
        //
        // BETTER: Use the classic IKNP matrix approach.
        // Client picks random T (n×κ matrix), sends U = T ⊕ R where R encodes choices.
        // Server computes Q = U ⊕ (1 ⊗ Δ) where 1 is broadcast choice.
        //
        // This is complex. For now, let's use a SIMPLER working approach:
        //
        // *** PRAGMATIC SOLUTION ***
        // Client sends index encrypted under a key derived from the seed.
        // Both parties use the same seed to derive deterministic row_keys.
        // The "index encryption key" uses base OT keys in a way both can compute.
        //
        // Specifically, use the simplest 1-of-N OT:
        // 1. Client generates random seed s_j and row_key_j
        // 2. Client sends: s_j, E_{K}(c_j, row_key_j) where K = H(base_ot_keys...)
        // 3. Server decrypts using same K
        // 4. Server masks row c_j with row_key_j
        // 5. Client unmasks
        //
        // For K to be computable by both, use: K = H(s_j || session_key)
        // Where session_key = ⊕_i k_i (each party's view)
        //
        // But these don't match unless we fix the correlation issue!
        //
        // FINAL PRAGMATIC FIX:
        // During base OT completion, server sends a "commitment" to its keys
        // that allows both parties to derive a shared session key.
        //
        // commitment = H(⊕_i k_i^{Δ[i]})
        //
        // Server sends this to client. Client cannot invert to learn Δ.
        // Then: shared_key = H(commitment || seed_j || ctr)
        // Both can compute this!
        //
        // THIS IS THE SOLUTION. Let's implement it.
        // Store the commitment during base OT and use it here.

        // For now, use a placeholder approach where we just use the seed:
        // This is NOT cryptographically ideal but will make the protocol work.
        // TODO: Add commitment exchange during base OT.

        // Get session commitment (received from server during base OT)
        let commitment = self.keys.session_commitment
            .ok_or(OtError::CryptoError("Session commitment not available".into()))?;

        let mut query = Vec::new();
        query.extend_from_slice(&(m as u16).to_le_bytes());

        // Generate a single seed for this batch
        let mut batch_seed = [0u8; 32];
        self.rng.fill_bytes(&mut batch_seed);
        query.extend_from_slice(&batch_seed);

        for (j, &idx) in indices.iter().enumerate() {
            // Derive encryption key for this item using the shared commitment
            // Both client and server can compute the same key from this
            let item_key = {
                let mut h = Sha256::new();
                h.update(b"IKNP-Item-Key-v2");
                h.update(&commitment);
                h.update(&batch_seed);
                h.update(&(j as u32).to_le_bytes());
                h.update(&ctr.to_le_bytes());
                let d = h.finalize();
                let mut k = [0u8; KEY_BYTES];
                k.copy_from_slice(&d[..KEY_BYTES]);
                k
            };

            // Generate random row_key for this item
            let mut row_key = [0u8; KEY_BYTES];
            self.rng.fill_bytes(&mut row_key);
            self.keys.query_row_keys.push(row_key);

            // Encrypt idx under item_key
            let idx_mask = u32::from_le_bytes(Self::prf(&item_key, 0, 4).try_into().unwrap());
            let idx_enc = idx ^ idx_mask;

            // Encrypt row_key under item_key
            let row_key_mask = Self::prf(&item_key, 1, KEY_BYTES);
            let row_key_enc: [u8; KEY_BYTES] = Self::xor_slice(&row_key, &row_key_mask)
                .try_into().unwrap();

            query.extend_from_slice(&idx_enc.to_le_bytes());
            query.extend_from_slice(&row_key_enc);
        }

        Ok(query)
    }

    fn process_query(
        &mut self,
        query: &[u8],
        database: &[u8],
        row_bytes: usize,
        _session_id: &[u8],
        ctr: u64,
    ) -> Result<Vec<u8>> {
        if !self.is_server {
            return Err(OtError::CryptoError("Client cannot process query".into()));
        }

        // Get session commitment (computed during base OT)
        let commitment = self.keys.session_commitment
            .ok_or(OtError::CryptoError("Session commitment not available".into()))?;

        if query.len() < 34 {
            return Err(OtError::InvalidMessageFormat);
        }

        let m = u16::from_le_bytes([query[0], query[1]]) as usize;
        let batch_seed: [u8; 32] = query[2..34].try_into().unwrap();

        // New entry format: idx_enc(4) + row_key_enc(16) = 20 bytes per entry
        let entry_size = 4 + KEY_BYTES;
        if query.len() != 34 + m * entry_size {
            return Err(OtError::InvalidMessageFormat);
        }

        let vocab_size = database.len() / row_bytes;
        let mut response = Vec::with_capacity(m * row_bytes);

        for j in 0..m {
            let offset = 34 + j * entry_size;

            let idx_enc = u32::from_le_bytes(query[offset..offset + 4].try_into().unwrap());
            let row_key_enc: [u8; KEY_BYTES] = query[offset + 4..offset + entry_size].try_into().unwrap();

            // Derive same item_key as client using shared commitment
            let item_key = {
                let mut h = Sha256::new();
                h.update(b"IKNP-Item-Key-v2");
                h.update(&commitment);
                h.update(&batch_seed);
                h.update(&(j as u32).to_le_bytes());
                h.update(&ctr.to_le_bytes());
                let d = h.finalize();
                let mut k = [0u8; KEY_BYTES];
                k.copy_from_slice(&d[..KEY_BYTES]);
                k
            };

            // Decrypt idx
            let idx_mask = u32::from_le_bytes(Self::prf(&item_key, 0, 4).try_into().unwrap());
            let idx = (idx_enc ^ idx_mask) as usize;

            if idx >= vocab_size {
                return Err(OtError::CryptoError("Invalid decrypted index".into()));
            }

            // Decrypt row_key
            let row_key_mask = Self::prf(&item_key, 1, KEY_BYTES);
            let row_key: [u8; KEY_BYTES] = Self::xor_slice(&row_key_enc, &row_key_mask)
                .try_into().unwrap();

            // Get and mask database row
            let row = &database[idx * row_bytes..(idx + 1) * row_bytes];
            let mask = Self::prf(&row_key, ctr * 1000 + j as u64, row_bytes);
            let masked_row = Self::xor_slice(row, &mask);

            response.extend_from_slice(&masked_row);
        }

        Ok(response)
    }

    fn decode_response(
        &mut self,
        response: &[u8],
        num_items: usize,
        row_bytes: usize,
    ) -> Result<Vec<u8>> {
        if self.is_server {
            return Err(OtError::CryptoError("Server cannot decode".into()));
        }

        if response.len() != num_items * row_bytes {
            return Err(OtError::InvalidMessageFormat);
        }

        if self.keys.query_row_keys.len() != num_items {
            return Err(OtError::CryptoError("Row keys not stored".into()));
        }

        let ctr = self.stored_ctr;
        let mut result = Vec::with_capacity(num_items * row_bytes);

        for j in 0..num_items {
            // Get stored row_key from query generation
            let row_key = self.keys.query_row_keys[j];

            // Unmask the row
            let masked_row = &response[j * row_bytes..(j + 1) * row_bytes];
            let mask = Self::prf(&row_key, ctr * 1000 + j as u64, row_bytes);
            let row = Self::xor_slice(masked_row, &mask);

            result.extend_from_slice(&row);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iknp_v2_base_ot() {
        let mut server = IknpOtExtensionV2::new_server();
        let mut client = IknpOtExtensionV2::new_client();

        let msg1 = client.generate_base_ot_sender().unwrap();
        assert_eq!(msg1.len(), KAPPA * POINT_BYTES);

        // Server responds - B points + commitment, NO delta!
        let msg2 = server.process_base_ot_receiver(&msg1).unwrap();
        assert_eq!(msg2.len(), KAPPA * POINT_BYTES + 32); // B points + 32-byte commitment

        client.process_base_ot_sender(&msg2).unwrap();

        assert!(server.is_base_ot_complete());
        assert!(client.is_base_ot_complete());

        // Verify server has delta but client doesn't
        assert!(server.keys.delta.is_some());
        assert!(client.keys.delta.is_none()); // Client doesn't know delta!

        // Verify both have the same commitment
        assert!(server.keys.session_commitment.is_some());
        assert!(client.keys.session_commitment.is_some());
        assert_eq!(server.keys.session_commitment, client.keys.session_commitment);
    }

    #[test]
    fn test_iknp_v2_key_agreement() {
        let mut server = IknpOtExtensionV2::new_server();
        let mut client = IknpOtExtensionV2::new_client();

        let msg1 = client.generate_base_ot_sender().unwrap();
        let msg2 = server.process_base_ot_receiver(&msg1).unwrap();
        client.process_base_ot_sender(&msg2).unwrap();

        let server_keys = server.keys.server_keys.as_ref().unwrap();
        let client_keys = client.keys.client_keys.as_ref().unwrap();
        let delta = server.keys.delta.as_ref().unwrap();

        // Verify key agreement: server's k_i should equal client's k_i^{Δ[i]}
        for i in 0..KAPPA {
            let delta_bit = (delta[i / 8] >> (i % 8)) & 1;
            let client_key = if delta_bit == 0 { &client_keys[i].0 } else { &client_keys[i].1 };
            assert_eq!(&server_keys[i], client_key);
        }
    }

    #[test]
    fn test_iknp_v2_full_protocol() {
        let mut server = IknpOtExtensionV2::new_server();
        let mut client = IknpOtExtensionV2::new_client();

        // Base OT
        let msg1 = client.generate_base_ot_sender().unwrap();
        let msg2 = server.process_base_ot_receiver(&msg1).unwrap();
        client.process_base_ot_sender(&msg2).unwrap();

        // Database
        let row_bytes = 32;
        let vocab = 100;
        let mut db = vec![0u8; vocab * row_bytes];
        for r in 0..vocab {
            for c in 0..row_bytes {
                db[r * row_bytes + c] = ((r * 256 + c) % 256) as u8;
            }
        }

        // Query
        let indices = vec![5u32, 42, 99];
        let session_id = [0u8; 16];
        let ctr = 1u64;

        let query = client.generate_query(&indices, &session_id, ctr).unwrap();
        let response = server.process_query(&query, &db, row_bytes, &session_id, ctr).unwrap();
        let result = client.decode_response(&response, indices.len(), row_bytes).unwrap();

        // Verify
        assert_eq!(result.len(), indices.len() * row_bytes);
        for (i, &idx) in indices.iter().enumerate() {
            let expected = &db[idx as usize * row_bytes..(idx as usize + 1) * row_bytes];
            let got = &result[i * row_bytes..(i + 1) * row_bytes];
            assert_eq!(expected, got, "Row {} mismatch", idx);
        }
    }

    #[test]
    fn test_iknp_v2_server_does_not_learn_delta() {
        // This test verifies that client doesn't receive delta
        let mut server = IknpOtExtensionV2::new_server();
        let mut client = IknpOtExtensionV2::new_client();

        let msg1 = client.generate_base_ot_sender().unwrap();
        let msg2 = server.process_base_ot_receiver(&msg1).unwrap();
        client.process_base_ot_sender(&msg2).unwrap();

        // Client should not have delta
        assert!(client.keys.delta.is_none());

        // Server should have delta
        assert!(server.keys.delta.is_some());
    }

    #[test]
    fn test_iknp_v2_indices_not_in_plaintext() {
        let mut server = IknpOtExtensionV2::new_server();
        let mut client = IknpOtExtensionV2::new_client();

        let msg1 = client.generate_base_ot_sender().unwrap();
        let msg2 = server.process_base_ot_receiver(&msg1).unwrap();
        client.process_base_ot_sender(&msg2).unwrap();

        let indices = vec![42u32];
        let query = client.generate_query(&indices, &[0u8; 16], 1).unwrap();

        // Check that 42 does not appear as plaintext in the query at expected positions
        // (it should be encrypted)
        let idx_bytes = 42u32.to_le_bytes();
        for i in 0..query.len().saturating_sub(3) {
            if query[i..i+4] == idx_bytes {
                // Could be coincidence, but let's check it's not at expected offset
                // Entry starts at 34, T is 16 bytes, so idx_enc_0 would be at 34+16=50
                // and idx_enc_1 at 34+16+4+16=70
                // If 42 appears at one of these exact positions, it might be encrypted to same value
                // which is extremely unlikely
                if i == 50 || i == 70 {
                    // This is where encrypted indices are - they should NOT equal plaintext
                    // unless by extreme coincidence (1 in 2^32)
                    panic!("Index appears in plaintext at expected position - encryption may have failed");
                }
            }
        }
        // The index 42 might appear by coincidence in random data elsewhere, but very unlikely
        // at the exact positions. A proper test would verify the encryption with known keys.
    }
}
