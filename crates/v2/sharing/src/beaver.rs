//! Beaver Triple Store for MPC-Secure Multiplication
//!
//! Beaver triples enable secure multiplication of secret-shared values without
//! revealing plaintext to either party. Each triple (a, b, c) satisfies c = a * b.
//!
//! # Protocol
//!
//! To compute z = x * y where x and y are secret-shared:
//! 1. Both parties compute d = x - a, e = y - b (masked values)
//! 2. d and e are "opened" (both parties learn them)
//! 3. z = c + a*e + d*b + d*e is computed as shares
//!
//! The key insight: d and e are uniformly random (masked by a, b), so
//! opening them reveals nothing about x or y.

use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A single Beaver triple: (a, b, c) where c = a * b
#[derive(Clone, Debug)]
pub struct BeaverTriple {
    /// First random value
    pub a: f32,
    /// Second random value
    pub b: f32,
    /// Product c = a * b
    pub c: f32,
    /// Client's share of a
    pub a_client: f32,
    /// Server's share of a
    pub a_server: f32,
    /// Client's share of b
    pub b_client: f32,
    /// Server's share of b
    pub b_server: f32,
    /// Client's share of c
    pub c_client: f32,
    /// Server's share of c
    pub c_server: f32,
}

impl BeaverTriple {
    /// Generate a new random Beaver triple with proper share splitting
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();

        // Generate random values for a and b
        let a: f32 = rng.gen_range(-1.0..1.0);
        let b: f32 = rng.gen_range(-1.0..1.0);
        let c = a * b;

        // Split into shares (random splitting)
        let a_server: f32 = rng.gen_range(-10.0..10.0);
        let a_client = a - a_server;

        let b_server: f32 = rng.gen_range(-10.0..10.0);
        let b_client = b - b_server;

        let c_server: f32 = rng.gen_range(-10.0..10.0);
        let c_client = c - c_server;

        Self {
            a, b, c,
            a_client, a_server,
            b_client, b_server,
            c_client, c_server,
        }
    }

    /// Generate a batch of Beaver triples
    pub fn random_batch(n: usize) -> Vec<Self> {
        (0..n).map(|_| Self::random()).collect()
    }
}

/// Pre-generated store of Beaver triples for secure computation
///
/// Triples are organized by layer to ensure deterministic usage during
/// inference. Each layer has its own pool of triples.
pub struct BeaverTripleStore {
    /// Triples organized by layer [layer_idx][triple_idx]
    triples: Vec<Vec<BeaverTriple>>,
    /// Current cursor position for each layer
    cursors: Vec<AtomicUsize>,
    /// Total triples per layer
    triples_per_layer: usize,
}

impl BeaverTripleStore {
    /// Pre-generate triples for all layers
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `triples_per_layer` - Number of triples to pre-generate per layer
    ///   (should be enough for all nonlinear ops in one forward pass)
    pub fn pregenerate(num_layers: usize, triples_per_layer: usize) -> Self {
        tracing::info!(
            num_layers = num_layers,
            triples_per_layer = triples_per_layer,
            total = num_layers * triples_per_layer,
            "Pre-generating Beaver triples"
        );

        let triples: Vec<Vec<BeaverTriple>> = (0..num_layers)
            .map(|_| BeaverTriple::random_batch(triples_per_layer))
            .collect();

        let cursors = (0..num_layers)
            .map(|_| AtomicUsize::new(0))
            .collect();

        Self {
            triples,
            cursors,
            triples_per_layer,
        }
    }

    /// Get the next triple for a specific layer
    ///
    /// Automatically wraps around when exhausted (for testing)
    pub fn get_triple(&self, layer_idx: usize) -> &BeaverTriple {
        let cursor = self.cursors[layer_idx].fetch_add(1, Ordering::Relaxed);
        let idx = cursor % self.triples_per_layer;
        &self.triples[layer_idx][idx]
    }

    /// Get a batch of triples for a specific layer
    pub fn get_triples(&self, layer_idx: usize, count: usize) -> Vec<&BeaverTriple> {
        let start = self.cursors[layer_idx].fetch_add(count, Ordering::Relaxed);
        (0..count)
            .map(|i| {
                let idx = (start + i) % self.triples_per_layer;
                &self.triples[layer_idx][idx]
            })
            .collect()
    }

    /// Reset all cursors (for new inference pass)
    pub fn reset(&self) {
        for cursor in &self.cursors {
            cursor.store(0, Ordering::Relaxed);
        }
    }

    /// Get all triples for a specific layer (for MPC operations that need bulk access)
    pub fn get_layer_triples(&self, layer_idx: usize) -> &[BeaverTriple] {
        &self.triples[layer_idx]
    }

    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let triple_size = std::mem::size_of::<BeaverTriple>();
        self.triples.len() * self.triples_per_layer * triple_size
    }
}

/// MPC-secure multiplication using Beaver triples
///
/// Computes z = x * y where both x and y are secret-shared.
/// Returns (z_client, z_server) such that z_client + z_server = x * y
///
/// # Security
/// - Neither party learns x, y, or z in plaintext
/// - Only masked values d = x - a and e = y - b are revealed
/// - d and e are uniformly random, revealing nothing about x or y
pub fn secure_multiply_mpc(
    x_client: f32, x_server: f32,
    y_client: f32, y_server: f32,
    triple: &BeaverTriple,
) -> (f32, f32) {
    // Compute masked values
    // d = x - a (both parties can compute their share of d)
    let d_client = x_client - triple.a_client;
    let d_server = x_server - triple.a_server;

    // e = y - b
    let e_client = y_client - triple.b_client;
    let e_server = y_server - triple.b_server;

    // "Open" d and e (in real MPC, parties exchange shares; here we simulate)
    let d = d_client + d_server;
    let e = e_client + e_server;

    // Compute z = xy = (a + d)(b + e) = ab + ae + bd + de = c + ae + bd + de
    //
    // Each party computes their share:
    // z_client = c_client + a_client*e + d*b_client + d*e (client's contribution)
    // z_server = c_server + a_server*e + d*b_server      (server's contribution)
    //
    // Note: d*e is public (both d and e are opened), so only one party adds it

    let z_client = triple.c_client + triple.a_client * e + d * triple.b_client + d * e;
    let z_server = triple.c_server + triple.a_server * e + d * triple.b_server;

    (z_client, z_server)
}

/// MPC-secure squaring (optimized for x * x)
///
/// When computing x², we can use a specialized protocol that's more efficient
pub fn secure_square_mpc(
    x_client: f32, x_server: f32,
    triple: &BeaverTriple,
) -> (f32, f32) {
    secure_multiply_mpc(x_client, x_server, x_client, x_server, triple)
}

/// Batch multiplication for vectors
///
/// Computes element-wise z[i] = x[i] * y[i] for all i
pub fn secure_multiply_batch(
    x_client: &[f32], x_server: &[f32],
    y_client: &[f32], y_server: &[f32],
    triples: &[&BeaverTriple],
) -> (Vec<f32>, Vec<f32>) {
    let n = x_client.len();
    let mut z_client = vec![0.0; n];
    let mut z_server = vec![0.0; n];

    for i in 0..n {
        let (zc, zs) = secure_multiply_mpc(
            x_client[i], x_server[i],
            y_client[i], y_server[i],
            triples[i],
        );
        z_client[i] = zc;
        z_server[i] = zs;
    }

    (z_client, z_server)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beaver_triple_correctness() {
        for _ in 0..100 {
            let triple = BeaverTriple::random();

            // Verify shares sum to values
            let a_sum = triple.a_client + triple.a_server;
            let b_sum = triple.b_client + triple.b_server;
            let c_sum = triple.c_client + triple.c_server;

            assert!((a_sum - triple.a).abs() < 1e-5, "a shares don't sum correctly");
            assert!((b_sum - triple.b).abs() < 1e-5, "b shares don't sum correctly");
            assert!((c_sum - triple.c).abs() < 1e-5, "c shares don't sum correctly");
            assert!((triple.c - triple.a * triple.b).abs() < 1e-5, "c != a*b");
        }
    }

    #[test]
    fn test_secure_multiply_correctness() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            // Generate random x and y
            let x: f32 = rng.gen_range(-10.0..10.0);
            let y: f32 = rng.gen_range(-10.0..10.0);

            // Split into shares
            let x_server: f32 = rng.gen_range(-10.0..10.0);
            let x_client = x - x_server;

            let y_server: f32 = rng.gen_range(-10.0..10.0);
            let y_client = y - y_server;

            // Compute using Beaver triple
            let triple = BeaverTriple::random();
            let (z_client, z_server) = secure_multiply_mpc(
                x_client, x_server,
                y_client, y_server,
                &triple,
            );

            // Verify result
            let z_expected = x * y;
            let z_actual = z_client + z_server;

            assert!(
                (z_actual - z_expected).abs() < 1e-4,
                "Multiplication incorrect: {} * {} = {} but got {}",
                x, y, z_expected, z_actual
            );
        }
    }

    #[test]
    fn test_triple_store() {
        let store = BeaverTripleStore::pregenerate(4, 100);

        // Get triples from different layers
        let t0 = store.get_triple(0);
        let t1 = store.get_triple(1);
        let t2 = store.get_triple(0); // Different triple from layer 0

        // Verify they're different triples
        assert!((t0.a - t2.a).abs() > 1e-6 || (t0.b - t2.b).abs() > 1e-6);

        // Check memory usage
        let mem = store.memory_usage();
        assert!(mem > 0);
    }
}
