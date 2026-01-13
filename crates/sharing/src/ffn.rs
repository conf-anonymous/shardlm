//! Secure Feed-Forward Network with client-side activation
//!
//! ## FFN Architecture (SwiGLU for Llama/TinyLlama)
//!
//! FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
//!
//! Where:
//! - gate_proj: [hidden_size -> intermediate_size]
//! - up_proj: [hidden_size -> intermediate_size]
//! - down_proj: [intermediate_size -> hidden_size]
//! - silu(x) = x * sigmoid(x)
//!
//! ## Privacy Model
//!
//! Client computes activation (silu) since server can't compute nonlinear
//! functions on secret-shared data without expensive protocols.
//! The flow is:
//!
//! 1. Client/Server: Compute gate_proj and up_proj using secret sharing
//! 2. Client: Reconstruct gate and up, compute silu(gate) * up
//! 3. Client: Create new secret shares of intermediate result
//! 4. Client/Server: Compute down_proj using secret sharing

use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;
use crate::matrix::SharedMatrix;
use crate::kv_cache::share_kv;

/// Client-side FFN state
pub struct FfnClient {
    /// Hidden dimension
    hidden_size: usize,
    /// Intermediate dimension
    intermediate_size: usize,
    /// Fixed-point scale
    scale: u8,
}

/// Server-side FFN state with weight matrices
pub struct FfnServer {
    /// Gate projection weights [hidden_size × intermediate_size]
    w_gate: SharedMatrix,
    /// Up projection weights [hidden_size × intermediate_size]
    w_up: SharedMatrix,
    /// Down projection weights [intermediate_size × hidden_size]
    w_down: SharedMatrix,
    /// Fixed-point scale
    scale: u8,
}

/// Result of FFN forward pass
#[derive(Debug)]
pub struct FfnOutput {
    /// Client's output share [hidden_size]
    pub client: Share,
    /// Server's output share
    pub server: Share,
}

impl FfnClient {
    /// Create FFN client
    pub fn new(hidden_size: usize, intermediate_size: usize, scale: u8) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            scale,
        }
    }

    /// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// Compute activation from gate and up projections
    ///
    /// Takes reconstructed gate and up, computes silu(gate) * up
    pub fn compute_activation(&self, gate: &FixedVector, up: &FixedVector) -> Result<FixedVector> {
        if gate.len() != self.intermediate_size || up.len() != self.intermediate_size {
            return Err(SharingError::DimensionMismatch {
                expected: self.intermediate_size,
                got: gate.len(),
            });
        }

        let scale_factor = (1u64 << self.scale) as f64;

        // Convert to f64, apply silu * up, convert back
        let result: Vec<i32> = gate
            .data
            .iter()
            .zip(&up.data)
            .map(|(&g, &u)| {
                let g_f64 = g as f64 / scale_factor;
                let u_f64 = u as f64 / scale_factor;
                let activated = Self::silu(g_f64) * u_f64;
                (activated * scale_factor).round() as i32
            })
            .collect();

        Ok(FixedVector::from_raw(result, self.scale))
    }

    /// Finalize gate and up projections from server products
    pub fn finalize_projections(
        &self,
        client_share: &Share,
        server_gate_product: &FixedVector,
        server_up_product: &FixedVector,
        w_gate_data: &[i32],
        w_up_data: &[i32],
    ) -> Result<(FixedVector, FixedVector)> {
        // Compute client's contributions
        let gate_client = self.matmul_client(client_share, w_gate_data)?;
        let up_client = self.matmul_client(client_share, w_up_data)?;

        // Combine with server products
        let gate = self.combine_products(&gate_client, server_gate_product)?;
        let up = self.combine_products(&up_client, server_up_product)?;

        Ok((gate, up))
    }

    /// Finalize down projection
    pub fn finalize_down(
        &self,
        client_share: &Share,
        server_down_product: &FixedVector,
        w_down_data: &[i32],
    ) -> Result<FfnOutput> {
        let client_contrib = self.matmul_client_to_hidden(client_share, w_down_data)?;
        let output = self.combine_products(&client_contrib, server_down_product)?;

        let (client, server) = share_kv(&output);
        Ok(FfnOutput { client, server })
    }

    fn matmul_client(&self, input: &Share, weight: &[i32]) -> Result<Vec<i64>> {
        let in_dim = input.len();
        let out_dim = self.intermediate_size;
        if weight.len() != in_dim * out_dim {
            return Err(SharingError::DimensionMismatch {
                expected: in_dim * out_dim,
                got: weight.len(),
            });
        }

        let mut output = vec![0i64; out_dim];
        for i in 0..in_dim {
            let x = input.data[i] as i64;
            for j in 0..out_dim {
                output[j] += x * weight[i * out_dim + j] as i64;
            }
        }
        Ok(output)
    }

    fn matmul_client_to_hidden(&self, input: &Share, weight: &[i32]) -> Result<Vec<i64>> {
        let in_dim = input.len();
        let out_dim = self.hidden_size;
        if weight.len() != in_dim * out_dim {
            return Err(SharingError::DimensionMismatch {
                expected: in_dim * out_dim,
                got: weight.len(),
            });
        }

        let mut output = vec![0i64; out_dim];
        for i in 0..in_dim {
            let x = input.data[i] as i64;
            for j in 0..out_dim {
                output[j] += x * weight[i * out_dim + j] as i64;
            }
        }
        Ok(output)
    }

    fn combine_products(
        &self,
        client_contrib: &[i64],
        server_product: &FixedVector,
    ) -> Result<FixedVector> {
        if client_contrib.len() != server_product.len() {
            return Err(SharingError::DimensionMismatch {
                expected: client_contrib.len(),
                got: server_product.len(),
            });
        }

        let result: Vec<i32> = client_contrib
            .iter()
            .zip(&server_product.data)
            .map(|(&c, &s)| {
                let c_rescaled = (c >> self.scale) as i32;
                c_rescaled.wrapping_add(s)
            })
            .collect();

        Ok(FixedVector::from_raw(result, self.scale))
    }
}

impl FfnServer {
    /// Create FFN server with weight matrices
    pub fn new(w_gate: SharedMatrix, w_up: SharedMatrix, w_down: SharedMatrix, scale: u8) -> Self {
        Self {
            w_gate,
            w_up,
            w_down,
            scale,
        }
    }

    /// Compute gate and up projections on server's share
    pub fn compute_projections(
        &self,
        server_share: &Share,
    ) -> Result<(FixedVector, FixedVector)> {
        let gate = self.matmul(&self.w_gate, server_share)?;
        let up = self.matmul(&self.w_up, server_share)?;
        Ok((gate, up))
    }

    /// Compute down projection on server's intermediate share
    pub fn compute_down(&self, intermediate_server: &Share) -> Result<FixedVector> {
        self.matmul(&self.w_down, intermediate_server)
    }

    fn matmul(&self, weight: &SharedMatrix, input: &Share) -> Result<FixedVector> {
        if weight.rows != input.len() {
            return Err(SharingError::DimensionMismatch {
                expected: weight.rows,
                got: input.len(),
            });
        }

        let mut output = vec![0i64; weight.cols];
        for i in 0..weight.rows {
            let x = input.data[i] as i64;
            for j in 0..weight.cols {
                output[j] += x * weight.data[i * weight.cols + j] as i64;
            }
        }

        let result: Vec<i32> = output.iter().map(|&x| (x >> self.scale) as i32).collect();
        Ok(FixedVector::from_raw(result, self.scale))
    }

    /// Get weight data for client computation
    pub fn get_w_gate_data(&self) -> &[i32] {
        &self.w_gate.data
    }

    pub fn get_w_up_data(&self) -> &[i32] {
        &self.w_up.data
    }

    pub fn get_w_down_data(&self) -> &[i32] {
        &self.w_down.data
    }
}

/// Complete FFN forward pass
///
/// Coordinates client and server to compute FFN(x) = down(silu(gate(x)) * up(x))
pub fn compute_ffn(
    client: &FfnClient,
    server: &FfnServer,
    hidden_client: &Share,
    hidden_server: &Share,
) -> Result<FfnOutput> {
    // Step 1: Server computes gate and up projections on its share
    let (gate_server_prod, up_server_prod) = server.compute_projections(hidden_server)?;

    // Step 2: Client finalizes gate and up, reconstructing them
    let (gate, up) = client.finalize_projections(
        hidden_client,
        &gate_server_prod,
        &up_server_prod,
        server.get_w_gate_data(),
        server.get_w_up_data(),
    )?;

    // Step 3: Client computes activation (silu(gate) * up)
    let intermediate = client.compute_activation(&gate, &up)?;

    // Step 4: Re-share intermediate for down projection
    let (intermediate_client, intermediate_server) = share_kv(&intermediate);

    // Step 5: Server computes down projection on its share
    let down_server_prod = server.compute_down(&intermediate_server)?;

    // Step 6: Client finalizes down projection
    client.finalize_down(
        &intermediate_client,
        &down_server_prod,
        server.get_w_down_data(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;
    use crate::kv_cache::reconstruct_kv;

    fn create_test_matrix(rows: usize, cols: usize, scale: u8) -> SharedMatrix {
        // Small identity-like matrix for testing
        let mut data = vec![0i32; rows * cols];
        let one = 1i32 << scale;
        for i in 0..rows.min(cols) {
            data[i * cols + i] = one;
        }
        SharedMatrix::from_raw(data, rows, cols, scale).unwrap()
    }

    #[test]
    fn test_silu() {
        // silu(0) = 0
        assert!((FfnClient::silu(0.0) - 0.0).abs() < 1e-10);

        // silu(x) approaches x for large positive x
        let large = 10.0;
        assert!((FfnClient::silu(large) - large).abs() < 0.1);

        // silu(x) approaches 0 for large negative x
        assert!(FfnClient::silu(-10.0).abs() < 0.01);
    }

    #[test]
    fn test_ffn_dimensions() {
        let hidden_size = 8;
        let intermediate_size = 16;

        let client = FfnClient::new(hidden_size, intermediate_size, DEFAULT_SCALE);

        let w_gate = create_test_matrix(hidden_size, intermediate_size, DEFAULT_SCALE);
        let w_up = create_test_matrix(hidden_size, intermediate_size, DEFAULT_SCALE);
        let w_down = create_test_matrix(intermediate_size, hidden_size, DEFAULT_SCALE);

        let server = FfnServer::new(w_gate, w_up, w_down, DEFAULT_SCALE);

        let hidden = FixedVector::from_f64_slice_default(
            &(0..hidden_size).map(|i| i as f64 * 0.1).collect::<Vec<_>>()
        ).unwrap();
        let (hidden_client, hidden_server) = share_kv(&hidden);

        let output = compute_ffn(&client, &server, &hidden_client, &hidden_server).unwrap();

        assert_eq!(output.client.len(), hidden_size);
        assert_eq!(output.server.len(), hidden_size);
    }

    #[test]
    fn test_ffn_activation() {
        let client = FfnClient::new(8, 4, DEFAULT_SCALE);

        let gate = FixedVector::from_f64_slice_default(&[1.0, -1.0, 0.5, 2.0]).unwrap();
        let up = FixedVector::from_f64_slice_default(&[1.0, 1.0, 1.0, 1.0]).unwrap();

        let result = client.compute_activation(&gate, &up).unwrap();

        // Check that activation preserves sign properties of silu
        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let result_f64: Vec<f64> = result.data.iter().map(|&x| x as f64 / scale_factor).collect();

        // silu(1) * 1 ≈ 0.73
        assert!(result_f64[0] > 0.7 && result_f64[0] < 0.8);
        // silu(-1) * 1 ≈ -0.27
        assert!(result_f64[1] > -0.3 && result_f64[1] < -0.2);
    }

    #[test]
    fn test_ffn_reconstruction() {
        let hidden_size = 4;
        let intermediate_size = 8;

        let client = FfnClient::new(hidden_size, intermediate_size, DEFAULT_SCALE);

        let w_gate = create_test_matrix(hidden_size, intermediate_size, DEFAULT_SCALE);
        let w_up = create_test_matrix(hidden_size, intermediate_size, DEFAULT_SCALE);
        let w_down = create_test_matrix(intermediate_size, hidden_size, DEFAULT_SCALE);

        let server = FfnServer::new(w_gate, w_up, w_down, DEFAULT_SCALE);

        let hidden = FixedVector::from_f64_slice_default(&[1.0, 0.5, -0.3, 0.8]).unwrap();
        let (hidden_client, hidden_server) = share_kv(&hidden);

        let output = compute_ffn(&client, &server, &hidden_client, &hidden_server).unwrap();

        // Verify we can reconstruct
        let reconstructed = reconstruct_kv(&output.client, &output.server).unwrap();
        assert_eq!(reconstructed.len(), hidden_size);
    }
}
