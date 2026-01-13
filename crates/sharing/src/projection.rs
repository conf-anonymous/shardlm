//! Secure QKV projections for private attention
//!
//! Computes Q = hidden @ W_q, K = hidden @ W_k, V = hidden @ W_v
//! using secret sharing so the server never sees the plaintext input or output.

use shardlm_fixed_point::FixedVector;

use crate::error::{Result, SharingError};
use crate::share::Share;
use crate::matrix::SharedMatrix;
use crate::kv_cache::share_kv;

/// Client-side projection state
pub struct ProjectionClient {
    /// Hidden dimension
    hidden_size: usize,
    /// Number of Q heads
    num_heads: usize,
    /// Number of KV heads (for GQA)
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Fixed-point scale
    scale: u8,
}

/// Server-side projection state with weight matrices
pub struct ProjectionServer {
    /// W_q weight matrix [hidden_size × (num_heads * head_dim)]
    w_q: SharedMatrix,
    /// W_k weight matrix [hidden_size × (num_kv_heads * head_dim)]
    w_k: SharedMatrix,
    /// W_v weight matrix [hidden_size × (num_kv_heads * head_dim)]
    w_v: SharedMatrix,
    /// W_o weight matrix [(num_heads * head_dim) × hidden_size]
    w_o: SharedMatrix,
    /// Number of Q heads (kept for validation)
    #[allow(dead_code)]
    num_heads: usize,
    /// Number of KV heads (kept for validation)
    #[allow(dead_code)]
    num_kv_heads: usize,
    /// Head dimension (kept for validation)
    #[allow(dead_code)]
    head_dim: usize,
    /// Fixed-point scale
    scale: u8,
}

/// Result of QKV projection
#[derive(Debug)]
pub struct QkvProjection {
    /// Client's Q share [num_heads * head_dim]
    pub q_client: Share,
    /// Server's Q share
    pub q_server: Share,
    /// Client's K share [num_kv_heads * head_dim]
    pub k_client: Share,
    /// Server's K share
    pub k_server: Share,
    /// Client's V share [num_kv_heads * head_dim]
    pub v_client: Share,
    /// Server's V share
    pub v_server: Share,
}

/// Result of output projection
#[derive(Debug)]
pub struct OutputProjection {
    /// Client's output share [hidden_size]
    pub client: Share,
    /// Server's output share
    pub server: Share,
}

impl ProjectionClient {
    /// Create projection client
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: u8,
    ) -> Self {
        Self {
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        }
    }

    /// Create a masked input share for secure projection
    ///
    /// Client has plaintext hidden state, creates masked version for server
    pub fn create_masked_input(&self, hidden: &FixedVector) -> Result<(Share, Share)> {
        if hidden.len() != self.hidden_size {
            return Err(SharingError::DimensionMismatch {
                expected: self.hidden_size,
                got: hidden.len(),
            });
        }

        // Split into client and server shares
        let (client_share, server_share) = share_kv(hidden);
        Ok((client_share, server_share))
    }

    /// Compute Q/K/V projections from pre-computed server products
    ///
    /// Server sends: W @ server_share (for each of Q, K, V)
    /// Client computes: (W @ server_share) + (W @ client_share) = W @ hidden
    /// Then creates new secret shares of the result
    pub fn finalize_qkv(
        &self,
        client_share: &Share,
        server_q_product: &FixedVector,
        server_k_product: &FixedVector,
        server_v_product: &FixedVector,
        w_q_data: &[i32],
        w_k_data: &[i32],
        w_v_data: &[i32],
    ) -> Result<QkvProjection> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        // Compute client's contribution: W @ client_share
        let q_client_contrib = self.matmul_client(client_share, w_q_data, q_dim)?;
        let k_client_contrib = self.matmul_client(client_share, w_k_data, kv_dim)?;
        let v_client_contrib = self.matmul_client(client_share, w_v_data, kv_dim)?;

        // Combine: result = client_contrib + server_product
        let q_full = self.combine_products(&q_client_contrib, server_q_product)?;
        let k_full = self.combine_products(&k_client_contrib, server_k_product)?;
        let v_full = self.combine_products(&v_client_contrib, server_v_product)?;

        // Re-share for secure attention computation
        let (q_client, q_server) = share_kv(&q_full);
        let (k_client, k_server) = share_kv(&k_full);
        let (v_client, v_server) = share_kv(&v_full);

        Ok(QkvProjection {
            q_client,
            q_server,
            k_client,
            k_server,
            v_client,
            v_server,
        })
    }

    /// Matrix-vector multiply for client's share contribution
    fn matmul_client(
        &self,
        input: &Share,
        weight: &[i32],
        out_dim: usize,
    ) -> Result<Vec<i64>> {
        let in_dim = input.len();
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

    /// Combine client and server products
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

        // Add and rescale
        let result: Vec<i32> = client_contrib
            .iter()
            .zip(&server_product.data)
            .map(|(&c, &s)| {
                // client_contrib is i64, server_product is already rescaled i32
                // We need to rescale client_contrib and add
                let c_rescaled = (c >> self.scale) as i32;
                c_rescaled.wrapping_add(s)
            })
            .collect();

        Ok(FixedVector::from_raw(result, self.scale))
    }

    /// Finalize output projection
    pub fn finalize_output(
        &self,
        context_client: &Share,
        server_product: &FixedVector,
        w_o_data: &[i32],
    ) -> Result<OutputProjection> {
        let in_dim = self.num_heads * self.head_dim;
        if context_client.len() != in_dim {
            return Err(SharingError::DimensionMismatch {
                expected: in_dim,
                got: context_client.len(),
            });
        }

        // Client contribution
        let client_contrib = self.matmul_client(context_client, w_o_data, self.hidden_size)?;

        // Combine
        let output_full = self.combine_products(&client_contrib, server_product)?;

        // Re-share
        let (client, server) = share_kv(&output_full);

        Ok(OutputProjection { client, server })
    }
}

impl ProjectionServer {
    /// Create projection server with weight matrices
    pub fn new(
        w_q: SharedMatrix,
        w_k: SharedMatrix,
        w_v: SharedMatrix,
        w_o: SharedMatrix,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        scale: u8,
    ) -> Self {
        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            num_kv_heads,
            head_dim,
            scale,
        }
    }

    /// Compute W @ server_share for Q, K, V
    ///
    /// Returns products that will be sent to client
    pub fn compute_qkv_products(
        &self,
        server_share: &Share,
    ) -> Result<(FixedVector, FixedVector, FixedVector)> {
        let q_product = self.matmul(&self.w_q, server_share)?;
        let k_product = self.matmul(&self.w_k, server_share)?;
        let v_product = self.matmul(&self.w_v, server_share)?;

        Ok((q_product, k_product, v_product))
    }

    /// Compute W_o @ server_share for output projection
    pub fn compute_output_product(&self, context_server: &Share) -> Result<FixedVector> {
        self.matmul(&self.w_o, context_server)
    }

    /// Matrix-vector multiply: W @ x, rescaling the result
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

        // Rescale
        let result: Vec<i32> = output.iter().map(|&x| (x >> self.scale) as i32).collect();
        Ok(FixedVector::from_raw(result, self.scale))
    }

    /// Get weight matrix data for client computation
    pub fn get_w_q_data(&self) -> &[i32] {
        &self.w_q.data
    }

    pub fn get_w_k_data(&self) -> &[i32] {
        &self.w_k.data
    }

    pub fn get_w_v_data(&self) -> &[i32] {
        &self.w_v.data
    }

    pub fn get_w_o_data(&self) -> &[i32] {
        &self.w_o.data
    }
}

/// Complete QKV projection for a single step
///
/// This coordinates client and server to compute Q, K, V from hidden state.
pub fn compute_qkv_projection(
    client: &ProjectionClient,
    server: &ProjectionServer,
    hidden: &FixedVector,
) -> Result<QkvProjection> {
    // Step 1: Client creates masked input
    let (client_share, server_share) = client.create_masked_input(hidden)?;

    // Step 2: Server computes products with its shares
    let (q_prod, k_prod, v_prod) = server.compute_qkv_products(&server_share)?;

    // Step 3: Client finalizes with its share
    let qkv = client.finalize_qkv(
        &client_share,
        &q_prod,
        &k_prod,
        &v_prod,
        server.get_w_q_data(),
        server.get_w_k_data(),
        server.get_w_v_data(),
    )?;

    Ok(qkv)
}

/// Complete output projection
pub fn compute_output_projection(
    client: &ProjectionClient,
    server: &ProjectionServer,
    context_client: &Share,
    context_server: &Share,
) -> Result<OutputProjection> {
    // Server computes its product
    let server_product = server.compute_output_product(context_server)?;

    // Client finalizes
    client.finalize_output(context_client, &server_product, server.get_w_o_data())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shardlm_fixed_point::DEFAULT_SCALE;

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
    fn test_qkv_projection_dimensions() {
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        let client = ProjectionClient::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            DEFAULT_SCALE,
        );

        let w_q = create_test_matrix(hidden_size, num_heads * head_dim, DEFAULT_SCALE);
        let w_k = create_test_matrix(hidden_size, num_kv_heads * head_dim, DEFAULT_SCALE);
        let w_v = create_test_matrix(hidden_size, num_kv_heads * head_dim, DEFAULT_SCALE);
        let w_o = create_test_matrix(num_heads * head_dim, hidden_size, DEFAULT_SCALE);

        let server = ProjectionServer::new(
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            num_kv_heads,
            head_dim,
            DEFAULT_SCALE,
        );

        let hidden = FixedVector::from_f64_slice_default(
            &(0..hidden_size).map(|i| i as f64 * 0.1).collect::<Vec<_>>()
        ).unwrap();

        let qkv = compute_qkv_projection(&client, &server, &hidden).unwrap();

        // Check dimensions
        assert_eq!(qkv.q_client.len(), num_heads * head_dim);
        assert_eq!(qkv.q_server.len(), num_heads * head_dim);
        assert_eq!(qkv.k_client.len(), num_kv_heads * head_dim);
        assert_eq!(qkv.k_server.len(), num_kv_heads * head_dim);
        assert_eq!(qkv.v_client.len(), num_kv_heads * head_dim);
        assert_eq!(qkv.v_server.len(), num_kv_heads * head_dim);
    }

    #[test]
    fn test_output_projection() {
        let hidden_size = 8;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;

        let client = ProjectionClient::new(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            DEFAULT_SCALE,
        );

        let w_q = create_test_matrix(hidden_size, num_heads * head_dim, DEFAULT_SCALE);
        let w_k = create_test_matrix(hidden_size, num_kv_heads * head_dim, DEFAULT_SCALE);
        let w_v = create_test_matrix(hidden_size, num_kv_heads * head_dim, DEFAULT_SCALE);
        let w_o = create_test_matrix(num_heads * head_dim, hidden_size, DEFAULT_SCALE);

        let server = ProjectionServer::new(
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            num_kv_heads,
            head_dim,
            DEFAULT_SCALE,
        );

        // Create context shares
        let context = FixedVector::from_f64_slice_default(
            &vec![1.0; num_heads * head_dim]
        ).unwrap();
        let (context_client, context_server) = share_kv(&context);

        let output = compute_output_projection(
            &client,
            &server,
            &context_client,
            &context_server,
        ).unwrap();

        assert_eq!(output.client.len(), hidden_size);
        assert_eq!(output.server.len(), hidden_size);
    }
}
