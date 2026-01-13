//! Multi-GPU communication primitives for tensor parallelism
//!
//! This module implements communication operations needed for distributed
//! tensor parallel inference:
//!
//! - **All-Reduce**: Sum partial results from all GPUs (used after row-parallel layers)
//! - **All-Gather**: Collect sharded outputs to all GPUs (used for vocabulary embedding lookup)
//! - **Reduce-Scatter**: Reduce and distribute results (optimization)
//!
//! # Communication Pattern in Tensor Parallelism
//!
//! For a single transformer block with tensor parallelism:
//!
//! ```text
//! Input X (replicated on all GPUs)
//!     |
//!     v
//! [Column Parallel: Q_proj, gate_proj, up_proj]
//!     |  (no communication needed - each GPU computes partial output)
//!     v
//! [Local Computation: Attention, SwiGLU]
//!     |
//!     v
//! [Row Parallel: O_proj, down_proj]
//!     |
//!     v
//! All-Reduce (sum partial results)
//!     |
//!     v
//! Output Y (replicated on all GPUs)
//! ```

use crate::error::{Result, V2Error};
use crate::gpu::{GpuDevice, MultiGpuContext};

#[cfg(not(feature = "cuda"))]
use crate::tensor::DType;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Communication operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommOp {
    /// Sum reduction across all GPUs
    AllReduceSum,
    /// Max reduction across all GPUs
    AllReduceMax,
    /// Gather data from all GPUs to all GPUs
    AllGather,
    /// Scatter data from one GPU to all GPUs
    Scatter,
    /// Reduce and scatter (fused operation)
    ReduceScatter,
}

/// Multi-GPU communicator for collective operations
pub struct GpuCommunicator {
    /// GPU context
    ctx: MultiGpuContext,
    /// Number of GPUs
    num_gpus: usize,
    /// Staging buffers for CPU-mediated communication (fallback)
    staging_buffers: Vec<Vec<u8>>,
}

impl GpuCommunicator {
    /// Create a new communicator
    pub fn new(ctx: MultiGpuContext) -> Self {
        let num_gpus = ctx.num_gpus;
        let staging_buffers = vec![Vec::new(); num_gpus];

        Self {
            ctx,
            num_gpus,
            staging_buffers,
        }
    }

    /// Get number of GPUs
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }

    /// Get device reference
    pub fn device(&self, idx: usize) -> &GpuDevice {
        self.ctx.device(idx)
    }

    /// All-reduce sum operation
    ///
    /// Sums tensors across all GPUs, with result available on all GPUs.
    /// Uses ring all-reduce for efficiency.
    #[cfg(feature = "cuda")]
    pub fn all_reduce_sum(&mut self, slices: &mut [CudaSlice<f32>]) -> Result<()> {
        if slices.len() != self.num_gpus {
            return Err(V2Error::Gpu(format!(
                "Expected {} slices, got {}",
                self.num_gpus,
                slices.len()
            )));
        }

        // For now, use simple CPU-mediated all-reduce
        // TODO: Implement NCCL or ring all-reduce for efficiency
        self.cpu_mediated_all_reduce_f32(slices)
    }

    /// CPU-mediated all-reduce (fallback implementation)
    #[cfg(feature = "cuda")]
    fn cpu_mediated_all_reduce_f32(&mut self, slices: &mut [CudaSlice<f32>]) -> Result<()> {
        use std::ops::AddAssign;

        // Step 1: Copy all data to host
        let mut host_data: Vec<Vec<f32>> = Vec::with_capacity(self.num_gpus);
        for (gpu_id, slice) in slices.iter().enumerate() {
            let data = self.ctx.device(gpu_id).dtoh(slice)?;
            host_data.push(data);
        }

        // Step 2: Sum on CPU
        let len = host_data[0].len();
        let mut result = vec![0.0f32; len];
        for data in &host_data {
            for (i, &val) in data.iter().enumerate() {
                result[i].add_assign(val);
            }
        }

        // Step 3: Copy result back to all GPUs
        for (gpu_id, slice) in slices.iter_mut().enumerate() {
            let device = self.ctx.device(gpu_id);
            *slice = device.htod(&result)?;
        }

        Ok(())
    }

    /// All-reduce sum for BF16 data
    #[cfg(feature = "cuda")]
    pub fn all_reduce_sum_bf16(&mut self, slices: &mut [CudaSlice<u16>]) -> Result<()> {
        if slices.len() != self.num_gpus {
            return Err(V2Error::Gpu(format!(
                "Expected {} slices, got {}",
                self.num_gpus,
                slices.len()
            )));
        }

        self.cpu_mediated_all_reduce_bf16(slices)
    }

    #[cfg(feature = "cuda")]
    fn cpu_mediated_all_reduce_bf16(&mut self, slices: &mut [CudaSlice<u16>]) -> Result<()> {
        use half::bf16;

        // Step 1: Copy all data to host
        let mut host_data: Vec<Vec<u16>> = Vec::with_capacity(self.num_gpus);
        for (gpu_id, slice) in slices.iter().enumerate() {
            let data = self.ctx.device(gpu_id).dtoh(slice)?;
            host_data.push(data);
        }

        // Step 2: Sum on CPU (convert BF16 -> F32, sum, convert back)
        let len = host_data[0].len();
        let mut result_f32 = vec![0.0f32; len];

        for data in &host_data {
            for (i, &val_u16) in data.iter().enumerate() {
                let val = bf16::from_bits(val_u16).to_f32();
                result_f32[i] += val;
            }
        }

        // Convert back to BF16
        let result_bf16: Vec<u16> = result_f32
            .iter()
            .map(|&v| bf16::from_f32(v).to_bits())
            .collect();

        // Step 3: Copy result back to all GPUs
        for (gpu_id, slice) in slices.iter_mut().enumerate() {
            let device = self.ctx.device(gpu_id);
            *slice = device.htod(&result_bf16)?;
        }

        Ok(())
    }

    /// All-gather operation
    ///
    /// Gathers sharded data from all GPUs, with full result on all GPUs.
    /// Used for gathering vocabulary embeddings after sharded lookup.
    #[cfg(feature = "cuda")]
    pub fn all_gather_f32(
        &mut self,
        shards: &[CudaSlice<f32>],
        shard_sizes: &[usize],
    ) -> Result<Vec<CudaSlice<f32>>> {
        if shards.len() != self.num_gpus {
            return Err(V2Error::Gpu(format!(
                "Expected {} shards, got {}",
                self.num_gpus,
                shards.len()
            )));
        }

        // CPU-mediated all-gather
        self.cpu_mediated_all_gather_f32(shards, shard_sizes)
    }

    #[cfg(feature = "cuda")]
    fn cpu_mediated_all_gather_f32(
        &mut self,
        shards: &[CudaSlice<f32>],
        shard_sizes: &[usize],
    ) -> Result<Vec<CudaSlice<f32>>> {
        // Step 1: Copy all shards to host
        let mut host_shards: Vec<Vec<f32>> = Vec::with_capacity(self.num_gpus);
        for (gpu_id, shard) in shards.iter().enumerate() {
            let data = self.ctx.device(gpu_id).dtoh(shard)?;
            host_shards.push(data);
        }

        // Step 2: Concatenate on CPU
        let total_size: usize = shard_sizes.iter().sum();
        let mut gathered = Vec::with_capacity(total_size);
        for shard in &host_shards {
            gathered.extend_from_slice(shard);
        }

        // Step 3: Copy full result to all GPUs
        let mut result = Vec::with_capacity(self.num_gpus);
        for gpu_id in 0..self.num_gpus {
            let device = self.ctx.device(gpu_id);
            let slice = device.htod(&gathered)?;
            result.push(slice);
        }

        Ok(result)
    }

    /// Synchronize all GPUs
    pub fn synchronize_all(&self) -> Result<()> {
        self.ctx.synchronize_all()
    }
}

/// All-reduce operation using raw byte buffers (for non-CUDA builds)
#[cfg(not(feature = "cuda"))]
impl GpuCommunicator {
    pub fn all_reduce_sum_bytes(&mut self, buffers: &mut [Vec<u8>], dtype: DType) -> Result<()> {
        match dtype {
            DType::F32 => self.all_reduce_sum_f32_cpu(buffers),
            DType::BF16 => self.all_reduce_sum_bf16_cpu(buffers),
            _ => Err(V2Error::Gpu(format!("Unsupported dtype for all-reduce: {:?}", dtype))),
        }
    }

    fn all_reduce_sum_f32_cpu(&mut self, buffers: &mut [Vec<u8>]) -> Result<()> {
        let len = buffers[0].len() / 4; // f32 = 4 bytes
        let mut result = vec![0.0f32; len];

        // Sum all buffers
        for buffer in buffers.iter() {
            for (i, chunk) in buffer.chunks_exact(4).enumerate() {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result[i] += val;
            }
        }

        // Write result back to all buffers
        let result_bytes: Vec<u8> = result.iter().flat_map(|v| v.to_le_bytes()).collect();
        for buffer in buffers.iter_mut() {
            buffer.copy_from_slice(&result_bytes);
        }

        Ok(())
    }

    fn all_reduce_sum_bf16_cpu(&mut self, buffers: &mut [Vec<u8>]) -> Result<()> {
        use half::bf16;

        let len = buffers[0].len() / 2; // bf16 = 2 bytes
        let mut result = vec![0.0f32; len];

        // Sum all buffers (convert BF16 -> F32)
        for buffer in buffers.iter() {
            for (i, chunk) in buffer.chunks_exact(2).enumerate() {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                let val = bf16::from_bits(bits).to_f32();
                result[i] += val;
            }
        }

        // Convert back to BF16 and write to all buffers
        let result_bytes: Vec<u8> = result
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();

        for buffer in buffers.iter_mut() {
            buffer.copy_from_slice(&result_bytes);
        }

        Ok(())
    }
}

impl std::fmt::Debug for GpuCommunicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuCommunicator")
            .field("num_gpus", &self.num_gpus)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comm_op_types() {
        assert_eq!(CommOp::AllReduceSum, CommOp::AllReduceSum);
        assert_ne!(CommOp::AllReduceSum, CommOp::AllGather);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cpu_all_reduce_f32() {
        // Create a mock communicator
        let ctx = MultiGpuContext::new(2).unwrap();
        let mut comm = GpuCommunicator::new(ctx);

        // Create test buffers (2 GPUs, each with 4 f32 values)
        let mut buffers = vec![
            vec![1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
            vec![5.0f32, 6.0, 7.0, 8.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        ];

        comm.all_reduce_sum_bytes(&mut buffers, DType::F32).unwrap();

        // Check result: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]
        for buffer in &buffers {
            let result: Vec<f32> = buffer
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        }
    }
}
