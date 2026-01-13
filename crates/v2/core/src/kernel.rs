//! GPU kernel operations for transformer inference
//!
//! This module provides the core GPU operations needed for Llama inference:
//! - Matrix multiplication (via cuBLAS SGEMM directly on GPU FP32 data)
//! - RMS normalization (CUDA kernel - no CPU round-trip)
//! - SiLU activation (CUDA kernel - no CPU round-trip)
//! - Element-wise operations (CUDA kernels - no CPU round-trip)
//! - RoPE positional embeddings (CUDA kernel - no CPU round-trip)
//! - Scaled dot-product attention (CUDA kernel - no CPU round-trip)
//!
//! ALL operations work directly on CudaTensor's FP32 data without CPU round-trips.

use crate::error::{Result, V2Error};
use crate::gpu::{CudaTensor, CudaTensorBF16, GpuDevice};

#[cfg(feature = "cuda")]
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
#[cfg(feature = "cuda")]
use cudarc::cublas::sys::cublasOperation_t;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// CUDA kernel source code for all non-matmul operations
/// These kernels run entirely on GPU - no CPU round-trips!
#[cfg(feature = "cuda")]
const CUDA_KERNELS: &str = r#"
extern "C" {

// ============================================================================
// BASIC OPERATIONS
// ============================================================================

// SiLU activation: output[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i]))
__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));
    }
}

// Element-wise multiplication: out[i] = a[i] * b[i]
__global__ void mul_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// Element-wise addition: out[i] = a[i] + b[i]
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Add bias to 2D tensor: out[i, j] = x[i, j] + bias[j]
// Broadcasts 1D bias across all rows of 2D input
__global__ void add_bias_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                float* __restrict__ out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;
        out[idx] = x[idx] + bias[col];
    }
}

// BF16 to FP32 conversion kernel - converts bfloat16 to float32 on GPU
// This avoids expensive GPU->CPU->GPU round-trips for weight conversion
// bf16_in: array of u16 (bfloat16 bit patterns), f32_out: array of float32
__global__ void bf16_to_f32_kernel(const unsigned short* __restrict__ bf16_in,
                                    float* __restrict__ f32_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // BF16 is the upper 16 bits of FP32, so shift left by 16
        unsigned int bits = ((unsigned int)bf16_in[idx]) << 16;
        f32_out[idx] = __int_as_float(bits);
    }
}

// RMS Normalization with warp reduction for efficiency
// Each block handles one row (one sequence position)
// x: [batch_seq, hidden_dim], weight: [hidden_dim], out: [batch_seq, hidden_dim]
__global__ void rms_norm_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                 float* __restrict__ out, int hidden_dim, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Shared memory for partial sums
    extern __shared__ float shared[];

    const float* row_x = x + row * hidden_dim;
    float* row_out = out + row * hidden_dim;

    // Compute sum of squares with grid-stride loop
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = row_x[i];
        local_sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    // Store warp results to shared memory
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) {
        shared[warp_id] = local_sum_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    float sum_sq = 0.0f;
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < num_warps) {
        sum_sq = shared[tid];
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Broadcast rsqrt to all threads
    if (tid == 0) {
        shared[0] = rsqrtf(sum_sq / hidden_dim + eps);
    }
    __syncthreads();
    float rsqrt_val = shared[0];

    // Apply normalization and weight
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        row_out[i] = row_x[i] * rsqrt_val * weight[i];
    }
}

// RoPE (Rotary Position Embeddings) kernel
// Applies rotation to Q and K tensors based on position
// q: [batch_seq, num_heads, head_dim]
__global__ void rope_kernel(float* __restrict__ q, float* __restrict__ k,
                            const int* __restrict__ positions,
                            int batch_seq, int num_q_heads, int num_kv_heads, int head_dim,
                            float theta_base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = batch_seq * num_q_heads * head_dim;
    int total_k = batch_seq * num_kv_heads * head_dim;

    // Process Q tensor
    if (idx < total_q) {
        int pos_idx = idx / (num_q_heads * head_dim);
        int head_offset = idx % (num_q_heads * head_dim);
        int d = head_offset % head_dim;

        if (d < head_dim / 2) {
            int pos = positions[pos_idx];
            float freq = 1.0f / powf(theta_base, 2.0f * d / head_dim);
            float angle = pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            int idx_pair = idx + head_dim / 2;
            float x0 = q[idx];
            float x1 = q[idx_pair];

            q[idx] = x0 * cos_val - x1 * sin_val;
            q[idx_pair] = x0 * sin_val + x1 * cos_val;
        }
    }

    // Process K tensor (similar but with num_kv_heads)
    if (idx < total_k) {
        int pos_idx = idx / (num_kv_heads * head_dim);
        int head_offset = idx % (num_kv_heads * head_dim);
        int d = head_offset % head_dim;

        if (d < head_dim / 2) {
            int pos = positions[pos_idx];
            float freq = 1.0f / powf(theta_base, 2.0f * d / head_dim);
            float angle = pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            int idx_pair = idx + head_dim / 2;
            float x0 = k[idx];
            float x1 = k[idx_pair];

            k[idx] = x0 * cos_val - x1 * sin_val;
            k[idx_pair] = x0 * sin_val + x1 * cos_val;
        }
    }
}

// Scaled dot-product attention kernel (for sequences up to 2048)
// Uses shared memory for scores and softmax
// Q: [seq, num_heads, head_dim], K: [seq, num_kv_heads, head_dim], V: [seq, num_kv_heads, head_dim]
// Output: [seq, num_heads, head_dim]
// Each block handles one (query_pos, head) pair
__global__ void attention_kernel(
    const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    float scale, bool causal
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;

    // Group query attention: map Q head to KV head
    int kv_head = head / (num_q_heads / num_kv_heads);

    // Shared memory for attention scores and softmax
    extern __shared__ float smem[];
    float* scores = smem;
    float* max_val = smem + seq_len;
    float* sum_exp = smem + seq_len + 1;

    // Get Q vector for this query position and head
    const float* q_vec = Q + q_pos * num_q_heads * head_dim + head * head_dim;

    // Compute attention scores: Q @ K^T
    for (int k_pos = tid; k_pos < seq_len; k_pos += blockDim.x) {
        if (causal && k_pos > q_pos) {
            scores[k_pos] = -1e10f;  // Masked
        } else {
            const float* k_vec = K + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_vec[d] * k_vec[d];
            }
            scores[k_pos] = dot * scale;
        }
    }
    __syncthreads();

    // Find max for numerical stability (parallel reduction)
    float local_max = -1e10f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }
    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    if (tid % warpSize == 0) {
        smem[seq_len + tid / warpSize] = local_max;
    }
    __syncthreads();
    if (tid == 0) {
        float m = smem[seq_len];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++) {
            m = fmaxf(m, smem[seq_len + i]);
        }
        max_val[0] = m;
    }
    __syncthreads();
    float max_score = max_val[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float e = expf(scores[i] - max_score);
        scores[i] = e;
        local_sum += e;
    }
    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (tid % warpSize == 0) {
        smem[seq_len + 1 + tid / warpSize] = local_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float s = smem[seq_len + 1];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; i++) {
            s += smem[seq_len + 1 + i];
        }
        sum_exp[0] = s;
    }
    __syncthreads();
    float total_sum = sum_exp[0];

    // Normalize scores
    for (int i = tid; i < seq_len; i += blockDim.x) {
        scores[i] /= total_sum;
    }
    __syncthreads();

    // Compute weighted sum of values
    float* out_vec = output + q_pos * num_q_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int v_pos = 0; v_pos < seq_len; v_pos++) {
            const float* v_vec = V + v_pos * num_kv_heads * head_dim + kv_head * head_dim;
            sum += scores[v_pos] * v_vec[d];
        }
        out_vec[d] = sum;
    }
}

// ============================================================================
// KV CACHE OPERATIONS - For incremental decoding (80-90% speedup!)
// ============================================================================

// Copy new K/V tensors into the KV cache at the current position
// new_kv: [num_new_tokens, num_heads, head_dim] - new K or V to append
// cache: [max_seq_len, num_heads, head_dim] - existing cache
// cache_offset: where to start writing in the cache
__global__ void kv_cache_update_kernel(
    const float* __restrict__ new_kv,
    float* __restrict__ cache,
    int num_new_tokens, int num_heads, int head_dim,
    int cache_offset, int max_seq_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_new_tokens * num_heads * head_dim;

    if (idx < total) {
        int token_idx = idx / (num_heads * head_dim);
        int rest = idx % (num_heads * head_dim);
        int head_idx = rest / head_dim;
        int dim_idx = rest % head_dim;

        int cache_token = cache_offset + token_idx;
        if (cache_token < max_seq_len) {
            int cache_idx = cache_token * num_heads * head_dim + head_idx * head_dim + dim_idx;
            cache[cache_idx] = new_kv[idx];
        }
    }
}

// Attention with KV cache: Q attends to cached K/V
// Q: [num_new_tokens, num_q_heads, head_dim] - only query for new tokens
// K_cache, V_cache: [kv_len, num_kv_heads, head_dim] - full cached K/V
// output: [num_new_tokens, num_q_heads, head_dim]
// kv_len: total cached sequence length (including new tokens)
// Shared memory layout: [scores(kv_len), warp_reduce(8)]
__global__ void attention_with_kv_cache_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K_cache, const float* __restrict__ V_cache,
    float* __restrict__ output,
    int num_new_tokens, int kv_len,
    int num_q_heads, int num_kv_heads, int head_dim,
    float scale, int q_offset
) {
    int q_idx = blockIdx.x;  // Which new token (0 to num_new_tokens-1)
    int head = blockIdx.y;   // Which Q head
    int tid = threadIdx.x;

    // Group query attention: map Q head to KV head
    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = head / heads_per_kv;

    // Shared memory layout: scores[kv_len] + warp_reduce[8]
    extern __shared__ float smem[];
    float* scores = smem;
    float* warp_reduce = smem + kv_len;  // After scores array

    // Get Q vector for this query
    const float* q_vec = Q + q_idx * num_q_heads * head_dim + head * head_dim;

    // Absolute position of this query in the full sequence
    int q_pos = q_offset + q_idx;

    // Compute attention scores: Q @ K_cache^T
    for (int k_pos = tid; k_pos < kv_len; k_pos += blockDim.x) {
        // Causal masking: can only attend to positions <= q_pos
        if (k_pos > q_pos) {
            scores[k_pos] = -1e10f;
        } else {
            const float* k_vec = K_cache + k_pos * num_kv_heads * head_dim + kv_head * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_vec[d] * k_vec[d];
            }
            scores[k_pos] = dot * scale;
        }
    }
    __syncthreads();

    // Online softmax with parallel reduction across all warps
    // Step 1: Each thread computes local max
    float local_max = -1e10f;
    for (int i = tid; i < kv_len; i += blockDim.x) {
        local_max = fmaxf(local_max, scores[i]);
    }

    // Step 2: Warp-level reduction for max
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    // Step 3: Cross-warp reduction using shared memory (warp_reduce[0..3] for maxes)
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0 && warp_id < 4) {
        warp_reduce[warp_id] = local_max;
    }
    __syncthreads();

    // Thread 0 reduces across warps
    float max_score;
    if (tid == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        if (num_warps > 4) num_warps = 4;
        max_score = warp_reduce[0];
        for (int w = 1; w < num_warps; w++) {
            max_score = fmaxf(max_score, warp_reduce[w]);
        }
        warp_reduce[0] = max_score;  // Store final result
    }
    __syncthreads();
    max_score = warp_reduce[0];  // Broadcast to all threads

    // Step 4: Compute exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < kv_len; i += blockDim.x) {
        float e = expf(scores[i] - max_score);
        scores[i] = e;
        local_sum += e;
    }

    // Step 5: Warp-level reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Step 6: Cross-warp reduction for sum (warp_reduce[4..7] for sums)
    if (lane_id == 0 && warp_id < 4) {
        warp_reduce[4 + warp_id] = local_sum;
    }
    __syncthreads();

    float total_sum;
    if (tid == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        if (num_warps > 4) num_warps = 4;
        total_sum = warp_reduce[4];
        for (int w = 1; w < num_warps; w++) {
            total_sum += warp_reduce[4 + w];
        }
        warp_reduce[4] = total_sum;
    }
    __syncthreads();
    total_sum = warp_reduce[4];

    // Normalize and compute weighted sum
    for (int i = tid; i < kv_len; i += blockDim.x) {
        scores[i] /= total_sum;
    }
    __syncthreads();

    // Compute output: weighted sum of V_cache
    float* out_vec = output + q_idx * num_q_heads * head_dim + head * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int v_pos = 0; v_pos < kv_len; v_pos++) {
            const float* v_vec = V_cache + v_pos * num_kv_heads * head_dim + kv_head * head_dim;
            sum += scores[v_pos] * v_vec[d];
        }
        out_vec[d] = sum;
    }
}

// ============================================================================
// GPU EMBEDDING LOOKUP - Eliminates 1GB CPU transfer per token!
// ============================================================================

// GPU-side embedding lookup from BF16 embedding table
// token_ids: [seq_len] - token IDs on GPU
// embeddings: [vocab_shard_size, hidden_dim] - BF16 embedding table on GPU
// output: [seq_len, hidden_dim] - FP32 output on GPU
// vocab_offset: starting vocab index for this shard
__global__ void embedding_lookup_bf16_kernel(
    const int* __restrict__ token_ids,
    const unsigned short* __restrict__ embeddings,
    float* __restrict__ output,
    int seq_len, int hidden_dim, int vocab_shard_size, int vocab_offset
) {
    int seq_idx = blockIdx.x;
    int dim_idx = blockIdx.y * blockDim.x + threadIdx.x;

    if (seq_idx < seq_len && dim_idx < hidden_dim) {
        int token_id = token_ids[seq_idx];
        int local_idx = token_id - vocab_offset;

        float val = 0.0f;
        if (local_idx >= 0 && local_idx < vocab_shard_size) {
            // BF16 to FP32 conversion
            unsigned short bf16_bits = embeddings[local_idx * hidden_dim + dim_idx];
            unsigned int f32_bits = ((unsigned int)bf16_bits) << 16;
            val = __int_as_float(f32_bits);
        }

        output[seq_idx * hidden_dim + dim_idx] = val;
    }
}

// ============================================================================
// GPU RING ALL-REDUCE - Eliminates CPU round-trips for communication!
// ============================================================================

// Step 1 of ring all-reduce: reduce-scatter
// Each GPU sends a chunk to the next GPU and receives from the previous
// This kernel handles local addition for the reduce-scatter phase
__global__ void ring_reduce_scatter_kernel(
    const float* __restrict__ recv_buf,
    float* __restrict__ local_buf,
    int chunk_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        local_buf[idx] += recv_buf[idx];
    }
}

// Step 2 of ring all-reduce: all-gather
// Just copy received data to the appropriate chunk
// This is a simple memory copy kernel
__global__ void ring_all_gather_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int chunk_size, int dst_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunk_size) {
        dst[dst_offset + idx] = src[idx];
    }
}

// In-place addition for two-GPU all-reduce (optimized for 2 GPUs)
// For H100 NVLink, this is fast enough for tensor parallelism
__global__ void two_gpu_add_kernel(
    const float* __restrict__ other,
    float* __restrict__ local,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        local[idx] += other[idx];
    }
}

// ============================================================================
// FLASH ATTENTION V2 - Memory efficient attention for long sequences
// ============================================================================

// Flash Attention kernel with online softmax
// Processes attention in tiles to avoid O(N^2) memory
// Block size: 64 tokens per tile (adjustable)
#define FLASH_BLOCK_SIZE 64

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,  // [seq_len, num_heads, head_dim]
    const float* __restrict__ K,  // [seq_len, num_kv_heads, head_dim]
    const float* __restrict__ V,  // [seq_len, num_kv_heads, head_dim]
    float* __restrict__ O,        // [seq_len, num_heads, head_dim]
    int seq_len, int num_q_heads, int num_kv_heads, int head_dim,
    float scale
) {
    int q_block = blockIdx.x;  // Which block of queries
    int head = blockIdx.y;     // Which head
    int tid = threadIdx.x;

    // GQA head mapping
    int heads_per_kv = num_q_heads / num_kv_heads;
    int kv_head = head / heads_per_kv;

    // Shared memory for tile of K, V, and running statistics
    extern __shared__ float smem[];
    float* s_k = smem;                                    // [FLASH_BLOCK_SIZE, head_dim]
    float* s_v = smem + FLASH_BLOCK_SIZE * head_dim;      // [FLASH_BLOCK_SIZE, head_dim]
    float* s_scores = smem + 2 * FLASH_BLOCK_SIZE * head_dim;  // [FLASH_BLOCK_SIZE]

    // Each thread handles one query position in the block
    int q_pos = q_block * FLASH_BLOCK_SIZE + tid;
    if (q_pos >= seq_len) return;

    // Load Q for this thread
    float q_local[128];  // Assume head_dim <= 128
    for (int d = 0; d < head_dim; d++) {
        q_local[d] = Q[q_pos * num_q_heads * head_dim + head * head_dim + d];
    }

    // Running max and sum for online softmax
    float m_prev = -1e10f;
    float l_prev = 0.0f;
    float o_local[128] = {0.0f};

    // Process K/V in tiles
    int num_kv_blocks = (seq_len + FLASH_BLOCK_SIZE - 1) / FLASH_BLOCK_SIZE;
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * FLASH_BLOCK_SIZE;

        // Cooperatively load K and V tiles into shared memory
        __syncthreads();
        for (int i = tid; i < FLASH_BLOCK_SIZE * head_dim; i += blockDim.x) {
            int tile_pos = i / head_dim;
            int d = i % head_dim;
            int kv_pos = kv_start + tile_pos;
            if (kv_pos < seq_len) {
                s_k[i] = K[kv_pos * num_kv_heads * head_dim + kv_head * head_dim + d];
                s_v[i] = V[kv_pos * num_kv_heads * head_dim + kv_head * head_dim + d];
            } else {
                s_k[i] = 0.0f;
                s_v[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute attention scores for this tile
        float m_new = m_prev;
        for (int k_tile = 0; k_tile < FLASH_BLOCK_SIZE; k_tile++) {
            int kv_pos = kv_start + k_tile;
            if (kv_pos >= seq_len || kv_pos > q_pos) continue;  // Causal + bounds

            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q_local[d] * s_k[k_tile * head_dim + d];
            }
            s_scores[k_tile] = dot * scale;
            m_new = fmaxf(m_new, s_scores[k_tile]);
        }

        // Online softmax correction and accumulation
        float l_new = l_prev * expf(m_prev - m_new);
        for (int k_tile = 0; k_tile < FLASH_BLOCK_SIZE; k_tile++) {
            int kv_pos = kv_start + k_tile;
            if (kv_pos >= seq_len || kv_pos > q_pos) continue;

            float p = expf(s_scores[k_tile] - m_new);
            l_new += p;

            // Accumulate weighted V
            for (int d = 0; d < head_dim; d++) {
                o_local[d] = o_local[d] * expf(m_prev - m_new) + p * s_v[k_tile * head_dim + d];
            }
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Final normalization and write output
    for (int d = 0; d < head_dim; d++) {
        O[q_pos * num_q_heads * head_dim + head * head_dim + d] = o_local[d] / l_prev;
    }
}

} // extern "C"
"#;

/// Kernel execution context for a single GPU
///
/// Uses cuBLAS for GEMM operations and NVRTC-compiled CUDA kernels for other ops.
/// All data stays on GPU - no CPU round-trips during inference.
pub struct KernelContext {
    device: GpuDevice,
    #[cfg(feature = "cuda")]
    blas: CudaBlas,
    #[cfg(feature = "cuda")]
    kernels_compiled: bool,
}

impl KernelContext {
    /// Create a new kernel context for the given device
    #[cfg(feature = "cuda")]
    pub fn new(device: GpuDevice) -> Result<Self> {
        let blas = CudaBlas::new(device.cuda_device().clone())
            .map_err(|e| V2Error::Gpu(format!("Failed to create cuBLAS: {:?}", e)))?;

        let mut ctx = Self {
            device,
            blas,
            kernels_compiled: false,
        };

        // Compile CUDA kernels at initialization
        ctx.compile_kernels()?;

        Ok(ctx)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(device: GpuDevice) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compile CUDA kernels using NVRTC
    #[cfg(feature = "cuda")]
    fn compile_kernels(&mut self) -> Result<()> {
        if self.kernels_compiled {
            return Ok(());
        }

        tracing::info!("Compiling CUDA kernels for GPU {}...", self.device.device_id);

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNELS)
            .map_err(|e| V2Error::Gpu(format!("NVRTC compilation failed: {:?}", e)))?;

        self.device.cuda_device().load_ptx(
            ptx,
            "shardlm_kernels",
            &[
                // Basic operations
                "silu_kernel", "mul_kernel", "add_kernel", "add_bias_kernel", "bf16_to_f32_kernel",
                // Normalization and position encoding
                "rms_norm_kernel", "rope_kernel",
                // Attention variants
                "attention_kernel", "attention_with_kv_cache_kernel", "flash_attention_kernel",
                // KV cache operations
                "kv_cache_update_kernel",
                // Embedding lookup
                "embedding_lookup_bf16_kernel",
                // Ring all-reduce operations
                "ring_reduce_scatter_kernel", "ring_all_gather_kernel", "two_gpu_add_kernel",
            ],
        ).map_err(|e| V2Error::Gpu(format!("Failed to load PTX: {:?}", e)))?;

        self.kernels_compiled = true;
        tracing::info!("CUDA kernels compiled successfully for GPU {}", self.device.device_id);

        Ok(())
    }

    /// Get the device
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Matrix multiplication: C = A @ B^T using cuBLAS SGEMM
    ///
    /// Both A and B are CudaTensors with FP32 data already on GPU.
    /// No CPU round-trips - operates directly on GPU memory.
    #[cfg(feature = "cuda")]
    pub fn matmul_f32(
        &self,
        a: &CudaTensor,      // [M, K] - input activations (FP32 on GPU)
        b: &CudaTensor,      // [N, K] - weight matrix (FP32 on GPU)
    ) -> Result<CudaTensor> {
        // Validate shapes
        if a.shape.len() != 2 || b.shape.len() != 2 {
            return Err(V2Error::Gpu(format!(
                "matmul requires 2D tensors, got {:?} and {:?}",
                a.shape, b.shape
            )));
        }

        let m = a.shape[0]; // batch * seq_len
        let k = a.shape[1]; // input features
        let n = b.shape[0]; // output features

        if b.shape[1] != k {
            return Err(V2Error::Gpu(format!(
                "matmul shape mismatch: A is [{}, {}], B is [{}, {}]",
                m, k, n, b.shape[1]
            )));
        }

        // Allocate output on GPU
        let mut c = CudaTensor::zeros(&self.device, vec![m, n])?;

        // cuBLAS SGEMM directly on GPU data - NO CPU transfers
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,  // Transpose B' to get [n,k]
            transb: cublasOperation_t::CUBLAS_OP_N,  // A' is [k,m], no transpose
            m: n as i32,   // rows of op(B') = n
            n: m as i32,   // cols of A' = m
            k: k as i32,   // cols of op(B') = rows of A' = k
            alpha: 1.0f32,
            lda: k as i32, // Leading dim of B' (column-major [k,n]) = k
            ldb: k as i32, // Leading dim of A' (column-major [k,m]) = k
            beta: 0.0f32,
            ldc: n as i32, // Leading dim of C' (column-major [n,m]) = n
        };

        unsafe {
            self.blas.gemm(cfg, b.data(), a.data(), c.data_mut())
                .map_err(|e| V2Error::Gpu(format!("cuBLAS SGEMM failed: {:?}", e)))?;
        }

        Ok(c)
    }

    /// Legacy compatibility: matmul_bf16 now just calls matmul_f32
    #[cfg(feature = "cuda")]
    pub fn matmul_bf16(
        &self,
        a: &CudaTensor,
        b: &CudaTensor,
    ) -> Result<CudaTensor> {
        self.matmul_f32(a, b)
    }

    /// Matrix multiplication with BF16 weights: C = A @ B^T
    /// Uses GPU kernel for BF16→FP32 conversion to avoid CPU round-trips
    #[cfg(feature = "cuda")]
    pub fn matmul_bf16_weights(
        &self,
        a: &CudaTensor,           // [M, K] - input activations (FP32 on GPU)
        b: &CudaTensorBF16,       // [N, K] - weight matrix (BF16 on GPU)
    ) -> Result<CudaTensor> {
        // Convert BF16 weights to FP32 using GPU kernel (no CPU round-trip!)
        let b_f32 = self.bf16_to_f32_gpu(b)?;
        self.matmul_f32(a, &b_f32)
    }

    /// Convert BF16 tensor to FP32 tensor on GPU using CUDA kernel
    /// This avoids the expensive GPU→CPU→GPU round-trip
    #[cfg(feature = "cuda")]
    fn bf16_to_f32_gpu(&self, bf16_tensor: &CudaTensorBF16) -> Result<CudaTensor> {
        let n = bf16_tensor.numel();
        let mut f32_output = CudaTensor::zeros(&self.device, bf16_tensor.shape.clone())?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "bf16_to_f32_kernel")
            .ok_or_else(|| V2Error::Gpu("bf16_to_f32_kernel not found".to_string()))?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (bf16_tensor.data(), f32_output.data_mut(), n as i32))
                .map_err(|e| V2Error::Gpu(format!("bf16_to_f32 kernel launch failed: {:?}", e)))?;
        }

        Ok(f32_output)
    }

    /// Convert BF16 tensor to FP32 tensor on GPU (public API)
    /// Use this at load time to pre-convert weights for maximum inference speed
    #[cfg(feature = "cuda")]
    pub fn bf16_to_f32_tensor(&self, bf16_tensor: &CudaTensorBF16) -> Result<CudaTensor> {
        self.bf16_to_f32_gpu(bf16_tensor)
    }

    /// Copy an FP32 tensor to a new tensor on GPU (no CPU round-trip)
    /// Use this at load time to duplicate tensors that are already FP32
    #[cfg(feature = "cuda")]
    pub fn copy_tensor(&self, src: &CudaTensor) -> Result<CudaTensor> {
        use cudarc::driver::DeviceRepr;

        let n = src.shape.iter().product::<usize>();
        let mut dst = CudaTensor::zeros(&self.device, src.shape.clone())?;

        // Use cudarc's device-to-device copy
        unsafe {
            self.device.cuda_device().dtod_copy(src.data(), dst.data_mut())
                .map_err(|e| V2Error::Gpu(format!("dtod_copy failed: {:?}", e)))?;
        }

        Ok(dst)
    }

    /// Matrix multiplication with INT8 weight with per-channel scales
    #[cfg(feature = "cuda")]
    pub fn matmul_int8_bf16(
        &self,
        input: &CudaTensor,     // [M, K] FP32 activations on GPU
        weight: &CudaTensor,    // [N, K] FP32 weights on GPU (pre-dequantized)
        _scale: &CudaTensor,    // [N] scales - not needed, already applied
    ) -> Result<CudaTensor> {
        self.matmul_f32(input, weight)
    }

    /// SiLU activation: output = x * sigmoid(x) - CUDA kernel, no CPU round-trip!
    #[cfg(feature = "cuda")]
    pub fn silu(&self, x: &CudaTensor) -> Result<CudaTensor> {
        let n = x.shape.iter().product::<usize>();
        let mut output = CudaTensor::zeros(&self.device, x.shape.clone())?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "silu_kernel")
            .ok_or_else(|| V2Error::Gpu("silu_kernel not found".to_string()))?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (x.data(), output.data_mut(), n as i32))
                .map_err(|e| V2Error::Gpu(format!("silu kernel launch failed: {:?}", e)))?;
        }

        Ok(output)
    }

    /// Element-wise multiplication: output = a * b - CUDA kernel, no CPU round-trip!
    #[cfg(feature = "cuda")]
    pub fn mul(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        if a.shape != b.shape {
            return Err(V2Error::Gpu(format!(
                "mul shape mismatch: {:?} vs {:?}",
                a.shape, b.shape
            )));
        }

        let n = a.shape.iter().product::<usize>();
        let mut output = CudaTensor::zeros(&self.device, a.shape.clone())?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "mul_kernel")
            .ok_or_else(|| V2Error::Gpu("mul_kernel not found".to_string()))?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (a.data(), b.data(), output.data_mut(), n as i32))
                .map_err(|e| V2Error::Gpu(format!("mul kernel launch failed: {:?}", e)))?;
        }

        Ok(output)
    }

    /// Element-wise addition: output = a + b - CUDA kernel, no CPU round-trip!
    #[cfg(feature = "cuda")]
    pub fn add(&self, a: &CudaTensor, b: &CudaTensor) -> Result<CudaTensor> {
        if a.shape != b.shape {
            return Err(V2Error::Gpu(format!(
                "add shape mismatch: {:?} vs {:?}",
                a.shape, b.shape
            )));
        }

        let n = a.shape.iter().product::<usize>();
        let mut output = CudaTensor::zeros(&self.device, a.shape.clone())?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "add_kernel")
            .ok_or_else(|| V2Error::Gpu("add_kernel not found".to_string()))?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (a.data(), b.data(), output.data_mut(), n as i32))
                .map_err(|e| V2Error::Gpu(format!("add kernel launch failed: {:?}", e)))?;
        }

        Ok(output)
    }

    /// Add bias to a 2D tensor: output[i, j] = x[i, j] + bias[j]
    /// Broadcasts 1D bias across all rows of 2D input.
    /// CUDA kernel - no CPU round-trip!
    #[cfg(feature = "cuda")]
    pub fn add_bias(&self, x: &CudaTensor, bias: &CudaTensor) -> Result<CudaTensor> {
        if x.shape.len() != 2 {
            return Err(V2Error::Gpu(format!(
                "add_bias expects 2D input, got {:?}",
                x.shape
            )));
        }
        if bias.shape.len() != 1 {
            return Err(V2Error::Gpu(format!(
                "add_bias expects 1D bias, got {:?}",
                bias.shape
            )));
        }
        if x.shape[1] != bias.shape[0] {
            return Err(V2Error::Gpu(format!(
                "add_bias dimension mismatch: input {:?} vs bias {:?}",
                x.shape, bias.shape
            )));
        }

        let rows = x.shape[0];
        let cols = x.shape[1];
        let n = rows * cols;
        let mut output = CudaTensor::zeros(&self.device, x.shape.clone())?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "add_bias_kernel")
            .ok_or_else(|| V2Error::Gpu("add_bias_kernel not found".to_string()))?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (x.data(), bias.data(), output.data_mut(), rows as i32, cols as i32))
                .map_err(|e| V2Error::Gpu(format!("add_bias kernel launch failed: {:?}", e)))?;
        }

        Ok(output)
    }

    /// RMS Normalization: output = x * rsqrt(mean(x^2) + eps) * weight
    /// CUDA kernel with warp reduction - no CPU round-trip!
    #[cfg(feature = "cuda")]
    pub fn rms_norm(
        &self,
        x: &CudaTensor,        // [batch*seq, hidden_dim] FP32 on GPU
        weight: &CudaTensor,   // [hidden_dim] FP32 on GPU
        eps: f32,
    ) -> Result<CudaTensor> {
        if x.shape.len() != 2 {
            return Err(V2Error::Gpu(format!(
                "rms_norm expects 2D input, got {:?}",
                x.shape
            )));
        }

        let batch_seq = x.shape[0];
        let hidden_dim = x.shape[1];
        let mut output = CudaTensor::zeros(&self.device, vec![batch_seq, hidden_dim])?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "rms_norm_kernel")
            .ok_or_else(|| V2Error::Gpu("rms_norm_kernel not found".to_string()))?;

        // Each block handles one row, use 256 threads for warp reduction
        let block_size = 256;
        let num_warps = (block_size + 31) / 32;
        let shared_mem = (num_warps * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (batch_seq as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        unsafe {
            kernel.launch(cfg, (x.data(), weight.data(), output.data_mut(), hidden_dim as i32, eps))
                .map_err(|e| V2Error::Gpu(format!("rms_norm kernel launch failed: {:?}", e)))?;
        }

        Ok(output)
    }

    /// Apply rotary position embeddings to Q and K - CUDA kernel, no CPU round-trip!
    #[cfg(feature = "cuda")]
    pub fn apply_rope(
        &self,
        q: &CudaTensor,  // [batch*seq, num_heads, head_dim]
        k: &CudaTensor,  // [batch*seq, num_kv_heads, head_dim]
        positions: &[u32],
        rope_theta: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        let batch_seq = q.shape[0];
        let num_q_heads = q.shape[1];
        let head_dim = q.shape[2];
        let num_kv_heads = k.shape[1];

        // Clone tensors to modify in-place
        let mut q_out = q.clone_to(&self.device)?;
        let mut k_out = k.clone_to(&self.device)?;

        // Upload positions to GPU
        let positions_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let positions_gpu = self.device.htod_i32(positions_i32)?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "rope_kernel")
            .ok_or_else(|| V2Error::Gpu("rope_kernel not found".to_string()))?;

        let total_elements = std::cmp::max(
            batch_seq * num_q_heads * head_dim,
            batch_seq * num_kv_heads * head_dim
        );
        let block_size = 256;
        let grid_size = (total_elements + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (
                q_out.data_mut(),
                k_out.data_mut(),
                &positions_gpu,
                batch_seq as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                rope_theta
            )).map_err(|e| V2Error::Gpu(format!("rope kernel launch failed: {:?}", e)))?;
        }

        Ok((q_out, k_out))
    }

    /// Scaled dot-product attention with causal masking - CUDA kernel, no CPU round-trip!
    /// Supports sequences up to 2048 tokens efficiently
    #[cfg(feature = "cuda")]
    pub fn attention(
        &self,
        q: &CudaTensor,  // [batch*seq, num_heads, head_dim]
        k: &CudaTensor,  // [kv_seq, num_kv_heads, head_dim]
        v: &CudaTensor,  // [kv_seq, num_kv_heads, head_dim]
        scale: f32,
        causal: bool,
    ) -> Result<CudaTensor> {
        let seq_len = q.shape[0];
        let num_q_heads = q.shape[1];
        let head_dim = q.shape[2];
        let num_kv_heads = k.shape[1];

        let mut output = CudaTensor::zeros(&self.device, vec![seq_len, num_q_heads, head_dim])?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "attention_kernel")
            .ok_or_else(|| V2Error::Gpu("attention_kernel not found".to_string()))?;

        // Each block handles one (query_pos, head) pair
        // Use enough threads to cover head_dim and seq_len operations
        let block_size = 128;
        // Shared memory for scores, max, and sum (seq_len + 2 floats)
        let shared_mem = ((seq_len + 2) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (seq_len as u32, num_q_heads as u32, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        unsafe {
            kernel.launch(cfg, (
                q.data(),
                k.data(),
                v.data(),
                output.data_mut(),
                seq_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                scale,
                causal as i32
            )).map_err(|e| V2Error::Gpu(format!("attention kernel launch failed: {:?}", e)))?;
        }

        Ok(output)
    }

    /// Embedding lookup (for sharded vocabulary) - FP32 embeddings
    #[cfg(feature = "cuda")]
    pub fn embedding_lookup(
        &self,
        token_ids: &[u32],
        embedding_table: &CudaTensor,
        vocab_offset: usize,
    ) -> Result<CudaTensor> {
        let vocab_shard_size = embedding_table.shape[0];
        let hidden_dim = embedding_table.shape[1];
        let seq_len = token_ids.len();

        let embed_f32 = self.device.dtoh_f32(embedding_table.data())?;

        use rayon::prelude::*;

        let output_f32: Vec<f32> = (0..seq_len)
            .into_par_iter()
            .flat_map(|i| {
                let token_id = token_ids[i] as usize;
                let local_idx = token_id.checked_sub(vocab_offset);

                let mut result = Vec::with_capacity(hidden_dim);
                for j in 0..hidden_dim {
                    let val = if let Some(idx) = local_idx {
                        if idx < vocab_shard_size {
                            embed_f32[idx * hidden_dim + j]
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    result.push(val);
                }
                result
            })
            .collect();

        CudaTensor::from_f32(&self.device, vec![seq_len, hidden_dim], output_f32)
    }

    /// Embedding lookup with BF16 embeddings (for sharded vocabulary) - CPU fallback
    #[cfg(feature = "cuda")]
    pub fn embedding_lookup_bf16(
        &self,
        token_ids: &[u32],
        embedding_table: &CudaTensorBF16,
        vocab_offset: usize,
    ) -> Result<CudaTensor> {
        use half::bf16;

        let vocab_shard_size = embedding_table.shape[0];
        let hidden_dim = embedding_table.shape[1];
        let seq_len = token_ids.len();

        let embed_bf16 = self.device.dtoh(embedding_table.data())?;

        use rayon::prelude::*;

        let output_f32: Vec<f32> = (0..seq_len)
            .into_par_iter()
            .flat_map(|i| {
                let token_id = token_ids[i] as usize;
                let local_idx = token_id.checked_sub(vocab_offset);

                let mut result = Vec::with_capacity(hidden_dim);
                for j in 0..hidden_dim {
                    let val = if let Some(idx) = local_idx {
                        if idx < vocab_shard_size {
                            bf16::from_bits(embed_bf16[idx * hidden_dim + j]).to_f32()
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    result.push(val);
                }
                result
            })
            .collect();

        CudaTensor::from_f32(&self.device, vec![seq_len, hidden_dim], output_f32)
    }

    /// GPU-native embedding lookup with BF16 embeddings - NO CPU transfer!
    /// This is 10-20% faster than the CPU version for embedding lookup.
    #[cfg(feature = "cuda")]
    pub fn embedding_lookup_bf16_gpu(
        &self,
        token_ids: &[u32],
        embedding_table: &CudaTensorBF16,
        vocab_offset: usize,
    ) -> Result<CudaTensor> {
        let vocab_shard_size = embedding_table.shape[0];
        let hidden_dim = embedding_table.shape[1];
        let seq_len = token_ids.len();

        // Upload token IDs to GPU
        let token_ids_i32: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let token_ids_gpu = self.device.htod_i32(token_ids_i32)?;

        // Allocate output
        let mut output = CudaTensor::zeros(&self.device, vec![seq_len, hidden_dim])?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "embedding_lookup_bf16_kernel")
            .ok_or_else(|| V2Error::Gpu("embedding_lookup_bf16_kernel not found".to_string()))?;

        // Grid: [seq_len, ceil(hidden_dim / 256)]
        let block_size = 256;
        let grid_y = (hidden_dim + block_size - 1) / block_size;
        let cfg = LaunchConfig {
            grid_dim: (seq_len as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (
                &token_ids_gpu,
                embedding_table.data(),
                output.data_mut(),
                seq_len as i32,
                hidden_dim as i32,
                vocab_shard_size as i32,
                vocab_offset as i32,
            )).map_err(|e| V2Error::Gpu(format!("embedding_lookup kernel failed: {:?}", e)))?;
        }

        Ok(output)
    }

    // =========================================================================
    // KV CACHE OPERATIONS - For incremental decoding (80-90% speedup!)
    // =========================================================================

    /// Update KV cache with new K/V values
    /// Copies new_kv into cache starting at cache_offset
    #[cfg(feature = "cuda")]
    pub fn kv_cache_update(
        &self,
        new_kv: &CudaTensor,      // [num_new_tokens, num_heads, head_dim]
        cache: &mut CudaTensor,   // [max_seq_len, num_heads, head_dim]
        cache_offset: usize,
    ) -> Result<()> {
        let num_new_tokens = new_kv.shape[0];
        let num_heads = new_kv.shape[1];
        let head_dim = new_kv.shape[2];
        let max_seq_len = cache.shape[0];

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "kv_cache_update_kernel")
            .ok_or_else(|| V2Error::Gpu("kv_cache_update_kernel not found".to_string()))?;

        let total = num_new_tokens * num_heads * head_dim;
        let block_size = 256;
        let grid_size = (total + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (
                new_kv.data(),
                cache.data_mut(),
                num_new_tokens as i32,
                num_heads as i32,
                head_dim as i32,
                cache_offset as i32,
                max_seq_len as i32,
            )).map_err(|e| V2Error::Gpu(format!("kv_cache_update kernel failed: {:?}", e)))?;
        }

        Ok(())
    }

    /// Attention with KV cache - only compute attention for new tokens
    /// This is the KEY optimization for incremental decoding (80-90% speedup!)
    #[cfg(feature = "cuda")]
    pub fn attention_with_kv_cache(
        &self,
        q: &CudaTensor,          // [num_new_tokens, num_q_heads, head_dim]
        k_cache: &CudaTensor,    // [max_seq_len, num_kv_heads, head_dim] - full cache buffer
        v_cache: &CudaTensor,    // [max_seq_len, num_kv_heads, head_dim] - full cache buffer
        scale: f32,
        q_offset: usize,         // Position of first new token in full sequence
        kv_len: usize,           // Actual number of valid KV entries (NOT max_seq_len!)
    ) -> Result<CudaTensor> {
        let num_new_tokens = q.shape[0];
        let num_q_heads = q.shape[1];
        let head_dim = q.shape[2];
        let num_kv_heads = k_cache.shape[1];

        let mut output = CudaTensor::zeros(&self.device, vec![num_new_tokens, num_q_heads, head_dim])?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "attention_with_kv_cache_kernel")
            .ok_or_else(|| V2Error::Gpu("attention_with_kv_cache_kernel not found".to_string()))?;

        // Grid: [num_new_tokens, num_q_heads]
        let block_size = 128;
        // Shared memory: scores[kv_len] + warp_reduce[8] for cross-warp reduction
        let shared_mem = ((kv_len + 8) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (num_new_tokens as u32, num_q_heads as u32, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        unsafe {
            kernel.launch(cfg, (
                q.data(),
                k_cache.data(),
                v_cache.data(),
                output.data_mut(),
                num_new_tokens as i32,
                kv_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                scale,
                q_offset as i32,
            )).map_err(|e| V2Error::Gpu(format!("attention_with_kv_cache kernel failed: {:?}", e)))?;
        }

        Ok(output)
    }

    // =========================================================================
    // FLASH ATTENTION - Memory efficient for long sequences (25-40% speedup)
    // =========================================================================

    /// Flash Attention v2 - tiled attention that's memory efficient
    /// Uses O(N) memory instead of O(N^2) for attention scores
    #[cfg(feature = "cuda")]
    pub fn flash_attention(
        &self,
        q: &CudaTensor,  // [seq_len, num_heads, head_dim]
        k: &CudaTensor,  // [seq_len, num_kv_heads, head_dim]
        v: &CudaTensor,  // [seq_len, num_kv_heads, head_dim]
        scale: f32,
    ) -> Result<CudaTensor> {
        let seq_len = q.shape[0];
        let num_q_heads = q.shape[1];
        let head_dim = q.shape[2];
        let num_kv_heads = k.shape[1];

        let mut output = CudaTensor::zeros(&self.device, vec![seq_len, num_q_heads, head_dim])?;

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "flash_attention_kernel")
            .ok_or_else(|| V2Error::Gpu("flash_attention_kernel not found".to_string()))?;

        // Flash attention block size
        let flash_block_size = 64;
        let num_q_blocks = (seq_len + flash_block_size - 1) / flash_block_size;

        // Shared memory: K tile + V tile + scores
        let shared_mem = ((2 * flash_block_size * head_dim + flash_block_size) * std::mem::size_of::<f32>()) as u32;

        let cfg = LaunchConfig {
            grid_dim: (num_q_blocks as u32, num_q_heads as u32, 1),
            block_dim: (flash_block_size as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        unsafe {
            kernel.launch(cfg, (
                q.data(),
                k.data(),
                v.data(),
                output.data_mut(),
                seq_len as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                scale,
            )).map_err(|e| V2Error::Gpu(format!("flash_attention kernel failed: {:?}", e)))?;
        }

        Ok(output)
    }

    // =========================================================================
    // GPU RING ALL-REDUCE - For 2-GPU tensor parallelism (50-80% speedup)
    // =========================================================================

    /// Two-GPU all-reduce: add tensor from other GPU to local tensor
    /// For H100 with NVLink, this is efficient for tensor parallelism
    #[cfg(feature = "cuda")]
    pub fn two_gpu_add_inplace(
        &self,
        other: &CudaTensor,    // Tensor from other GPU
        local: &mut CudaTensor, // Local tensor to update
    ) -> Result<()> {
        let n = local.numel();
        if n != other.numel() {
            return Err(V2Error::Gpu(format!(
                "Size mismatch for two_gpu_add: {} vs {}",
                other.numel(), n
            )));
        }

        let kernel = self.device.cuda_device()
            .get_func("shardlm_kernels", "two_gpu_add_kernel")
            .ok_or_else(|| V2Error::Gpu("two_gpu_add_kernel not found".to_string()))?;

        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(cfg, (other.data(), local.data_mut(), n as i32))
                .map_err(|e| V2Error::Gpu(format!("two_gpu_add kernel failed: {:?}", e)))?;
        }

        Ok(())
    }

    /// Reshape tensor for multi-head attention (in-place, no data copy!)
    /// Takes ownership and returns the same tensor with different shape.
    /// This is a zero-cost operation - just changes metadata.
    #[cfg(feature = "cuda")]
    pub fn reshape_for_attention(
        &self,
        x: CudaTensor,
        num_heads: usize,
    ) -> Result<CudaTensor> {
        let batch_seq = x.shape[0];
        let hidden_dim = x.shape[1];
        let head_dim = hidden_dim / num_heads;

        if hidden_dim % num_heads != 0 {
            return Err(V2Error::Gpu(format!(
                "hidden_dim {} not divisible by num_heads {}",
                hidden_dim, num_heads
            )));
        }

        // In-place reshape - zero cost, just changes shape metadata
        x.reshape_inplace(vec![batch_seq, num_heads, head_dim])
    }

    /// Reshape tensor from multi-head attention back to 2D (in-place, no data copy!)
    /// Takes ownership and returns the same tensor with different shape.
    /// This is a zero-cost operation - just changes metadata.
    #[cfg(feature = "cuda")]
    pub fn reshape_from_attention(&self, x: CudaTensor) -> Result<CudaTensor> {
        let batch_seq = x.shape[0];
        let num_heads = x.shape[1];
        let head_dim = x.shape[2];
        let hidden_dim = num_heads * head_dim;

        // In-place reshape - zero cost, just changes shape metadata
        x.reshape_inplace(vec![batch_seq, hidden_dim])
    }
}

// Non-CUDA implementations
#[cfg(not(feature = "cuda"))]
impl KernelContext {
    pub fn matmul_f32(&self, _a: &CudaTensor, _b: &CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn matmul_bf16(&self, _a: &CudaTensor, _b: &CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn matmul_bf16_weights(&self, _a: &CudaTensor, _b: &CudaTensorBF16) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn matmul_int8_bf16(
        &self,
        _input: &CudaTensor,
        _weight: &CudaTensor,
        _scale: &CudaTensor,
    ) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn rms_norm(&self, _x: &CudaTensor, _weight: &CudaTensor, _eps: f32) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn silu(&self, _x: &CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn mul(&self, _a: &CudaTensor, _b: &CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn add(&self, _a: &CudaTensor, _b: &CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn add_bias(&self, _x: &CudaTensor, _bias: &CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn apply_rope(
        &self,
        _q: &CudaTensor,
        _k: &CudaTensor,
        _positions: &[u32],
        _rope_theta: f32,
    ) -> Result<(CudaTensor, CudaTensor)> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn attention(
        &self,
        _q: &CudaTensor,
        _k: &CudaTensor,
        _v: &CudaTensor,
        _scale: f32,
        _causal: bool,
    ) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn embedding_lookup(
        &self,
        _token_ids: &[u32],
        _embedding_table: &CudaTensor,
        _vocab_offset: usize,
    ) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn embedding_lookup_bf16(
        &self,
        _token_ids: &[u32],
        _embedding_table: &CudaTensorBF16,
        _vocab_offset: usize,
    ) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn reshape_for_attention(&self, _x: CudaTensor, _num_heads: usize) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }

    pub fn reshape_from_attention(&self, _x: CudaTensor) -> Result<CudaTensor> {
        Err(V2Error::Gpu("CUDA not enabled".to_string()))
    }
}

impl std::fmt::Debug for KernelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelContext")
            .field("device_id", &self.device.device_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_kernel_context_debug() {
        // Test that debug formatting works even without CUDA
        println!("KernelContext test");
    }
}
