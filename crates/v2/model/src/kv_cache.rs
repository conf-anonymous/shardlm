//! KV Cache management for long context inference
//!
//! This module provides GPU-resident KV caching for efficient autoregressive generation.
//! Key features:
//! - GPU-resident cache (no CPU transfers during generation)
//! - Per-GPU sharding for tensor parallelism
//! - Incremental updates (only cache new tokens)

#[cfg(feature = "cuda")]
use shardlm_v2_core::gpu::{CudaTensor, GpuDevice};

use crate::error::{ModelError, Result};

/// GPU-resident KV Cache for a single layer on a single GPU
/// Stores K and V tensors directly on GPU for fast attention computation
#[cfg(feature = "cuda")]
pub struct GpuLayerKvCache {
    /// Key cache on GPU: [max_seq_len, num_kv_heads_per_gpu, head_dim]
    pub key: CudaTensor,
    /// Value cache on GPU: [max_seq_len, num_kv_heads_per_gpu, head_dim]
    pub value: CudaTensor,
    /// Current sequence length (number of tokens cached)
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV heads per GPU
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// GPU device ID
    pub device_id: usize,
}

#[cfg(feature = "cuda")]
impl GpuLayerKvCache {
    /// Create a new GPU-resident KV cache
    pub fn new(
        device: &GpuDevice,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        // Allocate cache tensors on GPU
        let key = CudaTensor::zeros(device, vec![max_seq_len, num_kv_heads, head_dim])
            .map_err(|e| ModelError::CudaError(e.to_string()))?;
        let value = CudaTensor::zeros(device, vec![max_seq_len, num_kv_heads, head_dim])
            .map_err(|e| ModelError::CudaError(e.to_string()))?;

        Ok(Self {
            key,
            value,
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
            device_id: device.device_id,
        })
    }

    /// Get current sequence length
    pub fn current_seq_len(&self) -> usize {
        self.seq_len
    }

    /// Update sequence length after appending new tokens
    pub fn set_seq_len(&mut self, new_len: usize) {
        self.seq_len = new_len;
    }

    /// Clear the cache (reset sequence length only - fast but may leave stale data)
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Clear the cache and zero GPU memory (slower but guarantees no stale data)
    pub fn clear_and_zero(&mut self, device: &GpuDevice) -> Result<()> {
        self.seq_len = 0;
        self.key.zero(device).map_err(|e| ModelError::CudaError(e.to_string()))?;
        self.value.zero(device).map_err(|e| ModelError::CudaError(e.to_string()))?;
        Ok(())
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.key.size_bytes() + self.value.size_bytes()
    }
}

/// GPU-resident KV Cache for all layers across all GPUs
/// Provides sharded caching for tensor-parallel inference
#[cfg(feature = "cuda")]
pub struct GpuKvCache {
    /// Per-layer, per-GPU caches: layers[layer_idx][gpu_id]
    layers: Vec<Vec<GpuLayerKvCache>>,
    /// Number of layers
    num_layers: usize,
    /// Number of GPUs
    num_gpus: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    /// Create a new GPU-resident KV cache for all layers and GPUs
    pub fn new(
        devices: &[&GpuDevice],
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads_per_gpu: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let num_gpus = devices.len();
        let mut layers = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let mut gpu_caches = Vec::with_capacity(num_gpus);
            for (gpu_id, device) in devices.iter().enumerate() {
                let cache = GpuLayerKvCache::new(
                    device,
                    max_seq_len,
                    num_kv_heads_per_gpu,
                    head_dim,
                )?;
                gpu_caches.push(cache);
            }
            layers.push(gpu_caches);

            if layer_idx % 20 == 0 {
                tracing::debug!("Allocated KV cache for layers 0-{}", layer_idx);
            }
        }

        tracing::info!(
            "GPU KV cache allocated: {} layers × {} GPUs, max_seq_len={}, {:.2} GB total",
            num_layers,
            num_gpus,
            max_seq_len,
            (num_layers * num_gpus * max_seq_len * num_kv_heads_per_gpu * head_dim * 4 * 2) as f64 / 1e9
        );

        Ok(Self {
            layers,
            num_layers,
            num_gpus,
            max_seq_len,
        })
    }

    /// Create KV cache for Llama 70B with tensor parallelism
    pub fn for_llama_70b(
        devices: &[&GpuDevice],
        max_seq_len: usize,
    ) -> Result<Self> {
        let num_gpus = devices.len();
        let num_kv_heads_per_gpu = 8 / num_gpus; // 8 KV heads total for Llama 70B
        Self::new(
            devices,
            80,         // num_layers
            max_seq_len,
            num_kv_heads_per_gpu,
            128,        // head_dim
        )
    }

    /// Get layer cache for a specific GPU
    pub fn layer(&self, layer_idx: usize, gpu_id: usize) -> &GpuLayerKvCache {
        &self.layers[layer_idx][gpu_id]
    }

    /// Get mutable layer cache for a specific GPU
    pub fn layer_mut(&mut self, layer_idx: usize, gpu_id: usize) -> &mut GpuLayerKvCache {
        &mut self.layers[layer_idx][gpu_id]
    }

    /// Current sequence length (same for all layers/GPUs)
    pub fn seq_len(&self) -> usize {
        self.layers.first()
            .and_then(|l| l.first())
            .map(|c| c.seq_len)
            .unwrap_or(0)
    }

    /// Update sequence length for all caches
    pub fn set_seq_len(&mut self, new_len: usize) {
        for layer in &mut self.layers {
            for cache in layer {
                cache.set_seq_len(new_len);
            }
        }
    }

    /// Clear all caches (fast but may leave stale data)
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            for cache in layer {
                cache.clear();
            }
        }
    }

    /// Clear all caches and zero GPU memory (slower but guarantees no stale data)
    pub fn clear_and_zero(&mut self, devices: &[&GpuDevice]) -> Result<()> {
        for layer in &mut self.layers {
            for (gpu_id, cache) in layer.iter_mut().enumerate() {
                cache.clear_and_zero(devices[gpu_id])?;
            }
        }
        Ok(())
    }

    /// Total memory usage in bytes
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter()
            .flat_map(|l| l.iter())
            .map(|c| c.memory_bytes())
            .sum()
    }

    /// Total memory usage in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

// ============================================================================
// CPU-based KV Cache (fallback for non-CUDA builds)
// ============================================================================

/// KV Cache for a single layer (CPU-based, for non-CUDA builds)
pub struct LayerKvCache {
    /// Key cache: [batch, num_kv_heads, seq_len, head_dim]
    pub key: Vec<f32>,
    /// Value cache: [batch, num_kv_heads, seq_len, head_dim]
    pub value: Vec<f32>,
    /// Current sequence length
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl LayerKvCache {
    /// Create a new KV cache
    pub fn new(max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let cache_size = max_seq_len * num_kv_heads * head_dim;

        Self {
            key: vec![0.0; cache_size],
            value: vec![0.0; cache_size],
            seq_len: 0,
            max_seq_len,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append new KV pairs
    pub fn append(&mut self, key: &[f32], value: &[f32], num_new_tokens: usize) -> Result<()> {
        let new_len = self.seq_len + num_new_tokens;

        if new_len > self.max_seq_len {
            return Err(ModelError::MemoryError(format!(
                "KV cache overflow: {} > {}",
                new_len, self.max_seq_len
            )));
        }

        let stride = self.num_kv_heads * self.head_dim;
        let start = self.seq_len * stride;
        let end = new_len * stride;

        self.key[start..end].copy_from_slice(key);
        self.value[start..end].copy_from_slice(value);
        self.seq_len = new_len;

        Ok(())
    }

    /// Get key cache for attention
    pub fn get_keys(&self) -> &[f32] {
        let size = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.key[..size]
    }

    /// Get value cache for attention
    pub fn get_values(&self) -> &[f32] {
        let size = self.seq_len * self.num_kv_heads * self.head_dim;
        &self.value[..size]
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        (self.key.len() + self.value.len()) * std::mem::size_of::<f32>()
    }
}

/// Full KV cache for all layers (CPU-based)
pub struct KvCache {
    /// Per-layer caches
    layers: Vec<LayerKvCache>,
    /// Number of layers
    num_layers: usize,
}

impl KvCache {
    /// Create a new KV cache for all layers
    pub fn new(
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKvCache::new(max_seq_len, num_kv_heads, head_dim))
            .collect();

        Self { layers, num_layers }
    }

    /// Create for Llama 70B
    pub fn for_llama_70b(max_seq_len: usize) -> Self {
        Self::new(
            80,         // num_layers
            max_seq_len,
            8,          // num_kv_heads
            128,        // head_dim (8192 / 64)
        )
    }

    /// Get layer cache
    pub fn layer(&self, idx: usize) -> &LayerKvCache {
        &self.layers[idx]
    }

    /// Get mutable layer cache
    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerKvCache {
        &mut self.layers[idx]
    }

    /// Current sequence length (same for all layers)
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len).unwrap_or(0)
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Total memory usage in bytes
    pub fn total_memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.memory_bytes()).sum()
    }

    /// Total memory usage in GB
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// KV cache configuration
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Use paged attention
    pub paged_attention: bool,
    /// Page size for paged attention
    pub page_size: usize,
}

impl KvCacheConfig {
    /// Create config for Llama 70B with 128K context
    pub fn llama_70b_128k() -> Self {
        Self {
            max_seq_len: 131072,
            num_layers: 80,
            num_kv_heads: 8,
            head_dim: 128,
            paged_attention: true,
            page_size: 16,
        }
    }

    /// Estimate memory usage in GB
    pub fn estimate_memory_gb(&self) -> f64 {
        // KV cache: 2 (K and V) * num_layers * max_seq_len * num_kv_heads * head_dim * 2 (FP16)
        let bytes = 2.0
            * self.num_layers as f64
            * self.max_seq_len as f64
            * self.num_kv_heads as f64
            * self.head_dim as f64
            * 2.0; // FP16

        bytes / (1024.0 * 1024.0 * 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_kv_cache() {
        let mut cache = LayerKvCache::new(1024, 8, 128);

        let key = vec![1.0f32; 8 * 128]; // 1 token
        let value = vec![2.0f32; 8 * 128];

        cache.append(&key, &value, 1).unwrap();
        assert_eq!(cache.seq_len, 1);

        let keys = cache.get_keys();
        assert_eq!(keys.len(), 8 * 128);
        assert_eq!(keys[0], 1.0);
    }

    #[test]
    fn test_kv_cache_memory() {
        let config = KvCacheConfig::llama_70b_128k();
        let mem_gb = config.estimate_memory_gb();

        // Should be around 40GB for 128K context with Llama 70B
        assert!(mem_gb > 30.0 && mem_gb < 50.0, "Expected ~40GB, got {mem_gb}GB");
    }

    #[test]
    fn test_full_kv_cache() {
        let cache = KvCache::for_llama_70b(1024); // Small for testing

        assert_eq!(cache.num_layers, 80);
        assert_eq!(cache.seq_len(), 0);

        let mem_gb = cache.total_memory_gb();
        assert!(mem_gb < 1.0); // Small cache for test
    }
}
