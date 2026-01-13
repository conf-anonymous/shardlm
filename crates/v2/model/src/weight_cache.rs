//! Weight caching for fast subsequent model loads
//!
//! This module saves quantized INT8 weights to disk after the first load,
//! enabling fast subsequent loads (~30-60s vs ~20-30min for full quantization).
//!
//! Cache format: `.shardlm-cache/` directory containing:
//! - `manifest.json`: Metadata about cached weights (model config, quantization, checksums)
//! - `embed_tokens.bin`: Embedding weights (BF16, sharded)
//! - `layer_XX.bin`: Per-layer weights (INT8 quantized)
//! - `norm.bin`: Final norm weights
//! - `lm_head.bin`: LM head weights (BF16, sharded)

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_core::quantization::QuantizationMode;
use shardlm_v2_core::tensor::DType;

use crate::error::{ModelError, Result};

/// Cache file format version - increment when format changes
const CACHE_VERSION: u32 = 1;

/// Magic bytes to identify cache files
const CACHE_MAGIC: &[u8; 8] = b"SHARDLM\x00";

/// Manifest file containing cache metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheManifest {
    /// Cache format version
    pub version: u32,
    /// Model architecture
    pub architecture: String,
    /// Number of layers
    pub num_layers: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Quantization mode used
    pub quantization: String,
    /// Number of GPUs (shards)
    pub num_gpus: usize,
    /// Checksums for each cached file
    pub checksums: HashMap<String, u64>,
    /// Total cached size in bytes
    pub total_size_bytes: u64,
}

/// A single cached tensor (can be sharded across GPUs)
#[derive(Debug)]
pub struct CachedTensor {
    /// Tensor shape (for each shard)
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Raw data for each GPU shard
    pub shards: Vec<Vec<u8>>,
}

/// Weight cache for saving/loading quantized weights
pub struct WeightCache {
    /// Cache directory
    cache_dir: PathBuf,
    /// Model directory (for source file comparison)
    model_dir: PathBuf,
    /// Number of GPUs
    num_gpus: usize,
}

impl WeightCache {
    /// Create a new weight cache for the given model directory
    pub fn new(model_dir: impl AsRef<Path>, num_gpus: usize) -> Self {
        let model_dir = model_dir.as_ref().to_path_buf();
        let cache_dir = model_dir.join(".shardlm-cache");

        Self {
            cache_dir,
            model_dir,
            num_gpus,
        }
    }

    /// Check if a valid cache exists
    pub fn exists(&self) -> bool {
        let manifest_path = self.cache_dir.join("manifest.json");
        if !manifest_path.exists() {
            return false;
        }

        // Try to load and validate manifest
        match self.load_manifest() {
            Ok(manifest) => {
                // Check version compatibility
                if manifest.version != CACHE_VERSION {
                    println!("Cache version mismatch: found {}, expected {}", manifest.version, CACHE_VERSION);
                    return false;
                }
                // Check GPU count matches
                if manifest.num_gpus != self.num_gpus {
                    println!("Cache GPU count mismatch: found {}, expected {}", manifest.num_gpus, self.num_gpus);
                    return false;
                }
                // Check that layer files exist (first and last layer)
                // This catches incomplete caches that were interrupted during save
                let first_layer = self.cache_dir.join("layer_0_q_proj.bin");
                let last_layer = self.cache_dir.join(format!("layer_{}_q_proj.bin", manifest.num_layers - 1));
                if !first_layer.exists() {
                    println!("Cache incomplete: missing layer_0_q_proj.bin");
                    return false;
                }
                if !last_layer.exists() {
                    println!("Cache incomplete: missing layer_{}_q_proj.bin", manifest.num_layers - 1);
                    return false;
                }
                true
            }
            Err(e) => {
                println!("Failed to load cache manifest: {}", e);
                false
            }
        }
    }

    /// Load the cache manifest
    pub fn load_manifest(&self) -> Result<CacheManifest> {
        let manifest_path = self.cache_dir.join("manifest.json");
        let file = File::open(&manifest_path)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to open manifest: {}", e))))?;
        let reader = BufReader::new(file);
        let manifest: CacheManifest = serde_json::from_reader(reader)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to parse manifest: {}", e)))?;
        Ok(manifest)
    }

    /// Create the cache directory
    pub fn create_cache_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.cache_dir)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to create cache dir: {}", e))))?;
        Ok(())
    }

    /// Save the cache manifest
    pub fn save_manifest(&self, manifest: &CacheManifest) -> Result<()> {
        let manifest_path = self.cache_dir.join("manifest.json");
        let file = File::create(&manifest_path)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to create manifest: {}", e))))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, manifest)
            .map_err(|e| ModelError::InvalidFormat(format!("Failed to write manifest: {}", e)))?;
        Ok(())
    }

    /// Save a tensor to cache
    ///
    /// Format:
    /// - 8 bytes: magic "SHARDLM\0"
    /// - 4 bytes: version (u32 LE)
    /// - 4 bytes: dtype (u32 LE, 0=BF16, 1=F16, 2=F32, 3=I8)
    /// - 4 bytes: num_dims (u32 LE)
    /// - num_dims * 8 bytes: shape (u64 LE each)
    /// - 4 bytes: num_shards (u32 LE)
    /// - For each shard:
    ///   - 8 bytes: shard_size (u64 LE)
    ///   - shard_size bytes: data
    pub fn save_tensor(&self, name: &str, tensor: &CachedTensor) -> Result<u64> {
        let path = self.cache_dir.join(format!("{}.bin", name));
        let file = File::create(&path)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to create {}: {}", name, e))))?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(CACHE_MAGIC)?;
        writer.write_all(&CACHE_VERSION.to_le_bytes())?;

        let dtype_id: u32 = match tensor.dtype {
            DType::BF16 => 0,
            DType::F16 => 1,
            DType::F32 => 2,
            DType::I8 => 3,
            DType::I32 => 4,
        };
        writer.write_all(&dtype_id.to_le_bytes())?;

        // Write shape
        let num_dims = tensor.shape.len() as u32;
        writer.write_all(&num_dims.to_le_bytes())?;
        for &dim in &tensor.shape {
            writer.write_all(&(dim as u64).to_le_bytes())?;
        }

        // Write shards
        let num_shards = tensor.shards.len() as u32;
        writer.write_all(&num_shards.to_le_bytes())?;

        let mut total_size: u64 = 8 + 4 + 4 + 4 + (tensor.shape.len() * 8) as u64 + 4;

        for shard in &tensor.shards {
            let shard_size = shard.len() as u64;
            writer.write_all(&shard_size.to_le_bytes())?;
            writer.write_all(shard)?;
            total_size += 8 + shard_size;
        }

        writer.flush()?;
        Ok(total_size)
    }

    /// Load a tensor from cache
    pub fn load_tensor(&self, name: &str) -> Result<CachedTensor> {
        let path = self.cache_dir.join(format!("{}.bin", name));
        let file = File::open(&path)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to open {}: {}", name, e))))?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(ModelError::InvalidFormat(format!("Invalid cache magic for {}", name)));
        }

        // Read and verify version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != CACHE_VERSION {
            return Err(ModelError::InvalidFormat(format!("Cache version mismatch for {}: found {}, expected {}", name, version, CACHE_VERSION)));
        }

        // Read dtype
        let mut dtype_bytes = [0u8; 4];
        reader.read_exact(&mut dtype_bytes)?;
        let dtype_id = u32::from_le_bytes(dtype_bytes);
        let dtype = match dtype_id {
            0 => DType::BF16,
            1 => DType::F16,
            2 => DType::F32,
            3 => DType::I8,
            4 => DType::I32,
            _ => return Err(ModelError::InvalidFormat(format!("Unknown dtype {} in cache for {}", dtype_id, name))),
        };

        // Read shape
        let mut num_dims_bytes = [0u8; 4];
        reader.read_exact(&mut num_dims_bytes)?;
        let num_dims = u32::from_le_bytes(num_dims_bytes) as usize;

        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes)?;
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        // Read shards
        let mut num_shards_bytes = [0u8; 4];
        reader.read_exact(&mut num_shards_bytes)?;
        let num_shards = u32::from_le_bytes(num_shards_bytes) as usize;

        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            let mut size_bytes = [0u8; 8];
            reader.read_exact(&mut size_bytes)?;
            let shard_size = u64::from_le_bytes(size_bytes) as usize;

            let mut shard_data = vec![0u8; shard_size];
            reader.read_exact(&mut shard_data)?;
            shards.push(shard_data);
        }

        Ok(CachedTensor {
            shape,
            dtype,
            shards,
        })
    }

    /// Save per-channel scales for a tensor
    pub fn save_scales(&self, name: &str, scales: &[Vec<f32>]) -> Result<u64> {
        let path = self.cache_dir.join(format!("{}.scales", name));
        let file = File::create(&path)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to create scales for {}: {}", name, e))))?;
        let mut writer = BufWriter::new(file);

        // Write header
        writer.write_all(CACHE_MAGIC)?;
        writer.write_all(&CACHE_VERSION.to_le_bytes())?;

        // Write number of shards
        let num_shards = scales.len() as u32;
        writer.write_all(&num_shards.to_le_bytes())?;

        let mut total_size: u64 = 8 + 4 + 4;

        for shard_scales in scales {
            let num_scales = shard_scales.len() as u64;
            writer.write_all(&num_scales.to_le_bytes())?;

            for &scale in shard_scales {
                writer.write_all(&scale.to_le_bytes())?;
            }

            total_size += 8 + (shard_scales.len() * 4) as u64;
        }

        writer.flush()?;
        Ok(total_size)
    }

    /// Load per-channel scales for a tensor
    pub fn load_scales(&self, name: &str) -> Result<Vec<Vec<f32>>> {
        let path = self.cache_dir.join(format!("{}.scales", name));
        let file = File::open(&path)
            .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to open scales for {}: {}", name, e))))?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(ModelError::InvalidFormat(format!("Invalid scales magic for {}", name)));
        }

        // Read version
        let mut version_bytes = [0u8; 4];
        reader.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != CACHE_VERSION {
            return Err(ModelError::InvalidFormat(format!("Scales version mismatch for {}", name)));
        }

        // Read number of shards
        let mut num_shards_bytes = [0u8; 4];
        reader.read_exact(&mut num_shards_bytes)?;
        let num_shards = u32::from_le_bytes(num_shards_bytes) as usize;

        let mut all_scales = Vec::with_capacity(num_shards);

        for _ in 0..num_shards {
            let mut num_scales_bytes = [0u8; 8];
            reader.read_exact(&mut num_scales_bytes)?;
            let num_scales = u64::from_le_bytes(num_scales_bytes) as usize;

            let mut shard_scales = Vec::with_capacity(num_scales);
            for _ in 0..num_scales {
                let mut scale_bytes = [0u8; 4];
                reader.read_exact(&mut scale_bytes)?;
                shard_scales.push(f32::from_le_bytes(scale_bytes));
            }
            all_scales.push(shard_scales);
        }

        Ok(all_scales)
    }

    /// Delete the cache
    pub fn delete(&self) -> Result<()> {
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| ModelError::Io(std::io::Error::new(e.kind(), format!("Failed to delete cache: {}", e))))?;
        }
        Ok(())
    }

    /// Get cache directory path
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Simple hash for checksum (FNV-1a)
    pub fn compute_checksum(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;

        let mut hash = FNV_OFFSET;
        for &byte in data {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }
}

/// Builder for creating a cache during model loading
pub struct CacheBuilder {
    cache: WeightCache,
    manifest: CacheManifest,
}

impl CacheBuilder {
    /// Create a new cache builder
    pub fn new(
        model_dir: impl AsRef<Path>,
        architecture: ModelArchitecture,
        quantization: QuantizationMode,
        num_gpus: usize,
        config: &shardlm_v2_core::model::LlamaConfig,
    ) -> Result<Self> {
        let cache = WeightCache::new(&model_dir, num_gpus);
        cache.create_cache_dir()?;

        let manifest = CacheManifest {
            version: CACHE_VERSION,
            architecture: format!("{:?}", architecture),
            num_layers: config.num_layers,
            hidden_dim: config.hidden_dim,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            vocab_size: config.vocab_size,
            quantization: format!("{:?}", quantization),
            num_gpus,
            checksums: HashMap::new(),
            total_size_bytes: 0,
        };

        Ok(Self { cache, manifest })
    }

    /// Save a tensor and record in manifest
    pub fn save_tensor(&mut self, name: &str, tensor: &CachedTensor) -> Result<()> {
        let size = self.cache.save_tensor(name, tensor)?;

        // Compute checksum of all shard data
        let mut all_data = Vec::new();
        for shard in &tensor.shards {
            all_data.extend_from_slice(shard);
        }
        let checksum = WeightCache::compute_checksum(&all_data);

        self.manifest.checksums.insert(name.to_string(), checksum);
        self.manifest.total_size_bytes += size;

        Ok(())
    }

    /// Save scales for a tensor
    pub fn save_scales(&mut self, name: &str, scales: &[Vec<f32>]) -> Result<()> {
        let size = self.cache.save_scales(name, scales)?;
        self.manifest.total_size_bytes += size;
        Ok(())
    }

    /// Finalize the cache and write manifest
    pub fn finalize(self) -> Result<()> {
        self.cache.save_manifest(&self.manifest)?;
        println!("Cache saved: {:.2} GB total", self.manifest.total_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cache_tensor_roundtrip() {
        let temp_dir = env::temp_dir().join("shardlm-cache-test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let cache = WeightCache::new(&temp_dir, 2);
        cache.create_cache_dir().unwrap();

        // Create test tensor
        let tensor = CachedTensor {
            shape: vec![4, 8],
            dtype: DType::I8,
            shards: vec![
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                vec![17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            ],
        };

        // Save and load
        cache.save_tensor("test_weight", &tensor).unwrap();
        let loaded = cache.load_tensor("test_weight").unwrap();

        assert_eq!(loaded.shape, tensor.shape);
        assert_eq!(loaded.dtype, tensor.dtype);
        assert_eq!(loaded.shards.len(), tensor.shards.len());
        assert_eq!(loaded.shards[0], tensor.shards[0]);
        assert_eq!(loaded.shards[1], tensor.shards[1]);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_cache_scales_roundtrip() {
        let temp_dir = env::temp_dir().join("shardlm-scales-test");
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();

        let cache = WeightCache::new(&temp_dir, 2);
        cache.create_cache_dir().unwrap();

        // Create test scales
        let scales = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.6, 0.7, 0.8],
        ];

        // Save and load
        cache.save_scales("test_scales", &scales).unwrap();
        let loaded = cache.load_scales("test_scales").unwrap();

        assert_eq!(loaded.len(), scales.len());
        for (orig, load) in scales.iter().zip(loaded.iter()) {
            for (a, b) in orig.iter().zip(load.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_checksum() {
        let data1 = b"hello world";
        let data2 = b"hello world";
        let data3 = b"hello worle";

        assert_eq!(
            WeightCache::compute_checksum(data1),
            WeightCache::compute_checksum(data2)
        );
        assert_ne!(
            WeightCache::compute_checksum(data1),
            WeightCache::compute_checksum(data3)
        );
    }
}
