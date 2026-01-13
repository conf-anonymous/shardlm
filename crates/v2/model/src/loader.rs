//! Model weight loading

use std::collections::HashMap;
use std::path::Path;

use memmap2::Mmap;
use safetensors::SafeTensors;
use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_core::model::{LlamaConfig, WeightDtype, WeightInfo};
use shardlm_v2_core::tensor::{DType, Device, GpuTensor};

use crate::error::{ModelError, Result};

/// Model loader for memory-mapped safetensors
pub struct ModelLoader {
    /// Model configuration
    config: LlamaConfig,
    /// Path to model directory
    model_dir: std::path::PathBuf,
    /// Memory-mapped files
    mmaps: HashMap<String, Mmap>,
    /// Target device
    device: Device,
}

impl ModelLoader {
    /// Create a new model loader
    pub fn new(model_dir: impl AsRef<Path>, arch: ModelArchitecture) -> Result<Self> {
        let model_dir = model_dir.as_ref().to_path_buf();

        if !model_dir.exists() {
            return Err(ModelError::ModelNotFound(model_dir.display().to_string()));
        }

        let config = LlamaConfig::from_architecture(arch);

        Ok(Self {
            config,
            model_dir,
            mmaps: HashMap::new(),
            device: Device::Cpu,
        })
    }

    /// Set target device
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Get model configuration
    pub fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// List safetensor files in model directory
    pub fn list_safetensor_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let mut files = Vec::new();

        for entry in std::fs::read_dir(&self.model_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "safetensors") {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    /// Memory-map a safetensor file
    pub fn mmap_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let filename = path.file_name().unwrap().to_string_lossy().to_string();
        self.mmaps.insert(filename, mmap);

        Ok(())
    }

    /// Load all safetensor files
    pub fn load_all_files(&mut self) -> Result<()> {
        let files = self.list_safetensor_files()?;

        for file in files {
            self.mmap_file(&file)?;
        }

        Ok(())
    }

    /// Get weight tensor by name
    pub fn get_weight(&self, name: &str) -> Result<GpuTensor> {
        // Search through all memory-mapped files
        for (filename, mmap) in &self.mmaps {
            let safetensors = SafeTensors::deserialize(&mmap)
                .map_err(|e| ModelError::InvalidFormat(format!("{}: {}", filename, e)))?;

            if let Ok(tensor) = safetensors.tensor(name) {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let dtype = match tensor.dtype() {
                    safetensors::Dtype::BF16 => DType::BF16,
                    safetensors::Dtype::F16 => DType::F16,
                    safetensors::Dtype::F32 => DType::F32,
                    dt => return Err(ModelError::UnsupportedDtype(format!("{:?}", dt))),
                };

                let data = tensor.data().to_vec();

                return Ok(GpuTensor::new_cpu(shape, dtype, data));
            }
        }

        Err(ModelError::MissingWeight(name.to_string()))
    }

    /// Load all weights for a layer
    pub fn load_layer(&self, layer_idx: usize) -> Result<LayerWeights> {
        let prefix = format!("model.layers.{layer_idx}");

        Ok(LayerWeights {
            q_proj: self.get_weight(&format!("{prefix}.self_attn.q_proj.weight"))?,
            k_proj: self.get_weight(&format!("{prefix}.self_attn.k_proj.weight"))?,
            v_proj: self.get_weight(&format!("{prefix}.self_attn.v_proj.weight"))?,
            o_proj: self.get_weight(&format!("{prefix}.self_attn.o_proj.weight"))?,
            gate_proj: self.get_weight(&format!("{prefix}.mlp.gate_proj.weight"))?,
            up_proj: self.get_weight(&format!("{prefix}.mlp.up_proj.weight"))?,
            down_proj: self.get_weight(&format!("{prefix}.mlp.down_proj.weight"))?,
            input_layernorm: self.get_weight(&format!("{prefix}.input_layernorm.weight"))?,
            post_attn_layernorm: self
                .get_weight(&format!("{prefix}.post_attention_layernorm.weight"))?,
        })
    }

    /// Calculate total model size in bytes
    pub fn total_size_bytes(&self) -> usize {
        self.mmaps.values().map(|m| m.len()).sum()
    }

    /// Calculate model size in GB
    pub fn total_size_gb(&self) -> f64 {
        self.total_size_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Weights for a single transformer layer
pub struct LayerWeights {
    pub q_proj: GpuTensor,
    pub k_proj: GpuTensor,
    pub v_proj: GpuTensor,
    pub o_proj: GpuTensor,
    pub gate_proj: GpuTensor,
    pub up_proj: GpuTensor,
    pub down_proj: GpuTensor,
    pub input_layernorm: GpuTensor,
    pub post_attn_layernorm: GpuTensor,
}

/// Model weights container
pub struct ModelWeights {
    /// Token embeddings
    pub embed_tokens: GpuTensor,
    /// Layer weights
    pub layers: Vec<LayerWeights>,
    /// Final layer norm
    pub norm: GpuTensor,
    /// LM head
    pub lm_head: GpuTensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_arch() {
        let loader = ModelLoader::new("/tmp/fake-model", ModelArchitecture::Llama3_1_70B);
        // Should fail since path doesn't exist, but config should be set
        assert!(loader.is_err());
    }
}
