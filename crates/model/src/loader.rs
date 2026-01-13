//! Safetensors model weight loading

use std::fs::File;
use std::io::Read;
use std::path::Path;

use safetensors::SafeTensors;

use crate::config::TinyLlamaConfig;
use crate::embedding::EmbeddingTable;
use crate::error::{ModelError, Result};
use crate::weights::{
    AttentionWeights, LinearWeights, MlpWeights, ModelWeights, RmsNormWeights,
    TransformerLayerWeights,
};

/// Load model weights from a safetensors file
pub struct SafetensorsLoader {
    /// Raw safetensors data
    data: Vec<u8>,
    /// Model configuration
    pub config: TinyLlamaConfig,
    /// Fixed-point scale
    scale: u8,
}

impl SafetensorsLoader {
    /// Load from a directory containing model.safetensors and config.json
    pub fn from_directory<P: AsRef<Path>>(dir: P, scale: u8) -> Result<Self> {
        let dir = dir.as_ref();

        // Load config
        let config_path = dir.join("config.json");
        let config = TinyLlamaConfig::from_json_file(&config_path)?;
        config.validate()?;

        // Load safetensors
        let model_path = dir.join("model.safetensors");
        let mut file = File::open(&model_path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        Ok(Self { data, config, scale })
    }

    /// Get tensor names in the safetensors file
    pub fn tensor_names(&self) -> Result<Vec<String>> {
        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|e| ModelError::Safetensors(e.to_string()))?;
        Ok(tensors.names().into_iter().map(String::from).collect())
    }

    /// Load a tensor as f32 values
    fn load_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|e| ModelError::Safetensors(e.to_string()))?;

        let tensor = tensors
            .tensor(name)
            .map_err(|_| ModelError::MissingTensor(name.to_string()))?;

        let data = tensor.data();
        let dtype = tensor.dtype();

        // Convert based on dtype
        let f32_values: Vec<f32> = match dtype {
            safetensors::Dtype::F32 => {
                // Direct f32
                data.chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect()
            }
            safetensors::Dtype::BF16 => {
                // BFloat16 -> f32
                data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        bf16_to_f32(bits)
                    })
                    .collect()
            }
            safetensors::Dtype::F16 => {
                // Float16 -> f32
                data.chunks_exact(2)
                    .map(|b| {
                        let bits = u16::from_le_bytes([b[0], b[1]]);
                        f16_to_f32(bits)
                    })
                    .collect()
            }
            _ => {
                return Err(ModelError::InvalidDtype {
                    expected: "f32, bf16, or f16".to_string(),
                    got: format!("{:?}", dtype),
                });
            }
        };

        Ok(f32_values)
    }

    /// Load the embedding table
    pub fn load_embeddings(&self) -> Result<EmbeddingTable> {
        let data = self.load_tensor_f32("model.embed_tokens.weight")?;

        EmbeddingTable::from_f32(
            &data,
            self.config.vocab_size,
            self.config.hidden_size,
            self.scale,
        )
    }

    /// Load a linear layer (weight only, no bias for LLaMA)
    pub fn load_linear(
        &self,
        weight_name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Result<LinearWeights> {
        self.load_linear_with_bias(weight_name, None, in_features, out_features)
    }

    /// Load a linear layer with optional bias
    pub fn load_linear_with_bias(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        in_features: usize,
        out_features: usize,
    ) -> Result<LinearWeights> {
        let data = self.load_tensor_f32(weight_name)?;

        // Safetensors stores as [out_features, in_features], we need [in_features, out_features]
        // So we need to transpose
        let mut transposed = vec![0.0f32; in_features * out_features];
        for i in 0..out_features {
            for j in 0..in_features {
                transposed[j * out_features + i] = data[i * in_features + j];
            }
        }

        // Load bias if present
        let bias_f32 = if let Some(bias_name) = bias_name {
            match self.load_tensor_f32(bias_name) {
                Ok(b) => Some(b),
                Err(_) => None, // Bias not found, that's okay
            }
        } else {
            None
        };

        LinearWeights::from_f32(&transposed, bias_f32.as_deref(), in_features, out_features, self.scale)
    }

    /// Load RMS norm weights
    pub fn load_rms_norm(&self, name: &str, eps: f64) -> Result<RmsNormWeights> {
        let data = self.load_tensor_f32(name)?;
        let scale_factor = (1u64 << self.scale) as f64;
        let weight: Vec<i32> = data
            .iter()
            .map(|&x| (x as f64 * scale_factor).round() as i32)
            .collect();
        Ok(RmsNormWeights::from_raw(weight, self.scale, eps))
    }

    /// Load a single transformer layer
    fn load_layer(&self, layer_idx: usize) -> Result<TransformerLayerWeights> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let eps = self.config.rms_norm_eps;

        let prefix = format!("model.layers.{}", layer_idx);

        // Load layer norms
        let input_layernorm =
            self.load_rms_norm(&format!("{}.input_layernorm.weight", prefix), eps)?;
        let post_attention_layernorm =
            self.load_rms_norm(&format!("{}.post_attention_layernorm.weight", prefix), eps)?;

        // Load attention weights (with optional bias for Qwen-style models)
        let q_proj = self.load_linear_with_bias(
            &format!("{}.self_attn.q_proj.weight", prefix),
            Some(&format!("{}.self_attn.q_proj.bias", prefix)),
            hidden_size,
            num_heads * head_dim,
        )?;
        let k_proj = self.load_linear_with_bias(
            &format!("{}.self_attn.k_proj.weight", prefix),
            Some(&format!("{}.self_attn.k_proj.bias", prefix)),
            hidden_size,
            num_kv_heads * head_dim,
        )?;
        let v_proj = self.load_linear_with_bias(
            &format!("{}.self_attn.v_proj.weight", prefix),
            Some(&format!("{}.self_attn.v_proj.bias", prefix)),
            hidden_size,
            num_kv_heads * head_dim,
        )?;
        let o_proj = self.load_linear(
            &format!("{}.self_attn.o_proj.weight", prefix),
            num_heads * head_dim,
            hidden_size,
        )?;

        // Load MLP weights
        let gate_proj = self.load_linear(
            &format!("{}.mlp.gate_proj.weight", prefix),
            hidden_size,
            intermediate_size,
        )?;
        let up_proj = self.load_linear(
            &format!("{}.mlp.up_proj.weight", prefix),
            hidden_size,
            intermediate_size,
        )?;
        let down_proj = self.load_linear(
            &format!("{}.mlp.down_proj.weight", prefix),
            intermediate_size,
            hidden_size,
        )?;

        Ok(TransformerLayerWeights {
            input_layernorm,
            self_attn: AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj,
            },
            post_attention_layernorm,
            mlp: MlpWeights {
                gate_proj,
                up_proj,
                down_proj,
            },
        })
    }

    /// Load full model weights
    pub fn load_model_weights(&self) -> Result<ModelWeights> {
        let hidden_size = self.config.hidden_size;
        let num_layers = self.config.num_hidden_layers;
        let eps = self.config.rms_norm_eps;

        println!("Loading embeddings...");
        let embeddings = self.load_embeddings()?;

        println!("Loading LM head...");
        let lm_head = if self.config.tie_word_embeddings {
            // For models with tied embeddings (like Qwen), lm_head uses embed_tokens.weight
            println!("  (using tied embeddings from embed_tokens)");
            self.load_linear("model.embed_tokens.weight", hidden_size, self.config.vocab_size)?
        } else {
            // Separate lm_head (like TinyLlama/Llama)
            self.load_linear("lm_head.weight", hidden_size, self.config.vocab_size)?
        };

        println!("Loading final norm...");
        let final_norm = self.load_rms_norm("model.norm.weight", eps)?;

        println!("Loading {} transformer layers...", num_layers);
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if i % 5 == 0 {
                println!("  Loading layer {}/{}...", i + 1, num_layers);
            }
            layers.push(self.load_layer(i)?);
        }

        Ok(ModelWeights {
            config: self.config.clone(),
            embeddings,
            layers,
            final_norm,
            lm_head,
            scale: self.scale,
        })
    }
}

/// Convert BFloat16 to f32
fn bf16_to_f32(bits: u16) -> f32 {
    // BF16 is just the upper 16 bits of f32
    let f32_bits = (bits as u32) << 16;
    f32::from_bits(f32_bits)
}

/// Convert Float16 to f32
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal
            let mut e = -14i32;
            let mut f = frac;
            while (f & 0x400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3FF;
            let f32_exp = ((e + 127) as u32) << 23;
            let f32_frac = f << 13;
            f32::from_bits((sign << 31) | f32_exp | f32_frac)
        }
    } else if exp == 31 {
        // Inf or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (frac << 13);
        f32::from_bits(f32_bits)
    } else {
        // Normal
        let f32_exp = ((exp as i32 - 15 + 127) as u32) << 23;
        let f32_frac = frac << 13;
        f32::from_bits((sign << 31) | f32_exp | f32_frac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_conversion() {
        // Test 1.0 in BF16 (0x3F80)
        let one = bf16_to_f32(0x3F80);
        assert!((one - 1.0).abs() < 0.01, "Got {}", one);

        // Test -1.0 in BF16 (0xBF80)
        let neg_one = bf16_to_f32(0xBF80);
        assert!((neg_one - (-1.0)).abs() < 0.01, "Got {}", neg_one);

        // Test 0.0 in BF16 (0x0000)
        let zero = bf16_to_f32(0x0000);
        assert!((zero - 0.0).abs() < 0.001, "Got {}", zero);
    }
}
