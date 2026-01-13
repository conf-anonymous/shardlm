//! INT8 quantization for memory-efficient weight storage
//!
//! This module implements per-channel symmetric INT8 quantization for weights:
//! - Weights are stored as INT8 with FP32 scale factors
//! - Activations remain in BF16/FP16 for quality
//! - Dequantization happens during matmul: output = (int8_weight * scale) @ activation
//!
//! Memory savings: BF16 (2 bytes) -> INT8 (1 byte) = 50% reduction
//!
//! Uses rayon for parallel quantization to speed up CPU processing.

use half::bf16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Quantization configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMode {
    /// No quantization (BF16 weights)
    None,
    /// INT8 weights with FP32 scales (per-channel)
    Int8PerChannel,
    /// INT8 weights with FP32 scales (per-tensor)
    Int8PerTensor,
}

impl Default for QuantizationMode {
    fn default() -> Self {
        Self::None
    }
}

/// Quantized tensor with scale factors
#[derive(Debug)]
pub struct QuantizedTensor {
    /// INT8 quantized weights
    pub data: Vec<i8>,
    /// Scale factors (one per output channel for per-channel, one for per-tensor)
    pub scales: Vec<f32>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization mode
    pub mode: QuantizationMode,
}

impl QuantizedTensor {
    /// Get memory size in bytes (data + scales)
    pub fn size_bytes(&self) -> usize {
        self.data.len() + self.scales.len() * 4
    }

    /// Get the equivalent BF16 size for comparison
    pub fn bf16_equivalent_size(&self) -> usize {
        self.shape.iter().product::<usize>() * 2
    }

    /// Get memory savings ratio
    pub fn savings_ratio(&self) -> f32 {
        self.size_bytes() as f32 / self.bf16_equivalent_size() as f32
    }
}

/// Quantize BF16 weights to INT8 with per-channel scaling (parallel)
///
/// For a weight matrix [out_features, in_features]:
/// - Compute max absolute value per output channel
/// - Scale = max_abs / 127
/// - Quantized = round(weight / scale)
///
/// Uses rayon for parallel processing across rows.
pub fn quantize_bf16_to_int8_per_channel(
    data: &[u8],
    shape: &[usize],
) -> QuantizedTensor {
    assert_eq!(shape.len(), 2, "Only 2D tensors supported for quantization");
    let (out_features, in_features) = (shape[0], shape[1]);

    // Convert bytes to bf16 values in parallel
    let bf16_data: Vec<bf16> = data
        .par_chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();

    // Process rows in parallel - each row produces (scale, quantized_row)
    let results: Vec<(f32, Vec<i8>)> = (0..out_features)
        .into_par_iter()
        .map(|row| {
            let row_start = row * in_features;
            let row_end = row_start + in_features;
            let row_data = &bf16_data[row_start..row_end];

            // Find max absolute value in this row
            let max_abs = row_data
                .iter()
                .map(|v| v.to_f32().abs())
                .fold(0.0f32, f32::max);

            // Compute scale (avoid division by zero)
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

            // Quantize each value in the row
            let quantized_row: Vec<i8> = row_data
                .iter()
                .map(|&val| {
                    let f32_val = val.to_f32();
                    (f32_val / scale).round().clamp(-128.0, 127.0) as i8
                })
                .collect();

            (scale, quantized_row)
        })
        .collect();

    // Flatten results
    let mut scales = Vec::with_capacity(out_features);
    let mut quantized = Vec::with_capacity(out_features * in_features);
    for (scale, row_data) in results {
        scales.push(scale);
        quantized.extend(row_data);
    }

    QuantizedTensor {
        data: quantized,
        scales,
        shape: shape.to_vec(),
        mode: QuantizationMode::Int8PerChannel,
    }
}

/// Quantize BF16 weights to INT8 with per-tensor scaling (simpler, slightly less accurate)
///
/// Uses rayon for parallel processing.
pub fn quantize_bf16_to_int8_per_tensor(
    data: &[u8],
    shape: &[usize],
) -> QuantizedTensor {
    // Convert bytes to bf16 values in parallel
    let bf16_data: Vec<bf16> = data
        .par_chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();

    // Find global max absolute value in parallel
    let max_abs = bf16_data
        .par_iter()
        .map(|v| v.to_f32().abs())
        .reduce(|| 0.0f32, f32::max);

    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };

    // Quantize all values in parallel
    let quantized: Vec<i8> = bf16_data
        .par_iter()
        .map(|&val| {
            let f32_val = val.to_f32();
            (f32_val / scale).round().clamp(-128.0, 127.0) as i8
        })
        .collect();

    QuantizedTensor {
        data: quantized,
        scales: vec![scale],
        shape: shape.to_vec(),
        mode: QuantizationMode::Int8PerTensor,
    }
}

/// Dequantize INT8 back to BF16 (for verification or fallback)
pub fn dequantize_int8_to_bf16(quantized: &QuantizedTensor) -> Vec<u8> {
    let (out_features, in_features) = (quantized.shape[0], quantized.shape[1]);
    let mut result = Vec::with_capacity(out_features * in_features * 2);

    match quantized.mode {
        QuantizationMode::Int8PerChannel => {
            for row in 0..out_features {
                let scale = quantized.scales[row];
                let row_start = row * in_features;

                for col in 0..in_features {
                    let int8_val = quantized.data[row_start + col];
                    let f32_val = int8_val as f32 * scale;
                    let bf16_val = bf16::from_f32(f32_val);
                    result.extend_from_slice(&bf16_val.to_bits().to_le_bytes());
                }
            }
        }
        QuantizationMode::Int8PerTensor => {
            let scale = quantized.scales[0];
            for &int8_val in &quantized.data {
                let f32_val = int8_val as f32 * scale;
                let bf16_val = bf16::from_f32(f32_val);
                result.extend_from_slice(&bf16_val.to_bits().to_le_bytes());
            }
        }
        QuantizationMode::None => {
            panic!("Cannot dequantize non-quantized tensor");
        }
    }

    result
}

/// Calculate quantization error (RMSE) for verification
pub fn calculate_quantization_error(original: &[u8], quantized: &QuantizedTensor) -> f32 {
    let dequantized = dequantize_int8_to_bf16(quantized);

    let original_bf16: Vec<f32> = original
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect();

    let dequantized_bf16: Vec<f32> = dequantized
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect();

    let mse: f32 = original_bf16
        .iter()
        .zip(dequantized_bf16.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / original_bf16.len() as f32;

    mse.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_per_channel() {
        // Create a larger tensor to see meaningful savings
        // 128x256 = 32768 elements
        let values: Vec<f32> = (0..32768).map(|i| (i as f32 - 16384.0) / 1000.0).collect();
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();

        let quantized = quantize_bf16_to_int8_per_channel(&bf16_data, &[128, 256]);

        assert_eq!(quantized.shape, vec![128, 256]);
        assert_eq!(quantized.data.len(), 32768);
        assert_eq!(quantized.scales.len(), 128);

        // Check savings - for large tensors should be close to 50%
        // INT8: 32768 bytes + 128 * 4 = 33280 bytes
        // BF16: 32768 * 2 = 65536 bytes
        // Ratio: 33280 / 65536 ≈ 0.508
        let ratio = quantized.savings_ratio();
        assert!(ratio < 0.55, "Savings ratio {} should be < 0.55", ratio);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        // Create test data
        let values: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();

        let quantized = quantize_bf16_to_int8_per_channel(&bf16_data, &[4, 4]);
        let error = calculate_quantization_error(&bf16_data, &quantized);

        // Error should be small (< 1% of max value)
        assert!(error < 0.05, "Quantization error too high: {}", error);
    }

    #[test]
    fn test_per_tensor_quantization() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let bf16_data: Vec<u8> = values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();

        let quantized = quantize_bf16_to_int8_per_tensor(&bf16_data, &[2, 4]);

        assert_eq!(quantized.scales.len(), 1);
        assert_eq!(quantized.data.len(), 8);
    }
}
