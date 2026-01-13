//! Tensor types for v2 GPU operations

use std::sync::Arc;

/// Tensor data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I8,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 => 1,
        }
    }
}

/// Device location for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize), // GPU device ID
}

/// GPU tensor wrapper (placeholder for actual CUDA tensor)
#[derive(Debug)]
pub struct GpuTensor {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Device location
    pub device: Device,
    /// Raw data (CPU fallback, will be replaced with CUDA memory)
    data: Arc<Vec<u8>>,
}

impl GpuTensor {
    /// Create a new tensor on CPU
    pub fn new_cpu(shape: Vec<usize>, dtype: DType, data: Vec<u8>) -> Self {
        Self {
            shape,
            dtype,
            device: Device::Cpu,
            data: Arc::new(data),
        }
    }

    /// Get the number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    /// Check if tensor is on GPU
    pub fn is_cuda(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
    }

    /// Get data as slice (CPU only)
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if self.device == Device::Cpu {
            Some(&self.data)
        } else {
            None
        }
    }
}

/// Tensor shape utilities
pub mod shape {
    /// Calculate the product of dimensions
    pub fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Compute strides for row-major layout
    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Broadcast shapes
    pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
        let max_len = a.len().max(b.len());
        let mut result = vec![0; max_len];

        for i in 0..max_len {
            let a_dim = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
            let b_dim = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

            if a_dim == b_dim {
                result[max_len - 1 - i] = a_dim;
            } else if a_dim == 1 {
                result[max_len - 1 - i] = b_dim;
            } else if b_dim == 1 {
                result[max_len - 1 - i] = a_dim;
            } else {
                return None; // Incompatible shapes
            }
        }

        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![0u8; 4096];
        let tensor = GpuTensor::new_cpu(vec![32, 32], DType::F32, data);
        assert_eq!(tensor.numel(), 1024);
        assert_eq!(tensor.size_bytes(), 4096);
        assert!(!tensor.is_cuda());
    }

    #[test]
    fn test_strides() {
        let strides = shape::compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_broadcast() {
        let result = shape::broadcast_shapes(&[3, 1], &[1, 4]);
        assert_eq!(result, Some(vec![3, 4]));

        let result = shape::broadcast_shapes(&[2, 3, 4], &[4]);
        assert_eq!(result, Some(vec![2, 3, 4]));

        let result = shape::broadcast_shapes(&[2, 3], &[3, 2]);
        assert_eq!(result, None);
    }
}
