//! GPU memory management and CUDA operations
//!
//! This module provides GPU memory management using cudarc.
//!
//! Memory layout:
//! - Weights: Stored as BF16 (u16) on GPU to save memory (~70GB for Llama 70B)
//! - Activations: Stored as FP32 on GPU for precision during computation
//! - Matmul: Converts BF16 weights to FP32 on-the-fly during computation

#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::error::{Result, V2Error};
use crate::tensor::DType;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr};

/// GPU device wrapper
#[derive(Clone)]
pub struct GpuDevice {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    #[cfg(not(feature = "cuda"))]
    _device_id: usize,
    pub device_id: usize,
}

impl GpuDevice {
    /// Create a new GPU device
    #[cfg(feature = "cuda")]
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| V2Error::Gpu(format!("Failed to create CUDA device {}: {:?}", device_id, e)))?;
        Ok(Self {
            device,
            device_id,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            _device_id: device_id,
            device_id,
        })
    }

    /// Get total memory in bytes
    #[cfg(feature = "cuda")]
    pub fn total_memory(&self) -> Result<usize> {
        Ok(24 * 1024 * 1024 * 1024) // Placeholder: 24GB for A10G
    }

    #[cfg(not(feature = "cuda"))]
    pub fn total_memory(&self) -> Result<usize> {
        Ok(0)
    }

    /// Get free memory in bytes
    #[cfg(feature = "cuda")]
    pub fn free_memory(&self) -> Result<usize> {
        Ok(24 * 1024 * 1024 * 1024) // Placeholder
    }

    #[cfg(not(feature = "cuda"))]
    pub fn free_memory(&self) -> Result<usize> {
        Ok(0)
    }

    /// Synchronize device
    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| V2Error::Gpu(format!("Failed to synchronize: {:?}", e)))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    /// Get the underlying CudaDevice (for advanced operations)
    #[cfg(feature = "cuda")]
    pub fn cuda_device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Bind this device to the current thread's CUDA context
    /// This is needed when switching between multiple GPUs
    #[cfg(feature = "cuda")]
    pub fn bind_to_thread(&self) -> Result<()> {
        self.device
            .bind_to_thread()
            .map_err(|e| V2Error::Gpu(format!("Failed to bind device to thread: {:?}", e)))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn bind_to_thread(&self) -> Result<()> {
        Ok(())
    }

    /// Get memory info (used, total) in bytes
    #[cfg(feature = "cuda")]
    pub fn memory_info(&self) -> (usize, usize) {
        match cudarc::driver::result::mem_get_info() {
            Ok((free, total)) => (total - free, total),
            Err(_) => (0, 24 * 1024 * 1024 * 1024),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn memory_info(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Allocate FP32 tensor on GPU
    #[cfg(feature = "cuda")]
    pub fn alloc_f32(&self, numel: usize) -> Result<CudaSlice<f32>> {
        self.device
            .alloc_zeros::<f32>(numel)
            .map_err(|e| V2Error::Gpu(format!("Failed to allocate GPU memory: {:?}", e)))
    }

    /// Copy FP32 data to GPU
    #[cfg(feature = "cuda")]
    pub fn htod_f32(&self, data: Vec<f32>) -> Result<CudaSlice<f32>> {
        self.device
            .htod_copy(data)
            .map_err(|e| V2Error::Gpu(format!("Failed to copy to GPU: {:?}", e)))
    }

    /// Copy FP32 data from GPU
    #[cfg(feature = "cuda")]
    pub fn dtoh_f32(&self, slice: &CudaSlice<f32>) -> Result<Vec<f32>> {
        self.device
            .dtoh_sync_copy(slice)
            .map_err(|e| V2Error::Gpu(format!("Failed to copy from GPU: {:?}", e)))
    }

    /// Generic device-to-host copy
    #[cfg(feature = "cuda")]
    pub fn dtoh<T: DeviceRepr + Clone + Default>(&self, slice: &CudaSlice<T>) -> Result<Vec<T>> {
        self.device
            .dtoh_sync_copy(slice)
            .map_err(|e| V2Error::Gpu(format!("Failed to copy from GPU: {:?}", e)))
    }

    /// Generic host-to-device copy
    #[cfg(feature = "cuda")]
    pub fn htod<T: DeviceRepr + Clone + Unpin>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        self.device
            .htod_copy(data.to_vec())
            .map_err(|e| V2Error::Gpu(format!("Failed to copy to GPU: {:?}", e)))
    }

    /// Copy i32 data to GPU (for positions array in RoPE)
    #[cfg(feature = "cuda")]
    pub fn htod_i32(&self, data: Vec<i32>) -> Result<CudaSlice<i32>> {
        self.device
            .htod_copy(data)
            .map_err(|e| V2Error::Gpu(format!("Failed to copy i32 to GPU: {:?}", e)))
    }
}

impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("device_id", &self.device_id)
            .finish()
    }
}

/// Multi-GPU context for tensor parallelism
pub struct MultiGpuContext {
    pub devices: Vec<GpuDevice>,
    pub num_gpus: usize,
}

impl MultiGpuContext {
    pub fn new(num_gpus: usize) -> Result<Self> {
        let mut devices = Vec::with_capacity(num_gpus);
        for i in 0..num_gpus {
            devices.push(GpuDevice::new(i)?);
        }
        Ok(Self { devices, num_gpus })
    }

    pub fn device(&self, idx: usize) -> &GpuDevice {
        assert!(
            idx < self.devices.len(),
            "GPU device index {} out of bounds (num_devices={}, num_gpus={})",
            idx, self.devices.len(), self.num_gpus
        );
        &self.devices[idx]
    }

    pub fn total_memory(&self) -> Result<usize> {
        let mut total = 0;
        for device in &self.devices {
            total += device.total_memory()?;
        }
        Ok(total)
    }

    pub fn synchronize_all(&self) -> Result<()> {
        for device in &self.devices {
            device.synchronize()?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for MultiGpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiGpuContext")
            .field("num_gpus", &self.num_gpus)
            .finish()
    }
}

/// GPU tensor storing FP32 data directly on device
/// This avoids all CPU<->GPU transfers during inference
#[cfg(feature = "cuda")]
pub struct CudaTensor {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Original dtype (for reference, but data is always FP32 on GPU)
    pub dtype: DType,
    /// FP32 data on GPU
    data: CudaSlice<f32>,
    /// Device ID
    pub device_id: usize,
}

#[cfg(not(feature = "cuda"))]
pub struct CudaTensor {
    pub shape: Vec<usize>,
    pub dtype: DType,
    data: Vec<f32>,
    pub device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaTensor {
    /// Create from FP32 data on host (uploads once, stays on GPU)
    pub fn from_f32(device: &GpuDevice, shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected = shape.iter().product::<usize>();
        if data.len() != expected {
            return Err(V2Error::Gpu(format!(
                "Data size mismatch: expected {}, got {}",
                expected, data.len()
            )));
        }
        let gpu_data = device.htod_f32(data)?;
        Ok(Self {
            shape,
            dtype: DType::F32,
            data: gpu_data,
            device_id: device.device_id,
        })
    }

    /// Create from BF16 bytes (converts to FP32 and uploads)
    pub fn from_bf16_bytes(device: &GpuDevice, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        use half::bf16;
        let f32_data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        Self::from_f32(device, shape, f32_data)
    }

    /// Create from INT8 bytes with per-channel scales (dequantizes to FP32)
    pub fn from_int8_with_scales(
        device: &GpuDevice,
        shape: Vec<usize>,
        int8_bytes: &[u8],
        scales: &[f32],
    ) -> Result<Self> {
        let rows = shape[0];
        let cols = shape[1];
        let mut f32_data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            let scale = scales[row];
            let row_start = row * cols;
            for col in 0..cols {
                let val = (int8_bytes[row_start + col] as i8) as f32 * scale;
                f32_data.push(val);
            }
        }

        Self::from_f32(device, shape, f32_data)
    }

    /// Create uninitialized tensor (for output buffers)
    pub fn zeros(device: &GpuDevice, shape: Vec<usize>) -> Result<Self> {
        let numel = shape.iter().product();
        let gpu_data = device.alloc_f32(numel)?;
        Ok(Self {
            shape,
            dtype: DType::F32,
            data: gpu_data,
            device_id: device.device_id,
        })
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.numel() * 4 // Always FP32
    }

    /// Get the underlying GPU data (for kernel operations)
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Get mutable reference to GPU data
    pub fn data_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.data
    }

    /// Copy to host as FP32
    pub fn to_f32_host(&self, device: &GpuDevice) -> Result<Vec<f32>> {
        device.dtoh_f32(&self.data)
    }

    /// Copy to host as BF16 bytes (for compatibility)
    pub fn to_host(&self, device: &GpuDevice) -> Result<Vec<u8>> {
        use half::bf16;
        let f32_data = self.to_f32_host(device)?;
        let bf16_bytes: Vec<u8> = f32_data
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        Ok(bf16_bytes)
    }

    /// Create a view with different shape (no copy, just reshape)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<CudaTensorView> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(V2Error::Gpu(format!(
                "Cannot reshape {} elements to {:?}",
                self.numel(), new_shape
            )));
        }
        Ok(CudaTensorView {
            shape: new_shape,
            data: &self.data,
            device_id: self.device_id,
        })
    }

    /// Reshape in-place (just changes metadata, no data copy)
    /// Returns the same tensor with updated shape - use for perf-critical paths
    pub fn reshape_inplace(mut self, new_shape: Vec<usize>) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(V2Error::Gpu(format!(
                "Cannot reshape {} elements to {:?}",
                self.numel(), new_shape
            )));
        }
        self.shape = new_shape;
        Ok(self)
    }

    /// Clone tensor to a (potentially different) device - creates a new GPU allocation
    /// Used when we need to modify data without affecting the original
    pub fn clone_to(&self, target_device: &GpuDevice) -> Result<Self> {
        // For cross-device copy, we need to:
        // 1. Bind source device and synchronize
        // 2. Download data to host
        // 3. Bind target device
        // 4. Upload data to target
        let source_device = GpuDevice::new(self.device_id)?;
        source_device.bind_to_thread()?;
        source_device.synchronize()?;
        let host_data = source_device.dtoh_f32(&self.data)?;

        target_device.bind_to_thread()?;
        Self::from_f32(target_device, self.shape.clone(), host_data)
    }

    /// Clone tensor within the same device using GPU-to-GPU copy (fast, no CPU round-trip!)
    /// Use this when you need a copy for mutation but don't need cross-device transfer
    /// If tensor is on a different GPU, falls back to clone_to which goes through host memory
    pub fn clone_on_device(&self, device: &GpuDevice) -> Result<Self> {
        // If tensor is on a different GPU, use clone_to which goes through host memory
        if self.device_id != device.device_id {
            return self.clone_to(device);
        }

        // Same device - use fast GPU-to-GPU copy
        let n = self.numel();
        let mut new_data = device.alloc_f32(n)?;

        unsafe {
            device.cuda_device().dtod_copy(&self.data, &mut new_data)
                .map_err(|e| V2Error::Gpu(format!("dtod_copy failed: {:?}", e)))?;
        }

        Ok(Self {
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            data: new_data,
            device_id: self.device_id,
        })
    }

    /// Get a reference to underlying data for kernel operations that need CudaSlice directly
    pub fn data_slice(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Zero out all GPU memory (for cache clearing to prevent stale data issues)
    #[cfg(feature = "cuda")]
    pub fn zero(&mut self, device: &GpuDevice) -> Result<()> {
        // Re-allocate with zeros (alloc_f32 uses alloc_zeros internally)
        let numel = self.numel();
        let new_data = device.alloc_f32(numel)?;
        self.data = new_data;
        Ok(())
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaTensor {
    pub fn from_f32(_device: &GpuDevice, shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        Ok(Self {
            shape,
            dtype: DType::F32,
            data,
            device_id: 0,
        })
    }

    pub fn from_bf16_bytes(_device: &GpuDevice, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        use half::bf16;
        let f32_data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        Ok(Self {
            shape,
            dtype: DType::F32,
            data: f32_data,
            device_id: 0,
        })
    }

    pub fn from_int8_with_scales(
        _device: &GpuDevice,
        shape: Vec<usize>,
        int8_bytes: &[u8],
        scales: &[f32],
    ) -> Result<Self> {
        let rows = shape[0];
        let cols = shape[1];
        let mut f32_data = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let scale = scales[row];
            let row_start = row * cols;
            for col in 0..cols {
                let val = (int8_bytes[row_start + col] as i8) as f32 * scale;
                f32_data.push(val);
            }
        }
        Ok(Self {
            shape,
            dtype: DType::F32,
            data: f32_data,
            device_id: 0,
        })
    }

    pub fn zeros(_device: &GpuDevice, shape: Vec<usize>) -> Result<Self> {
        let numel = shape.iter().product();
        Ok(Self {
            shape,
            dtype: DType::F32,
            data: vec![0.0; numel],
            device_id: 0,
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * 4
    }

    pub fn to_f32_host(&self, _device: &GpuDevice) -> Result<Vec<f32>> {
        Ok(self.data.clone())
    }

    pub fn to_host(&self, _device: &GpuDevice) -> Result<Vec<u8>> {
        use half::bf16;
        let bf16_bytes: Vec<u8> = self.data
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_le_bytes())
            .collect();
        Ok(bf16_bytes)
    }

    /// Clone tensor (non-CUDA version)
    pub fn clone_to(&self, _device: &GpuDevice) -> Result<Self> {
        Ok(Self {
            shape: self.shape.clone(),
            dtype: self.dtype.clone(),
            data: self.data.clone(),
            device_id: self.device_id,
        })
    }

    /// Reshape in-place (just changes metadata, no data copy)
    pub fn reshape_inplace(mut self, new_shape: Vec<usize>) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(V2Error::Gpu(format!(
                "Cannot reshape {} elements to {:?}",
                self.numel(), new_shape
            )));
        }
        self.shape = new_shape;
        Ok(self)
    }
}

/// View into a CudaTensor (for reshaping without copying)
#[cfg(feature = "cuda")]
pub struct CudaTensorView<'a> {
    pub shape: Vec<usize>,
    pub data: &'a CudaSlice<f32>,
    pub device_id: usize,
}

// ============================================================================
// BF16 Weight Tensor - stores weights compressed on GPU
// ============================================================================

/// GPU tensor storing BF16 weights (half memory vs FP32)
/// Used for model weights to fit 70B models in 96GB GPU memory
#[cfg(feature = "cuda")]
pub struct CudaTensorBF16 {
    pub shape: Vec<usize>,
    /// BF16 data stored as u16 on GPU
    data: CudaSlice<u16>,
    pub device_id: usize,
}

#[cfg(not(feature = "cuda"))]
pub struct CudaTensorBF16 {
    pub shape: Vec<usize>,
    data: Vec<u16>,
    pub device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaTensorBF16 {
    /// Create from BF16 bytes (uploads directly as u16)
    pub fn from_bf16_bytes(device: &GpuDevice, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        let u16_data: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        let expected = shape.iter().product::<usize>();
        if u16_data.len() != expected {
            return Err(V2Error::Gpu(format!(
                "BF16 data size mismatch: expected {}, got {}",
                expected, u16_data.len()
            )));
        }

        let gpu_data = device.htod(&u16_data)?;
        Ok(Self {
            shape,
            data: gpu_data,
            device_id: device.device_id,
        })
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get size in bytes (BF16 = 2 bytes per element)
    pub fn size_bytes(&self) -> usize {
        self.numel() * 2
    }

    /// Get the underlying GPU data as u16
    pub fn data(&self) -> &CudaSlice<u16> {
        &self.data
    }

    /// Convert to FP32 CudaTensor (allocates new memory)
    /// This is used during matmul - converts weight just before computation
    pub fn to_f32(&self, device: &GpuDevice) -> Result<CudaTensor> {
        use half::bf16;
        // Download BF16, convert to FP32, re-upload
        // TODO: Replace with GPU kernel for better performance
        let u16_data = device.dtoh(&self.data)?;
        let f32_data: Vec<f32> = u16_data
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect();
        CudaTensor::from_f32(device, self.shape.clone(), f32_data)
    }

    /// Copy to host as BF16 bytes
    pub fn to_host(&self, device: &GpuDevice) -> Result<Vec<u8>> {
        let u16_data = device.dtoh(&self.data)?;
        let bytes: Vec<u8> = u16_data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        Ok(bytes)
    }

    /// Clone this tensor (copies GPU data)
    /// Used for tied embeddings where lm_head shares embed_tokens
    pub fn clone_on_device(&self, device: &GpuDevice) -> Result<Self> {
        let u16_data = device.dtoh(&self.data)?;
        let gpu_data = device.htod(&u16_data)?;
        Ok(Self {
            shape: self.shape.clone(),
            data: gpu_data,
            device_id: self.device_id,
        })
    }
}

#[cfg(not(feature = "cuda"))]
impl CudaTensorBF16 {
    pub fn from_bf16_bytes(_device: &GpuDevice, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        let u16_data: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Self {
            shape,
            data: u16_data,
            device_id: 0,
        })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.numel() * 2
    }

    pub fn to_f32(&self, device: &GpuDevice) -> Result<CudaTensor> {
        use half::bf16;
        let f32_data: Vec<f32> = self.data
            .iter()
            .map(|&bits| bf16::from_bits(bits).to_f32())
            .collect();
        CudaTensor::from_f32(device, self.shape.clone(), f32_data)
    }

    pub fn to_host(&self, _device: &GpuDevice) -> Result<Vec<u8>> {
        let bytes: Vec<u8> = self.data
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        Ok(bytes)
    }

    /// Clone this tensor
    pub fn clone_on_device(&self, _device: &GpuDevice) -> Result<Self> {
        Ok(Self {
            shape: self.shape.clone(),
            data: self.data.clone(),
            device_id: self.device_id,
        })
    }
}

impl std::fmt::Debug for CudaTensorBF16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensorBF16")
            .field("shape", &self.shape)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device_id", &self.device_id)
            .finish()
    }
}

// ============================================================================
// Legacy compatibility layer (to be removed after migration)
// ============================================================================

/// GPU buffer wrapper for raw memory (legacy)
pub struct GpuBuffer {
    #[cfg(feature = "cuda")]
    slice: CudaSlice<u8>,
    #[cfg(not(feature = "cuda"))]
    data: Vec<u8>,
    pub device_id: usize,
    pub size_bytes: usize,
    pub dtype: DType,
}

impl GpuBuffer {
    #[cfg(feature = "cuda")]
    pub fn from_host(device: &GpuDevice, data: &[u8], dtype: DType) -> Result<Self> {
        let slice = device.cuda_device()
            .htod_copy(data.to_vec())
            .map_err(|e| V2Error::Gpu(format!("Failed to copy to device: {:?}", e)))?;
        Ok(Self {
            slice,
            device_id: device.device_id,
            size_bytes: data.len(),
            dtype,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from_host(_device: &GpuDevice, data: &[u8], dtype: DType) -> Result<Self> {
        Ok(Self {
            data: data.to_vec(),
            device_id: 0,
            size_bytes: data.len(),
            dtype,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn to_host(&self, device: &GpuDevice) -> Result<Vec<u8>> {
        device.cuda_device()
            .dtoh_sync_copy(&self.slice)
            .map_err(|e| V2Error::Gpu(format!("Failed to copy to host: {:?}", e)))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn to_host(&self, _device: &GpuDevice) -> Result<Vec<u8>> {
        Ok(self.data.clone())
    }

    pub fn size(&self) -> usize {
        self.size_bytes
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("device_id", &self.device_id)
            .field("size_bytes", &self.size_bytes)
            .field("dtype", &self.dtype)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_device_creation() {
        let device = GpuDevice::new(0);
        assert!(device.is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_tensor_f32() {
        let device = GpuDevice::new(0).unwrap();
        let shape = vec![32, 32];
        let data = vec![1.0f32; 32 * 32];
        let tensor = CudaTensor::from_f32(&device, shape, data);
        assert!(tensor.is_ok());
        let tensor = tensor.unwrap();
        assert_eq!(tensor.numel(), 1024);
    }
}
