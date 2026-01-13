# Multi-GPU V3-OT Inference Status

## Summary

This document describes the work done on multi-GPU inference for the V3-OT (Oblivious Transfer) privacy-preserving inference endpoint on the Qwen 14B model.

## What Was Accomplished

### 1. Multi-GPU Weight Distribution (Working)
- The `GpuSecureModelWeights` initialization now distributes layer weights across multiple GPUs using round-robin: layer `i` goes to GPU `i % num_gpus`
- For 14B (48 layers) on 2 GPUs: GPU 0 gets layers 0,2,4,...,46; GPU 1 gets layers 1,3,5,...,47
- GPU memory distribution confirmed working: GPU 0: ~68.4 GB, GPU 1: ~65.4 GB

### 2. Multi-GPU Kernel Context Initialization (Working)
- Changed from single `KernelContext` to `Vec<KernelContext>` (one per GPU)
- Each kernel context is properly initialized with its own cuBLAS handle

### 3. Multi-GPU Inference Code in `batched_prefill_gpu_v3` (Implemented)
Changes made to `crates/v2/server/src/routes/secure_inference.rs`:
- Use devices from kernel contexts instead of creating new ones
- Track current GPU ID to know which GPU holds the hidden states
- When layer GPU differs from current GPU, transfer hidden states:
  1. Bind source device
  2. Download tensors to host
  3. Drop old tensors (while source device is bound for correct context)
  4. Bind target device
  5. Upload tensors to target device
- Pass correct kernel context and device to each layer operation

## The Problem

### CUDA Context Crash During Cleanup
When inference runs, the server crashes with:
```
CUDA_ERROR_ILLEGAL_ADDRESS, "an illegal memory access was encountered"
```
The crash occurs in `CudaSlice::drop` during tensor cleanup.

### Root Cause Analysis
The issue is related to CUDA context management in async Rust:

1. **Device Instance Mismatch**: The GPU secure weights (layer tensors) are created during initialization with one set of `GpuDevice` instances. The kernel contexts are created with a different set of `GpuDevice` instances. Even though they point to the same physical GPUs, they may have different CUDA context references.

2. **Async/Thread Issues**: Tokio's async runtime can move futures between threads. CUDA contexts are per-thread, and `bind_to_thread()` must be called to make a device current on a thread. When an async function is polled from different threads, device bindings may be lost.

3. **Drop Context**: When `CudaSlice::drop` is called, it tries to bind the device that was used to create the tensor, then free the memory. If the stored device reference doesn't properly set up the CUDA context (due to issues #1 or #2), the memory free fails.

### What We Tried
1. Using devices from kernel contexts instead of creating new ones - still crashes
2. Explicitly binding devices before and after tensor operations - still crashes
3. Synchronizing devices before and after transfers - still crashes
4. Downloading tensors to host, dropping old tensors while source device is bound, then uploading to target - still crashes

## Potential Solutions to Explore

1. **Use `spawn_blocking`**: Wrap the entire GPU computation in `tokio::task::spawn_blocking` to ensure all CUDA operations happen on a dedicated thread, avoiding the async/thread issue.

2. **Unified Device Management**: Store and reuse the same `GpuDevice` instances throughout the entire application lifecycle - during initialization, for kernel contexts, and during inference.

3. **Manual Memory Management**: Instead of relying on Rust's Drop trait, explicitly free CUDA memory with proper device binding before tensors go out of scope.

4. **cudarc Investigation**: The issue may be in how cudarc handles CUDA primary contexts when multiple `CudaDevice` instances are created for the same GPU.

## Files Modified

- `crates/v2/server/src/routes/secure_inference.rs`: Multi-GPU transfer logic in `batched_prefill_gpu_v3`
- `crates/v2/server/src/state.rs`: Changed to `Vec<KernelContext>` and multi-GPU initialization
- `crates/v2/core/src/gpu.rs`: Added `bind_to_thread()` method, improved `clone_to` for cross-device copy

## Environment
- RunPod with 2x H100 80GB GPUs
- Model: Qwen 2.5 14B Instruct (48 layers)
- P2P/NVLink topology shows OK for peer access between GPUs
