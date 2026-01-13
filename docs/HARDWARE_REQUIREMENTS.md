# ShardLM Hardware Requirements for Large Model Benchmarking

## Executive Summary

This document summarizes our successful benchmarking of ShardLM variants on Qwen 2.5-1.5B and 7B models, and provides hardware requirements for scaling to 14B and 72B models.

**Current Status**: Successfully benchmarked 1.5B and 7B models on H100 80GB. Multi-GPU support for 14B is in progress but blocked by CUDA context management issues in async Rust (see Section 3.3).

---

## 1. Successful Benchmarks Completed

### 1.1 Models Benchmarked

| Model | Parameters | GPU Memory Used | Status |
|-------|------------|-----------------|--------|
| Qwen 2.5-1.5B | 1.5B | ~3.7 GB | ✓ Complete |
| Qwen 2.5-7B | 7B | ~15.8 GB | ✓ Complete |
| Qwen 2.5-14B | 14B | ~95 GB (needed) | ⚠ Multi-GPU in progress |

### 1.2 ShardLM Variants Implemented

| Variant | Description | Security Model |
|---------|-------------|----------------|
| **V2** | Baseline with secret sharing | Additive shares, plaintext reconstruction |
| **V3** | GPU-optimized V2 | Same as V2, CUDA acceleration |
| **V3-CC** | Confidential Computing | H100 hardware encryption + attestation |
| **V3-MPC** | Multi-Party Computation | Beaver triples, polynomial approximation |
| **V3-OT** | Oblivious Transfer | 1-of-N OT function lookup tables |

### 1.3 Benchmark Results Summary

#### Qwen 2.5-1.5B (All Variants)

| Variant | Prefill (ms) | Throughput | vs V2 |
|---------|-------------|------------|-------|
| V2 | 1407.7 | 39.8 tok/s | - |
| V3 | 1122.2 | 49.9 tok/s | +25.4% |
| V3-CC | 1115.3 | 50.2 tok/s | +26.2% |
| **V3-OT** | 1132.2 | 49.5 tok/s | +24.3% |
| V3-MPC | 1232.3 | 45.4 tok/s | +14.2% |

#### Qwen 2.5-7B (V3-OT Only)

| Metric | 1.5B | 7B | Scaling |
|--------|------|-----|---------|
| Prefill | 1206 ms | 2237 ms | 1.85x |
| Throughput | 46.4 tok/s | 25.0 tok/s | 0.54x |
| **V3-OT Tables** | **48 KB** | **48 KB** | **1.0x (constant)** |

---

## 2. Best Variant for Scaling: V3-OT

### 2.1 Why V3-OT Scales Best

V3-OT has **constant 48 KB overhead** regardless of model size:

| Model Size | V3-MPC Overhead | V3-OT Overhead | V3-OT Advantage |
|------------|-----------------|----------------|-----------------|
| 1.5B | 2.65 MB | 48 KB | 55x smaller |
| 7B | 6.1 MB | 48 KB | 127x smaller |
| 14B | ~10 MB | 48 KB | 213x smaller |
| 72B | ~30 MB | 48 KB | 640x smaller |

### 2.2 V3-OT Properties

- **Constant table memory**: 48 KB (SiLU + exp + rsqrt lookup tables)
- **High accuracy**: ~0.02% discretization error (vs ~0.5-2% for V3-MPC polynomial)
- **No H100 requirement**: Works on any CUDA GPU (unlike V3-CC)
- **Information-theoretic security**: Server cannot learn access patterns

---

## 3. 14B Multi-GPU Status

### 3.1 Memory Breakdown for Secure Inference

ShardLM secure inference requires multiple weight copies:

```
Memory Required = Model Weights + Secure Weights + GPU Secure Weights + KV Cache + Working Memory
```

| Component | 7B | 14B | 72B |
|-----------|-----|-----|-----|
| Model Weights (BF16) | ~14 GB | ~28 GB | ~144 GB |
| Secure Weights | ~14 GB | ~28 GB | ~144 GB |
| GPU Secure Weights | ~14 GB | ~28 GB | ~144 GB |
| KV Cache (8K context) | ~1 GB | ~2.5 GB | ~12 GB |
| Working Memory | ~2 GB | ~4 GB | ~10 GB |
| **Total** | **~45 GB** | **~90 GB** | **~454 GB** |

### 3.2 Single GPU Limitation

- **H100 80GB**: Successfully runs 7B (~45 GB total)
- **H100 80GB**: Cannot run 14B (~90 GB needed) - OOM

### 3.3 Multi-GPU Attempt (2× H100 80GB)

Multi-GPU support was implemented with round-robin weight distribution:

**What Works:**
- Weight distribution across GPUs (round-robin by layer)
- GPU memory allocation confirmed: GPU 0 ~68.4 GB, GPU 1 ~65.4 GB
- Kernel context initialization per GPU

**Current Blocker:**
- CUDA_ERROR_ILLEGAL_ADDRESS crash during tensor cleanup
- Root cause: CUDA context management in async Rust (cudarc library)
- The crash occurs when tensors created in one async task are dropped in another

**Potential Solutions (Not Yet Implemented):**
1. Use `spawn_blocking` for all CUDA operations
2. Unified device management with explicit context switching
3. Manual memory management instead of Drop-based cleanup

See [MULTI_GPU_INFERENCE_STATUS.md](MULTI_GPU_INFERENCE_STATUS.md) for detailed technical analysis.

---

## 4. Hardware Requirements for 14B and 72B

### 4.1 Qwen 2.5-14B Requirements

**Minimum Configuration:**

| Component | Requirement |
|-----------|-------------|
| **GPU** | 2× H100 80GB (160 GB total) OR 1× H100 NVL 94GB + optimization |
| **GPU Memory** | ≥120 GB recommended |
| **System RAM** | ≥256 GB |
| **Storage** | ≥100 GB NVMe SSD |
| **Interconnect** | NVLink for multi-GPU |

**Recommended RunPod Configuration:**
- **Pod Type**: 2× H100 80GB SXM5 with NVLink
- **Or**: 1× H200 141GB (when available)

### 4.2 Qwen 2.5-72B Requirements

**Minimum Configuration:**

| Component | Requirement |
|-----------|-------------|
| **GPU** | 8× H100 80GB (640 GB total) |
| **GPU Memory** | ≥500 GB recommended |
| **System RAM** | ≥1 TB |
| **Storage** | ≥500 GB NVMe SSD |
| **Interconnect** | NVLink + NVSwitch |

**Recommended RunPod Configuration:**
- **Pod Type**: 8× H100 80GB SXM5 cluster
- **Or**: 4× H200 141GB (when available)

### 4.3 Quick Reference Table

| Model | Min GPU Config | Recommended | Est. V3-OT Throughput |
|-------|---------------|-------------|----------------------|
| 1.5B | 1× A10 24GB | 1× H100 80GB | ~46 tok/s |
| 7B | 1× A100 40GB | 1× H100 80GB | ~25 tok/s |
| **14B** | **2× H100 80GB** | **2× H100 80GB** | ~15 tok/s (proj.) |
| **72B** | **8× H100 80GB** | **8× H100 80GB** | ~5 tok/s (proj.) |

---

## 5. Deployment Instructions

### 5.1 For 14B Benchmark

1. **Provision Machine**:
   ```
   RunPod: 2× NVIDIA H100 80GB SXM5
   RAM: 256 GB+
   Storage: 100 GB+ NVMe
   ```

2. **Clone Repository**:
   ```bash
   git clone <repo-url> /workspace/shardlm
   cd /workspace/shardlm
   ```

3. **Build with Multi-GPU Support**:
   ```bash
   cargo build --release --features "cuda,h100-cc,mpc-secure"
   ```

4. **Download 14B Model**:
   ```bash
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-14B-Instruct', local_dir='/workspace/qwen-2.5-14b-instruct')"
   ```

5. **Start Server with 2 GPUs**:
   ```bash
   SHARDLM_V2_MODEL_DIR=/workspace/qwen-2.5-14b-instruct \
   SHARDLM_V2_MODEL_ARCH=qwen2_5_14b \
   SHARDLM_V2_PORT=9095 \
   SHARDLM_V2_NUM_GPUS=2 \
   ./target/release/shardlm-v2-server
   ```

6. **Run V3-OT Benchmark**:
   ```bash
   ./target/release/shardlm-v2-client benchmark \
       -s http://localhost:9095 \
       -p "Explain the concept of machine learning in simple terms." \
       --runs 10 --warmup 2 \
       --endpoint v3-ot \
       --output benchmark_results/14b_model/v3-ot_14b_benchmark.json
   ```

### 5.2 For 72B Benchmark

1. **Provision Machine**:
   ```
   RunPod: 8× NVIDIA H100 80GB SXM5 (or equivalent cluster)
   RAM: 1 TB+
   Storage: 500 GB+ NVMe
   ```

2. **Start Server with 8 GPUs**:
   ```bash
   SHARDLM_V2_MODEL_DIR=/workspace/qwen-2.5-72b-instruct \
   SHARDLM_V2_MODEL_ARCH=qwen2_5_72b \
   SHARDLM_V2_PORT=9095 \
   SHARDLM_V2_NUM_GPUS=8 \
   ./target/release/shardlm-v2-server
   ```

---

## 6. Expected Scaling Results

Based on our 1.5B → 7B scaling analysis, projected results for larger models:

| Model | Prefill (proj.) | Throughput (proj.) | V3-OT Tables |
|-------|-----------------|-------------------|--------------|
| 1.5B | 1206 ms | 46.4 tok/s | 48 KB |
| 7B | 2237 ms | 25.0 tok/s | 48 KB |
| 14B | ~3500 ms | ~15 tok/s | **48 KB** |
| 72B | ~12000 ms | ~5 tok/s | **48 KB** |

**Key Validation Goal**: Confirm V3-OT's constant 48 KB overhead at 14B and 72B scales.

---

## 7. Files and Documentation Reference

| Document | Description |
|----------|-------------|
| `docs/V3_REFERENCE_IMPLEMENTATION.md` | Complete V3 technical documentation |
| `docs/V3_OT_BENCHMARK_REPORT.md` | V3-OT implementation and benchmarks |
| `docs/SCALING_ANALYSIS.md` | Comprehensive scaling analysis (1.5B to 72B) |
| `docs/V2_SECURITY_MODEL.md` | Security model and threat analysis |
| `docs/MULTI_GPU_INFERENCE_STATUS.md` | Multi-GPU implementation status and blockers |
| `benchmark_results/` | Raw benchmark JSON files |

---

## 8. Next Steps

1. **Resolve CUDA context issue** for multi-GPU support (see Section 3.3)
2. **Complete 14B benchmark** once multi-GPU is stable
3. **Validate constant 48 KB overhead** at 14B scale
4. **Document results** and update scaling projections
5. **(Future)** Deploy 8× H100 cluster for 72B benchmarking

