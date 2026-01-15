# ShardLM Benchmark Results - Qwen 2.5 7B

This document contains the official benchmark results for ShardLM secure inference implementations using the **Qwen 2.5 7B Instruct** model. 

## Benchmark Environment

### Hardware Configuration

#### GPU Specifications

| Component | Specification |
|-----------|---------------|
| GPU Model | NVIDIA H100 80GB HBM3 |
| GPU Architecture | Hopper |
| Compute Capability | 9.0 |
| GPU Memory (VRAM) | 81,559 MiB (80 GB HBM3) |
| Memory Bandwidth | 3.35 TB/s (theoretical) |
| Max Memory Clock | 2,619 MHz |
| Max SM Clock | 1,980 MHz |
| TDP / Power Limit | 700 W |
| PCIe Generation | Gen 5 |
| ECC | Enabled |
| MIG Mode | Disabled |
| Persistence Mode | Enabled |

#### CPU Specifications

| Component | Specification |
|-----------|---------------|
| CPU Model | Intel Xeon Platinum 8468 |
| Architecture | x86_64 (Sapphire Rapids) |
| Sockets | 2 |
| Cores per Socket | 40 |
| Threads per Core | 2 (Hyper-Threading) |
| Total CPUs (Threads) | 160 |
| NUMA Nodes | 2 |

#### System Memory

| Component | Specification |
|-----------|---------------|
| Total RAM | 1.5 TiB |
| Available RAM | ~1.4 TiB |

### Software Configuration

| Component | Value |
|-----------|-------|
| Operating System | Ubuntu 22.04.5 LTS (Jammy Jellyfish) |
| Kernel | 6.8.0-56-generic (x86_64) |
| NVIDIA Driver | 570.195.03 |
| CUDA Version | 12.8 |
| GSP Firmware | 570.195.03 |
| Platform | RunPod Cloud GPU |

### Model Configuration

| Component | Value |
|-----------|-------|
| Model | Qwen 2.5 7B Instruct |
| Model Architecture | `Qwen2_5_7B` |
| Model Parameters | 7 Billion |
| Model Layers | 28 |
| Hidden Dimension | 3,584 |
| GPU Memory Usage | ~15.8 GB |
| Benchmark Date | 2026-01-14 |
| Benchmark Run ID | `20260114_231827_7b` |

### Benchmark Parameters

| Parameter | Value |
|-----------|-------|
| Prompt | `"Hello, how are you?"` |
| Prompt Tokens | 19 |
| Max New Tokens | 50 |
| Temperature | 0.7 |
| Benchmark Runs | 20 |
| Warmup Runs | 3 |

## Protocol Versions

ShardLM implements five secure inference protocols:

| Version | Description | API Endpoint |
|---------|-------------|--------------|
| **V2** | Baseline secure inference with secret sharing | `/v2/secure/gpu/prefill_v2` |
| **V3** | Optimized secure inference (baseline) | `/v2/secure/gpu/prefill_v3` |
| **V3-OT** | Oblivious Transfer-based secure inference | `/v3/ot/prefill` |
| **V3-MPC** | Multi-Party Computation secure inference | `/v3/mpc/prefill` |
| **V3-CC** | Confidential Computing secure inference | `/v3/cc/prefill` |

## Summary Results

### Throughput Comparison

| Version | Throughput (tokens/sec) | Relative to V3 Baseline |
|---------|------------------------|-------------------------|
| **V3** | **26.97** | 1.00x (baseline) |
| V3-OT | 24.20 | 0.90x |
| V3-CC | 23.04 | 0.85x |
| V2 | 21.30 | 0.79x |
| V3-MPC | 20.09 | 0.74x |

### End-to-End Latency Comparison

| Version | Mean (ms) | Std (ms) | P95 (ms) | Overhead vs V3 |
|---------|-----------|----------|----------|----------------|
| **V3** | **722.17** | 31.48 | 761.62 | — |
| V3-OT | 801.59 | 36.46 | 873.97 | +11.0% |
| V3-CC | 840.32 | 32.96 | 886.82 | +16.4% |
| V2 | 909.46 | 57.26 | 987.05 | +25.9% |
| V3-MPC | 964.20 | 67.45 | 1080.19 | +33.5% |

## Detailed Results by Version

### V3 (Optimized Baseline)

The V3 protocol serves as the performance baseline, providing optimized secure inference with minimal overhead.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 17.78 ms |
| Std | 2.23 ms |
| Min | 13.98 ms |
| Max | 23.06 ms |
| P50 | 18.09 ms |
| P95 | 20.05 ms |
| P99 | 23.06 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 704.39 ms |
| Std | 30.66 ms |
| Min | 652.53 ms |
| Max | 746.55 ms |
| P50 | 718.65 ms |
| P95 | 738.56 ms |
| P99 | 746.55 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 722.17 ms |
| Std | 31.48 ms |
| Min | 667.74 ms |
| Max | 766.49 ms |
| P50 | 737.34 ms |
| P95 | 761.62 ms |
| P99 | 766.49 ms |
| **Tokens/sec** | **26.97** |

---

### V2 (Baseline Secure Inference)

The V2 protocol provides the original secure inference implementation with secret sharing.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 17.54 ms |
| Std | 1.90 ms |
| Min | 14.00 ms |
| Max | 21.00 ms |
| P50 | 17.39 ms |
| P95 | 20.34 ms |
| P99 | 21.00 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 891.92 ms |
| Std | 56.08 ms |
| Min | 827.83 ms |
| Max | 975.51 ms |
| P50 | 900.85 ms |
| P95 | 967.71 ms |
| P99 | 975.51 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 909.46 ms |
| Std | 57.26 ms |
| Min | 844.73 ms |
| Max | 995.86 ms |
| P50 | 920.25 ms |
| P95 | 987.05 ms |
| P99 | 995.86 ms |
| **Tokens/sec** | **21.30** |

---

### V3-OT (Oblivious Transfer)

The V3-OT protocol uses Oblivious Transfer for secure token embedding retrieval, preventing the server from learning which tokens the client requested.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 16.31 ms |
| Std | 2.26 ms |
| Min | 12.51 ms |
| Max | 22.09 ms |
| P50 | 16.11 ms |
| P95 | 20.04 ms |
| P99 | 22.09 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 785.28 ms |
| Std | 34.96 ms |
| Min | 753.74 ms |
| Max | 862.35 ms |
| P50 | 768.75 ms |
| P95 | 852.03 ms |
| P99 | 862.35 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 801.59 ms |
| Std | 36.46 ms |
| Min | 769.98 ms |
| Max | 882.39 ms |
| P50 | 784.17 ms |
| P95 | 873.97 ms |
| P99 | 882.39 ms |
| **Tokens/sec** | **24.20** |

---

### V3-MPC (Multi-Party Computation)

The V3-MPC protocol provides the strongest security guarantees using Multi-Party Computation with Beaver triples for secure multiplication.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 18.36 ms |
| Std | 2.20 ms |
| Min | 14.11 ms |
| Max | 22.36 ms |
| P50 | 19.26 ms |
| P95 | 20.63 ms |
| P99 | 22.36 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 945.84 ms |
| Std | 67.30 ms |
| Min | 883.68 ms |
| Max | 1062.50 ms |
| P50 | 911.01 ms |
| P95 | 1057.83 ms |
| P99 | 1062.50 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 964.20 ms |
| Std | 67.45 ms |
| Min | 898.15 ms |
| Max | 1082.43 ms |
| P50 | 929.42 ms |
| P95 | 1080.19 ms |
| P99 | 1082.43 ms |
| **Tokens/sec** | **20.09** |

---

### V3-CC (Confidential Computing)

The V3-CC protocol leverages hardware-based Confidential Computing for memory encryption. On H100 GPUs without CPU-side TDX support, software AES-256-GCM encryption is used as a fallback.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 15.84 ms |
| Std | 2.90 ms |
| Min | 11.45 ms |
| Max | 22.35 ms |
| P50 | 15.12 ms |
| P95 | 19.83 ms |
| P99 | 22.35 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 824.48 ms |
| Std | 31.74 ms |
| Min | 772.65 ms |
| Max | 892.05 ms |
| P50 | 830.39 ms |
| P95 | 866.99 ms |
| P99 | 892.05 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 840.32 ms |
| Std | 32.96 ms |
| Min | 785.38 ms |
| Max | 910.93 ms |
| P50 | 846.16 ms |
| P95 | 886.82 ms |
| P99 | 910.93 ms |
| **Tokens/sec** | **23.04** |

---

## Raw Timing Data

### V3 (Baseline) - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 19.25 | 727.96 | 747.21 |
| 2 | 13.98 | 667.15 | 681.13 |
| 3 | 19.86 | 723.69 | 743.55 |
| 4 | 14.62 | 726.93 | 741.55 |
| 5 | 18.69 | 718.65 | 737.34 |
| 6 | 20.05 | 714.30 | 734.35 |
| 7 | 18.66 | 724.00 | 742.66 |
| 8 | 17.71 | 727.01 | 744.72 |
| 9 | 17.27 | 702.49 | 719.76 |
| 10 | 17.67 | 717.58 | 735.25 |
| 11 | 15.22 | 652.53 | 667.74 |
| 12 | 18.81 | 690.72 | 709.53 |
| 13 | 18.09 | 662.36 | 680.44 |
| 14 | 15.46 | 662.45 | 677.91 |
| 15 | 19.93 | 746.55 | 766.49 |
| 16 | 17.93 | 665.68 | 683.61 |
| 17 | 19.10 | 656.43 | 675.54 |
| 18 | 23.06 | 738.56 | 761.62 |
| 19 | 15.09 | 733.37 | 748.46 |
| 20 | 15.11 | 729.33 | 744.45 |

### V2 - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 18.49 | 963.38 | 981.87 |
| 2 | 21.00 | 946.16 | 967.16 |
| 3 | 17.39 | 927.90 | 945.29 |
| 4 | 15.97 | 830.11 | 846.08 |
| 5 | 16.66 | 842.56 | 859.22 |
| 6 | 16.26 | 835.27 | 851.53 |
| 7 | 16.90 | 827.83 | 844.73 |
| 8 | 16.25 | 842.97 | 859.22 |
| 9 | 18.73 | 946.15 | 964.89 |
| 10 | 18.48 | 829.72 | 848.20 |
| 11 | 19.93 | 932.69 | 952.63 |
| 12 | 16.76 | 841.02 | 857.78 |
| 13 | 18.54 | 954.13 | 972.67 |
| 14 | 19.34 | 967.71 | 987.05 |
| 15 | 19.39 | 900.85 | 920.25 |
| 16 | 20.34 | 975.51 | 995.86 |
| 17 | 14.22 | 944.03 | 958.25 |
| 18 | 16.70 | 849.60 | 866.30 |
| 19 | 15.48 | 837.89 | 853.37 |
| 20 | 14.00 | 842.88 | 856.88 |

### V3-OT - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 20.03 | 839.76 | 859.79 |
| 2 | 14.87 | 768.75 | 783.62 |
| 3 | 17.97 | 766.20 | 784.17 |
| 4 | 20.04 | 862.35 | 882.39 |
| 5 | 17.26 | 781.85 | 799.12 |
| 6 | 22.09 | 851.88 | 873.97 |
| 7 | 16.11 | 852.03 | 868.15 |
| 8 | 14.58 | 799.15 | 813.73 |
| 9 | 14.10 | 761.65 | 775.75 |
| 10 | 14.41 | 767.32 | 781.73 |
| 11 | 16.24 | 753.74 | 769.98 |
| 12 | 16.45 | 754.06 | 770.51 |
| 13 | 15.60 | 759.32 | 774.92 |
| 14 | 14.81 | 765.53 | 780.35 |
| 15 | 16.49 | 758.66 | 775.15 |
| 16 | 12.51 | 772.15 | 784.66 |
| 17 | 17.26 | 764.39 | 781.65 |
| 18 | 15.75 | 766.81 | 782.55 |
| 19 | 15.29 | 789.27 | 804.56 |
| 20 | 14.36 | 770.75 | 785.11 |

### V3-MPC - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 20.36 | 898.55 | 918.91 |
| 2 | 19.26 | 922.87 | 942.13 |
| 3 | 19.78 | 903.61 | 923.40 |
| 4 | 20.63 | 926.35 | 946.98 |
| 5 | 22.36 | 1057.83 | 1080.19 |
| 6 | 15.57 | 1051.92 | 1067.49 |
| 7 | 16.24 | 892.97 | 909.22 |
| 8 | 19.49 | 911.27 | 930.76 |
| 9 | 18.41 | 911.01 | 929.42 |
| 10 | 20.03 | 906.27 | 926.30 |
| 11 | 20.16 | 903.57 | 923.73 |
| 12 | 19.27 | 907.39 | 926.66 |
| 13 | 19.94 | 1062.50 | 1082.43 |
| 14 | 17.02 | 1056.81 | 1073.84 |
| 15 | 14.47 | 883.68 | 898.15 |
| 16 | 18.48 | 883.74 | 902.22 |
| 17 | 17.32 | 885.12 | 902.43 |
| 18 | 19.04 | 1052.56 | 1071.60 |
| 19 | 14.11 | 988.64 | 1002.76 |
| 20 | 15.36 | 910.09 | 925.45 |

### V3-CC - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 22.35 | 831.97 | 854.32 |
| 2 | 17.66 | 789.22 | 806.89 |
| 3 | 17.75 | 845.81 | 863.56 |
| 4 | 17.25 | 840.79 | 858.04 |
| 5 | 14.14 | 809.64 | 823.78 |
| 6 | 19.83 | 866.99 | 886.82 |
| 7 | 13.18 | 815.07 | 828.25 |
| 8 | 18.88 | 892.05 | 910.93 |
| 9 | 13.31 | 866.29 | 879.59 |
| 10 | 17.98 | 810.72 | 828.70 |
| 11 | 18.33 | 814.46 | 832.79 |
| 12 | 14.80 | 778.14 | 792.94 |
| 13 | 11.74 | 776.96 | 788.69 |
| 14 | 11.45 | 806.33 | 817.78 |
| 15 | 12.73 | 772.65 | 785.38 |
| 16 | 14.79 | 806.28 | 821.07 |
| 17 | 12.93 | 849.41 | 862.34 |
| 18 | 14.33 | 855.44 | 869.78 |
| 19 | 15.12 | 831.04 | 846.16 |
| 20 | 18.24 | 830.39 | 848.63 |

---

## Analysis

### Performance Ranking

1. **V3 (Baseline)** - Fastest at 26.97 tokens/sec
2. **V3-OT** - 10.3% slower, provides input privacy via Oblivious Transfer
3. **V3-CC** - 14.6% slower, provides memory encryption for confidentiality
4. **V2** - 21.0% slower, original secure sharing implementation
5. **V3-MPC** - 25.5% slower, strongest cryptographic guarantees

### Variance Analysis

| Version | Std Dev (ms) | Coefficient of Variation |
|---------|--------------|-------------------------|
| V3 | 31.48 | 4.4% |
| V3-CC | 32.96 | 3.9% |
| V3-OT | 36.46 | 4.5% |
| V2 | 57.26 | 6.3% |
| V3-MPC | 67.45 | 7.0% |

All versions show good consistency with coefficient of variation under 7%, with V3-CC showing the most stable performance.

### Security vs Performance Trade-offs

| Version | Security Level | Primary Protection | Performance Impact |
|---------|---------------|-------------------|-------------------|
| V3 | Moderate | Secret sharing | Baseline |
| V3-OT | High | Input privacy (OT) | +11.0% |
| V3-CC | High | Memory encryption | +16.4% |
| V2 | Moderate | Secret sharing | +25.9% |
| V3-MPC | Highest | Full MPC protocol | +33.5% |

### Comparison with 1.5B Model

| Version | 1.5B (tok/s) | 7B (tok/s) | Slowdown Factor |
|---------|--------------|------------|-----------------|
| V3 | 54.82 | 26.97 | 2.03x |
| V3-OT | 41.49 | 24.20 | 1.71x |
| V3-CC | 37.80 | 23.04 | 1.64x |
| V2 | 38.58 | 21.30 | 1.81x |
| V3-MPC | 32.63 | 20.09 | 1.62x |

The 7B model (4.7x more parameters than 1.5B) shows a slowdown of 1.6-2.0x, demonstrating good scaling efficiency.

---

## Reproducibility

These benchmarks can be reproduced using:

```bash
# Set environment for 7B model
export SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-7b-instruct-weights
export SHARDLM_V2_MODEL_ARCH=qwen2_5_7b
export BENCHMARK_ITERATIONS=20
export BENCHMARK_MAX_TOKENS=50

# Run all benchmarks
./scripts/run_benchmarks.sh --v2 --v3 --v3-ot --v3-mpc --v3-cc
```

Or run individual benchmarks:

```bash
./target/release/shardlm-v2-client benchmark \
    -s http://localhost:9090 \
    -e v3 \
    -r 20 \
    -m 50 \
    -w 3 \
    -o results.json \
    --raw
```

---

## Data Files

The raw benchmark data is available in JSON format:

- `benchmarks_results/20260114_231827_7b/v2_benchmark.json`
- `benchmarks_results/20260114_231827_7b/v3_benchmark.json`
- `benchmarks_results/20260114_231827_7b/v3-ot_benchmark.json`
- `benchmarks_results/20260114_231827_7b/v3-mpc_benchmark.json`
- `benchmarks_results/20260114_231827_7b/v3-cc_benchmark.json`

System information: `benchmarks_results/20260114_231827_7b/system_info.txt`

---

## Notes

1. **All implementations are production-ready** with no mocks, placeholders, or simulations.

2. **V3-CC uses software AES-256-GCM** because the benchmark environment (RunPod) does not expose Intel TDX to the VM. On a true Confidential VM with TDX enabled, hardware AES-256-XTS encryption would be used with potentially different performance characteristics.

3. **GPU memory usage** for the 7B model is approximately 15.8 GB, leaving ample headroom on the H100's 80 GB HBM3 for larger batch sizes or longer sequences.

4. **Server initialization time** is longer for the 7B model (~5-6 minutes) due to secure weight processing for 28 layers with 3584 hidden dimensions.

5. **Tokens/sec calculation** is based on the 19 prompt tokens processed during the prefill phase. Decode phase was not measured in these benchmarks.

6. **Warmup runs** (3 iterations) are excluded from all statistics to ensure measurements reflect steady-state performance.

7. **Temperature setting** (0.7) does not affect prefill performance but is included for completeness.
