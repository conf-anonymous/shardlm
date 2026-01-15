# ShardLM Benchmark Results

This document contains the official benchmark results for ShardLM secure inference implementations. 

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
| VBIOS Version | 96.00.99.00.01 |
| GPU UUID | GPU-69d4c9cd-7c1a-5e4c-4bac-1be6283dd130 |
| Serial Number | 1651024013404 |
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
| Model | Qwen 2.5 1.5B Instruct |
| Model Architecture | `Qwen2_5_1_5B` |
| Model Parameters | 1.5 Billion |
| Benchmark Date | 2026-01-14 |
| Benchmark Run ID | `20260114_224009` |

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
| **V3** | **54.82** | 1.00x (baseline) |
| V3-OT | 41.49 | 0.76x |
| V2 | 38.58 | 0.70x |
| V3-CC | 37.80 | 0.69x |
| V3-MPC | 32.63 | 0.60x |

### End-to-End Latency Comparison

| Version | Mean (ms) | Std (ms) | P95 (ms) | Overhead vs V3 |
|---------|-----------|----------|----------|----------------|
| **V3** | **355.74** | 3.97 | 361.63 | — |
| V3-OT | 466.60 | 52.49 | 542.74 | +31.2% |
| V2 | 502.16 | 43.37 | 550.48 | +41.2% |
| V3-CC | 512.79 | 45.16 | 573.43 | +44.2% |
| V3-MPC | 592.53 | 58.59 | 642.44 | +66.6% |

## Detailed Results by Version

### V3 (Optimized Baseline)

The V3 protocol serves as the performance baseline, providing optimized secure inference with minimal overhead.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 9.17 ms |
| Std | 1.01 ms |
| Min | 7.07 ms |
| Max | 11.60 ms |
| P50 | 9.27 ms |
| P95 | 10.50 ms |
| P99 | 11.60 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 346.57 ms |
| Std | 3.22 ms |
| Min | 341.60 ms |
| Max | 354.37 ms |
| P50 | 346.77 ms |
| P95 | 351.26 ms |
| P99 | 354.37 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 355.74 ms |
| Std | 3.97 ms |
| Min | 350.29 ms |
| Max | 365.98 ms |
| P50 | 355.19 ms |
| P95 | 361.63 ms |
| P99 | 365.98 ms |
| **Tokens/sec** | **54.82** |

---

### V2 (Baseline Secure Inference)

The V2 protocol provides the original secure inference implementation with secret sharing.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 9.65 ms |
| Std | 1.37 ms |
| Min | 7.33 ms |
| Max | 12.09 ms |
| P50 | 9.58 ms |
| P95 | 11.68 ms |
| P99 | 12.09 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 492.51 ms |
| Std | 42.33 ms |
| Min | 444.13 ms |
| Max | 542.86 ms |
| P50 | 502.41 ms |
| P95 | 538.79 ms |
| P99 | 542.86 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 502.16 ms |
| Std | 43.37 ms |
| Min | 451.92 ms |
| Max | 553.97 ms |
| P50 | 513.16 ms |
| P95 | 550.48 ms |
| P99 | 553.97 ms |
| **Tokens/sec** | **38.58** |

---

### V3-OT (Oblivious Transfer)

The V3-OT protocol uses Oblivious Transfer for secure token embedding retrieval, preventing the server from learning which tokens the client requested.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 8.62 ms |
| Std | 0.97 ms |
| Min | 7.02 ms |
| Max | 10.50 ms |
| P50 | 8.50 ms |
| P95 | 10.04 ms |
| P99 | 10.50 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 457.98 ms |
| Std | 52.35 ms |
| Min | 374.27 ms |
| Max | 541.93 ms |
| P50 | 449.82 ms |
| P95 | 534.02 ms |
| P99 | 541.93 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 466.60 ms |
| Std | 52.49 ms |
| Min | 382.07 ms |
| Max | 550.93 ms |
| P50 | 459.16 ms |
| P95 | 542.74 ms |
| P99 | 550.93 ms |
| **Tokens/sec** | **41.49** |

---

### V3-MPC (Multi-Party Computation)

The V3-MPC protocol provides the strongest security guarantees using Multi-Party Computation with Beaver triples for secure multiplication.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 10.28 ms |
| Std | 1.27 ms |
| Min | 7.49 ms |
| Max | 12.32 ms |
| P50 | 10.26 ms |
| P95 | 12.21 ms |
| P99 | 12.32 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 582.25 ms |
| Std | 57.81 ms |
| Min | 484.12 ms |
| Max | 653.00 ms |
| P50 | 612.81 ms |
| P95 | 630.11 ms |
| P99 | 653.00 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 592.53 ms |
| Std | 58.59 ms |
| Min | 494.24 ms |
| Max | 664.83 ms |
| P50 | 624.18 ms |
| P95 | 642.44 ms |
| P99 | 664.83 ms |
| **Tokens/sec** | **32.63** |

---

### V3-CC (Confidential Computing)

The V3-CC protocol leverages hardware-based Confidential Computing for memory encryption. On H100 GPUs without CPU-side TDX support, software AES-256-GCM encryption is used as a fallback.

**Embedding Phase:**
| Metric | Value |
|--------|-------|
| Mean | 10.09 ms |
| Std | 2.44 ms |
| Min | 7.49 ms |
| Max | 19.16 ms |
| P50 | 9.95 ms |
| P95 | 12.53 ms |
| P99 | 19.16 ms |

**Prefill Phase (Prompt Processing):**
| Metric | Value |
|--------|-------|
| Mean | 502.71 ms |
| Std | 45.45 ms |
| Min | 432.72 ms |
| Max | 564.46 ms |
| P50 | 481.10 ms |
| P95 | 564.46 ms |
| P99 | 564.46 ms |

**Total (End-to-End):**
| Metric | Value |
|--------|-------|
| Mean | 512.79 ms |
| Std | 45.16 ms |
| Min | 442.88 ms |
| Max | 574.15 ms |
| P50 | 492.98 ms |
| P95 | 573.43 ms |
| P99 | 574.15 ms |
| **Tokens/sec** | **37.80** |

---

## Raw Timing Data

### V3 (Baseline) - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 8.92 | 346.06 | 354.97 |
| 2 | 9.69 | 349.16 | 358.85 |
| 3 | 10.37 | 351.26 | 361.63 |
| 4 | 9.77 | 347.24 | 357.00 |
| 5 | 7.56 | 347.64 | 355.19 |
| 6 | 9.48 | 350.42 | 359.90 |
| 7 | 9.63 | 344.62 | 354.25 |
| 8 | 8.58 | 343.37 | 351.95 |
| 9 | 11.60 | 354.37 | 365.98 |
| 10 | 7.07 | 343.93 | 351.00 |
| 11 | 8.15 | 342.14 | 350.29 |
| 12 | 9.54 | 349.71 | 359.24 |
| 13 | 8.94 | 341.60 | 350.54 |
| 14 | 9.19 | 348.15 | 357.34 |
| 15 | 8.78 | 343.30 | 352.08 |
| 16 | 8.17 | 344.43 | 352.60 |
| 17 | 9.27 | 343.87 | 353.14 |
| 18 | 10.50 | 346.77 | 357.27 |
| 19 | 9.55 | 347.40 | 356.95 |
| 20 | 8.56 | 346.00 | 354.57 |

### V2 - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 10.66 | 533.61 | 544.27 |
| 2 | 10.50 | 531.67 | 542.18 |
| 3 | 9.54 | 534.56 | 544.11 |
| 4 | 12.09 | 537.98 | 550.07 |
| 5 | 11.11 | 542.86 | 553.97 |
| 6 | 11.26 | 534.48 | 545.74 |
| 7 | 9.12 | 534.22 | 543.34 |
| 8 | 11.68 | 538.79 | 550.48 |
| 9 | 9.55 | 502.41 | 511.96 |
| 10 | 9.67 | 534.11 | 543.78 |
| 11 | 11.06 | 502.10 | 513.16 |
| 12 | 7.33 | 452.07 | 459.40 |
| 13 | 7.85 | 445.39 | 453.24 |
| 14 | 7.77 | 445.23 | 453.01 |
| 15 | 9.03 | 444.28 | 453.31 |
| 16 | 8.19 | 450.09 | 458.28 |
| 17 | 9.12 | 446.10 | 455.23 |
| 18 | 10.03 | 451.53 | 461.57 |
| 19 | 7.78 | 444.13 | 451.92 |
| 20 | 9.58 | 444.66 | 454.24 |

### V3-OT - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 8.50 | 527.54 | 536.04 |
| 2 | 9.00 | 541.93 | 550.93 |
| 3 | 8.52 | 450.64 | 459.16 |
| 4 | 7.69 | 528.42 | 536.11 |
| 5 | 8.25 | 436.24 | 444.49 |
| 6 | 7.93 | 533.65 | 541.59 |
| 7 | 10.50 | 449.82 | 460.31 |
| 8 | 9.71 | 435.26 | 444.97 |
| 9 | 8.72 | 534.02 | 542.74 |
| 10 | 9.95 | 519.66 | 529.62 |
| 11 | 8.19 | 434.54 | 442.73 |
| 12 | 7.99 | 441.40 | 449.38 |
| 13 | 7.84 | 428.83 | 436.67 |
| 14 | 7.80 | 374.27 | 382.07 |
| 15 | 7.31 | 388.61 | 395.92 |
| 16 | 7.02 | 407.99 | 415.01 |
| 17 | 9.40 | 404.50 | 413.90 |
| 18 | 10.04 | 406.17 | 416.21 |
| 19 | 9.87 | 459.32 | 469.20 |
| 20 | 8.10 | 456.86 | 464.96 |

### V3-MPC - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 10.68 | 612.81 | 623.49 |
| 2 | 9.83 | 602.77 | 612.59 |
| 3 | 12.21 | 624.06 | 636.27 |
| 4 | 9.36 | 611.26 | 620.63 |
| 5 | 8.71 | 616.68 | 625.39 |
| 6 | 9.60 | 605.28 | 614.88 |
| 7 | 11.99 | 612.52 | 624.50 |
| 8 | 11.09 | 623.15 | 634.24 |
| 9 | 11.83 | 653.00 | 664.83 |
| 10 | 10.83 | 538.65 | 549.48 |
| 11 | 11.64 | 625.83 | 637.47 |
| 12 | 9.14 | 493.22 | 502.36 |
| 13 | 10.26 | 615.76 | 626.02 |
| 14 | 10.12 | 484.12 | 494.24 |
| 15 | 10.01 | 617.39 | 627.41 |
| 16 | 7.49 | 489.78 | 497.26 |
| 17 | 10.27 | 613.91 | 624.18 |
| 18 | 8.72 | 489.97 | 498.70 |
| 19 | 12.32 | 630.11 | 642.44 |
| 20 | 9.56 | 484.70 | 494.26 |

### V3-CC - All 20 Runs

| Run | Embedding (ms) | Prefill (ms) | Total (ms) |
|-----|----------------|--------------|------------|
| 1 | 8.97 | 564.46 | 573.43 |
| 2 | 10.28 | 550.96 | 561.24 |
| 3 | 8.61 | 560.83 | 569.45 |
| 4 | 8.07 | 480.40 | 488.46 |
| 5 | 9.68 | 564.46 | 574.15 |
| 6 | 10.49 | 475.12 | 485.61 |
| 7 | 9.34 | 472.68 | 482.01 |
| 8 | 7.49 | 563.29 | 570.79 |
| 9 | 11.60 | 557.24 | 568.84 |
| 10 | 12.53 | 480.45 | 492.98 |
| 11 | 10.08 | 552.99 | 563.07 |
| 12 | 8.51 | 530.76 | 539.28 |
| 13 | 10.15 | 432.72 | 442.88 |
| 14 | 9.95 | 463.84 | 473.78 |
| 15 | 10.00 | 462.07 | 472.07 |
| 16 | 7.78 | 481.10 | 488.88 |
| 17 | 9.28 | 465.57 | 474.85 |
| 18 | 19.16 | 482.08 | 501.24 |
| 19 | 8.40 | 437.02 | 445.43 |
| 20 | 11.40 | 476.05 | 487.45 |

---

## Analysis

### Performance Ranking

1. **V3 (Baseline)** - Fastest at 54.82 tokens/sec with remarkably low variance (σ = 3.97 ms)
2. **V3-OT** - 31.2% overhead, best balance of security and performance for input privacy
3. **V2** - 41.2% overhead, original secure sharing implementation
4. **V3-CC** - 44.2% overhead, hardware-assisted confidentiality (software fallback in this benchmark)
5. **V3-MPC** - 66.6% overhead, strongest cryptographic guarantees

### Variance Analysis

| Version | Std Dev (ms) | Coefficient of Variation |
|---------|--------------|-------------------------|
| V3 | 3.97 | 1.1% |
| V2 | 43.37 | 8.6% |
| V3-CC | 45.16 | 8.8% |
| V3-OT | 52.49 | 11.3% |
| V3-MPC | 58.59 | 9.9% |

V3 shows exceptional consistency with only 1.1% coefficient of variation, making it highly predictable for production deployments.

### Security vs Performance Trade-offs

| Version | Security Level | Primary Protection | Performance Impact |
|---------|---------------|-------------------|-------------------|
| V3 | Moderate | Secret sharing | Baseline |
| V3-OT | High | Input privacy (OT) | +31.2% |
| V2 | Moderate | Secret sharing | +41.2% |
| V3-CC | High | Memory encryption | +44.2% |
| V3-MPC | Highest | Full MPC protocol | +66.6% |

---

## Reproducibility

These benchmarks can be reproduced using:

```bash
./scripts/run_benchmarks.sh --v2 --v3 --v3-ot --v3-mpc --v3-cc
```

Or individually:

```bash
./scripts/run_benchmarks.sh --v2      # V2 only
./scripts/run_benchmarks.sh --v3      # V3 only
./scripts/run_benchmarks.sh --v3-ot   # V3-OT only
./scripts/run_benchmarks.sh --v3-mpc  # V3-MPC only
./scripts/run_benchmarks.sh --v3-cc   # V3-CC only
```

Configuration options:

```bash
./scripts/run_benchmarks.sh --iterations 20 --max-tokens 50
```

Results are saved to `benchmarks_results/<timestamp>/` with a `latest` symlink pointing to the most recent run.

---

## Data Files

The raw benchmark data is available in JSON format:

- `benchmarks_results/20260114_224009/v2_benchmark.json`
- `benchmarks_results/20260114_224009/v3_benchmark.json`
- `benchmarks_results/20260114_224009/v3-ot_benchmark.json`
- `benchmarks_results/20260114_224009/v3-mpc_benchmark.json`
- `benchmarks_results/20260114_224009/v3-cc_benchmark.json`

System information: `benchmarks_results/20260114_224009/system_info.txt`

---

## Notes

1. **All implementations are production-ready** with no mocks, placeholders, or simulations.

2. **V3-CC uses software AES-256-GCM** because the benchmark environment (RunPod) does not expose Intel TDX to the VM. On a true Confidential VM with TDX enabled, hardware AES-256-XTS encryption would be used with potentially different performance characteristics.

3. **Tokens/sec calculation** is based on the 19 prompt tokens processed during the prefill phase. Decode phase was not measured in these benchmarks.

4. **Warmup runs** (3 iterations) are excluded from all statistics to ensure measurements reflect steady-state performance.

5. **Temperature setting** (0.7) does not affect prefill performance but is included for completeness.
