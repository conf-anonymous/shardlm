# ShardLM - Anonymous Submission

This repository contains the implementation artifacts for the paper submission.

## Repository Structure

```
crates/
├── protocol/          # Wire format, framing, CRC validation
├── fixed_point/       # Q16.16 fixed-point arithmetic
├── ot/                # Oblivious Transfer (IKNP extension)
├── sharing/           # Secret sharing, attention, FFN, RMSNorm
├── model/             # Weight loading, tokenizer, RoPE embeddings
├── harness/           # Plaintext reference implementation
├── server/            # v1 REST API server (CPU)
├── cli/               # Command-line demo interface
├── wasm-client/       # Browser WASM SDK
└── v2/
    ├── core/          # GPU config, tensor operations
    ├── sharing/       # CUDA-accelerated secret sharing
    ├── model/         # Model loaders, KV cache
    ├── protocol/      # Extended protocol (128K context)
    ├── server/        # v2/v3 REST API server (GPU)
    ├── client/        # Headless client for benchmarking
    └── cc/            # Confidential Computing module

scripts/               # Setup and benchmark scripts
docs/                  # Technical specifications
benchmark_results/     # Pre-recorded benchmark results
```

## Quick Start

### Build (v1 - CPU)
```bash
cargo build --release -p shardlm-server -p shardlm-cli
```

### Build (v2/v3 - GPU)
```bash
cargo build --release -p shardlm-v2-server -p shardlm-v2-client --features cuda
```

### Run Tests
```bash
cargo test --workspace
```

## Model Weights

Download model weights from HuggingFace (not included in repository):
- Qwen/Qwen2.5-1.5B-Instruct
- Qwen/Qwen2.5-7B-Instruct

## Hardware Requirements

- **v1 (CPU):** Any modern x86-64 CPU, 16GB+ RAM
- **v2/v3 (GPU):** NVIDIA GPU with CUDA 12.0+, 24GB+ VRAM
- **Full evaluation:** NVIDIA H100 80GB with Confidential Computing

See [docs/HARDWARE_REQUIREMENTS.md](docs/HARDWARE_REQUIREMENTS.md) for details.

## Documentation

- [V3_REFERENCE_IMPLEMENTATION.md](docs/V3_REFERENCE_IMPLEMENTATION.md) - Comprehensive V3 technical documentation
- [V2_SECURITY_MODEL.md](docs/V2_SECURITY_MODEL.md) - Security model and threat analysis
- [SCALING_ANALYSIS.md](docs/SCALING_ANALYSIS.md) - Scaling analysis across model sizes
- [V3_OT_BENCHMARK_REPORT.md](docs/V3_OT_BENCHMARK_REPORT.md) - Performance benchmarks

## License

Apache-2.0
