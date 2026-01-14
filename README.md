# ShardLM

**Privacy-preserving language model inference where the server learns nothing about your inputs or outputs.**

ShardLM is a practical system for privacy-preserving LLM inference that combines additive secret sharing with GPU acceleration. Our key insight is that linear operations (99%+ of LLM computation) can be performed directly on secret shares without interaction, while nonlinear operations are handled securely via hardware-based trusted execution (V3-CC), Beaver triple MPC (V3-MPC), or oblivious transfer (V3-OT).

**Key Features:**
- **Strong Privacy**: Server never sees plaintext tokens, embeddings, activations, or logits
- **Practical Performance**: ~2× overhead vs plaintext inference, 25-46 tokens/second on 1.5B-7B models
- **Multiple Security Variants**: Choose between hardware security (V3-CC), cryptographic MPC (V3-MPC), or OT-based protocols (V3-OT)
- **Constant Overhead**: V3-OT achieves 48 KB cryptographic overhead regardless of model size
- **Model Agnostic**: Works with any open-weights transformer model (tested on Qwen 2.5 1.5B/7B)

## Quick Start (GPU)

### Prerequisites

- **NVIDIA GPU** with CUDA support (required for V2/V3 versions)
  - Qwen 2.5 1.5B: ~3GB VRAM (most GPUs)
  - Qwen 2.5 7B: ~14GB VRAM (A10G, RTX 4090, or better)
  - V3-CC: Requires H100 with Confidential Computing mode
- CUDA 12.0+
- Linux (tested on Ubuntu 22.04)

### Automated Setup (Recommended)

The easiest way to get started on a cloud GPU machine (RunPod, Lambda Labs, etc.):

```bash
# Clone the repository
git clone https://github.com/conf-anonymous/shardlm.git
cd shardlm

# Run the setup script
./scripts/setup_nvidia_machine.sh --1.5b
```

The script will automatically:
1. Detect your GPU and available features
2. Install Rust and system dependencies
3. Build the server and client with CUDA support
4. Download the Qwen 2.5 1.5B model weights

**Options:**
```bash
./scripts/setup_nvidia_machine.sh --1.5b    # Install 1.5B model (~3 GB)
./scripts/setup_nvidia_machine.sh --7b      # Install 7B model (~14 GB)
./scripts/setup_nvidia_machine.sh --all     # Install both models
./scripts/setup_nvidia_machine.sh --skip    # Skip model download
```

### Start the Server

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
./target/release/shardlm-v2-server
```

### Run the Client

```bash
# Generate text with streaming output
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "Hello!" \
  --max-tokens 50 \
  --timing \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json

# Interactive chat mode
./target/release/shardlm-v2-client chat \
  -s http://localhost:9090 \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Example Output:**
```
Generating from: "Hello!"
Server: http://localhost:9090
Endpoint: v2

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Generated: Hello! How can I assist you today?

Timing:
  Embedding:      44.9 ms
  Prefill:       376.9 ms
  Decode:        836.0 ms
  Total:        1257.8 ms
  Tokens:     10
  Speed:      11.96 tok/s
```

## Running Different Protocol Versions

ShardLM supports multiple protocol versions with different security/performance trade-offs:

| Version | Security Model | GPU Required | Use Case |
|---------|---------------|--------------|----------|
| **V2** | Secret sharing + server reconstruction | Any CUDA GPU | Development, baseline benchmarks |
| **V3** | Secret sharing + client reconstruction | Any CUDA GPU | Production baseline |
| **V3-CC** | Hardware TEE (H100 Confidential Computing) | H100 only | Maximum hardware security |
| **V3-MPC** | Beaver triples MPC | Any CUDA GPU | Cryptographic security |
| **V3-OT** | Oblivious Transfer | Any CUDA GPU | Information-theoretic security |

Use the `--endpoint` flag to select the protocol version:

```bash
# V2 (default)
./target/release/shardlm-v2-client generate -s http://localhost:9090 \
  -p "Hello!" --endpoint v2 --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json

# V3-OT (full cryptographic security)
./target/release/shardlm-v2-client generate -s http://localhost:9090 \
  -p "Hello!" --endpoint v3-ot --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

### Feature Flags for V3-CC and V3-MPC

V3-CC and V3-MPC require rebuilding the server with additional feature flags:

```bash
# V3-CC (requires H100 GPU)
cargo build -p shardlm-v2-server --features h100-cc,cuda --release

# V3-MPC
cargo build -p shardlm-v2-server --features mpc-secure,cuda --release

# All features
cargo build -p shardlm-v2-server --features h100-cc,mpc-secure,cuda --release
```

Without the appropriate feature flags, V3-CC and V3-MPC endpoints return 404.

**For complete step-by-step instructions, see [docs/RUNNING_EACH_VERSION.md](docs/RUNNING_EACH_VERSION.md)** - includes build commands, server configuration, example prompts, and expected outputs for each version.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                            CLIENT                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Tokenizer   │  │    Share     │  │      Sampler           │ │
│  │  + Embed     │→ │  Generation  │→ │     (logits)           │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│         Holds client share (x_C)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                        HTTP / TLS
                              │
┌─────────────────────────────────────────────────────────────────┐
│                     SERVER (Untrusted)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Axum API    │  │   Linear Ops │  │    Model Weights       │ │
│  │  (REST)      │  │  (on shares) │  │    (safetensors)       │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│         Computes on server share (x_S) only                     │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────────┐  │
│  │              NONLINEAR OPERATIONS (varies by version)     │  │
│  │  V3-CC:  GPU TEE (H100 Confidential Computing)            │  │
│  │  V3-MPC: Beaver triples + polynomial approximation        │  │
│  │  V3-OT:  Oblivious Transfer function tables               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Privacy Guarantees** (vary by version):

| Data | V2/V3 | V3-CC | V3-MPC | V3-OT |
|------|-------|-------|--------|-------|
| Token IDs | Client-side | Client-side | Client-side | Client-side |
| Embeddings | Secret-shared | Secret-shared | Secret-shared | Secret-shared |
| Linear activations | Secret-shared | Secret-shared | Secret-shared | Secret-shared |
| Nonlinear activations | Reconstructed | Hardware-encrypted | MPC (Beaver triples) | OT function tables |
| Output logits | Client reconstructs | Client reconstructs | Client reconstructs | Client reconstructs |
| Security | Partial | Full (HW) | Full (Crypto) | Full (Crypto) |

## How It Works

### 1. Client-Side Tokenization and Embedding

The client tokenizes input locally and computes embeddings, ensuring the server never sees token IDs:

```
Client                              Server
  |                                   |
  | 1. Tokenize input locally         |
  | 2. Compute embeddings locally     |
  | 3. Generate secret shares:        |
  |    X = X_client + X_server        |
  |                                   |
  | Send X_server share ----------->  |
  |                                   |
```

### 2. Secret Sharing for Linear Operations

For linear layer Y = XW + b (99%+ of computation):

```
X = X_client + X_server  (additive shares)

Server computes on shares (no interaction needed):
  Y_server = X_server × W + b

Client receives Y_server and reconstructs:
  Y = X_client × W + Y_server
```

### 3. Secure Nonlinear Operations

Nonlinear functions (softmax, SiLU, RMSNorm) are handled differently per variant:

| Variant | Method | Security |
|---------|--------|----------|
| V3-CC | GPU TEE reconstruction | Hardware-encrypted |
| V3-MPC | Beaver triples + polynomial approximation | Information-theoretic |
| V3-OT | Precomputed function tables via 1-of-N OT | Information-theoretic |

## Threat Model

ShardLM considers a **malicious server** adversary—the strongest practical threat in cloud-based ML inference.

| Entity | Capabilities | Trust Assumption |
|--------|--------------|------------------|
| Server | Root access to OS, hypervisor, GPU drivers; can inspect/modify all network traffic and CPU memory | Cannot break cryptographic assumptions (V3-MPC, V3-OT) or GPU hardware security (V3-CC) |
| Client | Holds private input data | Fully trusted |

**Security Guarantees** (V3-MPC and V3-OT provide information-theoretic security):
- Server learns nothing about client inputs beyond sequence length
- Server learns nothing about inference outputs
- Linear operations protected by additive secret sharing (all versions)
- Nonlinear operations protected by hardware encryption (V3-CC) or cryptographic protocols (V3-MPC, V3-OT)

**Accepted Leakage:**
- Sequence length (number of tokens)
- Timing of operations
- Information inherently leaked by the output

## Project Structure

```
shardlm/
├── crates/
│   ├── protocol/            # Wire format, framing, CRC validation
│   ├── ot/                  # Oblivious Transfer (IKNP extension)
│   ├── fixed_point/         # Q16.16 fixed-point arithmetic
│   ├── sharing/             # Additive secret sharing, attention, FFN
│   ├── model/               # Weight loading, tokenizer, RoPE
│   ├── server/              # V1 REST API server (CPU)
│   ├── harness/             # Plaintext reference for validation
│   ├── cli/                 # Command-line demo
│   ├── wasm-client/         # Browser WASM SDK
│   └── v2/                  # GPU-accelerated infrastructure
│       ├── core/            # GPU config, model architecture
│       ├── sharing/         # CUDA-accelerated secret sharing
│       ├── protocol/        # Extended protocol (128K context)
│       ├── model/           # Model loader, KV cache
│       ├── server/          # V2/V3 REST API server (GPU)
│       ├── client/          # Headless client for benchmarking
│       └── cc/              # Confidential Computing module
├── scripts/                 # Setup and benchmark scripts
├── benchmark_results/       # Pre-recorded benchmark data
├── docs/                    # Technical specifications
└── Cargo.toml               # Workspace configuration
```

## Supported Models

ShardLM is designed to work with any open-weights transformer model. The architecture is model-agnostic and supports standard transformer components (attention, FFN, RMSNorm, RoPE).

**Tested Models:**

| Model | Parameters | API Version | Status |
|-------|------------|-------------|--------|
| Qwen 2.5 1.5B Instruct | 1.5B | V2/V3 (GPU) | Fully tested |
| Qwen 2.5 7B Instruct | 7B | V2/V3 (GPU) | Fully tested |

**In Progress:**

| Model | Parameters | Notes |
|-------|------------|-------|
| Qwen 2.5 72B Instruct | 72B | Multi-GPU tensor parallelism |
| Llama 3.1 70B | 70B | GQA architecture validation |

## Development

### Manual Build (without setup script)

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build server with CUDA
cargo build -p shardlm-v2-server --features cuda --release

# Build client
cargo build -p shardlm-v2-client --release

# Download model weights
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='./qwen2.5-1.5b-instruct-weights')"
```

### Run Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p shardlm-sharing

# V2 crates (CPU mode)
cargo test -p shardlm-v2-core -p shardlm-v2-model
```

### Server Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARDLM_V2_MODEL_DIR` | - | Path to model weights (required) |
| `SHARDLM_V2_MODEL_ARCH` | `qwen2_5_1_5b` | Model architecture (`qwen2_5_1_5b`, `qwen2_5_7b`) |
| `SHARDLM_V2_HOST` | `0.0.0.0` | Server bind host |
| `SHARDLM_V2_PORT` | `9090` | Server bind port |
| `SHARDLM_V2_NUM_GPUS` | `1` | Number of GPUs for tensor parallelism |
| `SHARDLM_V2_MAX_SEQ_LEN` | `8192` | Maximum sequence length |

## Benchmarking

```bash
# Basic benchmark (V2 endpoint)
./target/release/shardlm-v2-client benchmark -s http://localhost:9090 --runs 10 --warmup 2

# V3-OT endpoint benchmark with JSON output
./target/release/shardlm-v2-client benchmark -s http://localhost:9090 \
  --runs 10 --warmup 2 --endpoint v3-ot --output results.json
```

## Documentation

- **[RUNNING_EACH_VERSION.md](docs/RUNNING_EACH_VERSION.md)** - Step-by-step tutorial for running each protocol version (V2, V3, V3-CC, V3-MPC, V3-OT)
- [V3_REFERENCE_IMPLEMENTATION.md](docs/V3_REFERENCE_IMPLEMENTATION.md) - Comprehensive V3 technical documentation
- [V2_SECURE_INFERENCE_SPEC.md](docs/V2_SECURE_INFERENCE_SPEC.md) - V2 secure inference specification
- [V2_SECURITY_MODEL.md](docs/V2_SECURITY_MODEL.md) - Security model and threat analysis
- [V3_OT_BENCHMARK_REPORT.md](docs/V3_OT_BENCHMARK_REPORT.md) - Performance benchmarks
- [SCALING_ANALYSIS.md](docs/SCALING_ANALYSIS.md) - Scaling analysis across model sizes
- [HARDWARE_REQUIREMENTS.md](docs/HARDWARE_REQUIREMENTS.md) - Hardware requirements guide

## License

Apache-2.0
