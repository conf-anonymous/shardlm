# ShardLM

**Privacy-preserving language model inference where the server learns nothing about your inputs or outputs.**

ShardLM is a practical system for privacy-preserving LLM inference that combines additive secret sharing with GPU acceleration. Our key insight is that linear operations (99%+ of LLM computation) can be performed directly on secret shares without interaction, while nonlinear operations are handled securely via hardware-based trusted execution (V3-CC), Beaver triple MPC (V3-MPC), or oblivious transfer (V3-OT).

**Key Features:**
- **Strong Privacy**: Server never sees plaintext tokens, embeddings, activations, or logits
- **Practical Performance**: ~2× overhead vs plaintext inference, 25-46 tokens/second on 1.5B-7B models
- **Multiple Security Variants**: Choose between hardware security (V3-CC), cryptographic MPC (V3-MPC), or OT-based protocols (V3-OT)
- **Constant Overhead**: V3-OT achieves 48 KB cryptographic overhead regardless of model size
- **Model Agnostic**: Works with any open-weights transformer model (tested on Qwen 2.5 1.5B/7B)

## Quick Start

### Prerequisites

- Rust 1.75+
- Python 3.8+ (for model downloads)
- CUDA 12.0+ (for GPU-accelerated inference)

### 1. Clone and Build

```bash
git clone https://github.com/conf-anonymous/shardlm.git
cd shardlm
cargo build --release
```

### 2. Download Model Weights

```bash
# Qwen 2.5 1.5B Instruct (recommended for v1)
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='./qwen2.5-1.5b-instruct-weights')"
```

### 3. Run the Server

#### V1 Server (CPU)

```bash
SHARDLM_MODEL_DIR=./qwen2.5-1.5b-instruct-weights \
cargo run -p shardlm-server --release
```

Server starts at `http://localhost:8080`

#### V2/V3 Server (GPU)

```bash
# Qwen 2.5 1.5B
SHARDLM_V2_MODEL_DIR=./qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_PORT=9090 \
cargo run -p shardlm-v2-server --features cuda --release

# Qwen 2.5 7B
SHARDLM_V2_MODEL_DIR=./qwen2.5-7b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_7b \
SHARDLM_V2_PORT=9090 \
cargo run -p shardlm-v2-server --features cuda --release
```

Server starts at `http://localhost:9090`

**V2/V3 Server Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARDLM_V2_MODEL_DIR` | - | Path to model weights (required) |
| `SHARDLM_V2_MODEL_ARCH` | `llama3_1_70b` | Model architecture (`qwen2_5_1_5b`, `qwen2_5_7b`) |
| `SHARDLM_V2_HOST` | `0.0.0.0` | Server bind host |
| `SHARDLM_V2_PORT` | `8081` | Server bind port |
| `SHARDLM_V2_NUM_GPUS` | `4` | Number of GPUs for tensor parallelism |
| `SHARDLM_V2_MAX_SEQ_LEN` | `8192` | Maximum sequence length |

### 4. Run the Client

#### Build the Client

```bash
cargo build -p shardlm-v2-client --release
```

#### Basic Commands

```bash
# Health check
./target/release/shardlm-v2-client health -s http://localhost:9090

# Get server info (model, GPU, dimensions)
./target/release/shardlm-v2-client info -s http://localhost:9090

# Generate text
./target/release/shardlm-v2-client generate -s http://localhost:9090 -p "Hello, how are you?"

# Generate with options
./target/release/shardlm-v2-client generate -s http://localhost:9090 \
  -p "Explain quantum computing" --max-tokens 100 --temperature 0.7 --timing

# Interactive chat
./target/release/shardlm-v2-client chat -s http://localhost:9090
```

#### Benchmarking

```bash
# Basic benchmark (V2 endpoint)
./target/release/shardlm-v2-client benchmark -s http://localhost:9090 --runs 10 --warmup 2

# V3 endpoint benchmark
./target/release/shardlm-v2-client benchmark -s http://localhost:9090 \
  --runs 10 --warmup 2 --endpoint v3

# V3-OT endpoint benchmark with output
./target/release/shardlm-v2-client benchmark -s http://localhost:9090 \
  --runs 10 --warmup 2 --endpoint v3-ot --output results.json
```

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

## Versions

### V1 - CPU Baseline

V1 provides a CPU-based reference implementation for development and validation.

| Feature | Value |
|---------|-------|
| Model | Qwen 2.5 1.5B Instruct |
| Context | 1024 input + 1024 output tokens |
| Compute | CPU (with SIMD on aarch64) |
| Precision | i32 fixed-point (Q16.16) |

**Capabilities:**
- Private embedding retrieval via batched 1-of-V OT
- Secure linear layers via additive secret sharing
- Full transformer forward pass (attention, FFN, RMSNorm, RoPE)
- Autoregressive generation with streaming tokens

**Limitations:**
- Side-channel resistance (timing, memory patterns)
- Model extraction resistance
- Production performance

### V2 - GPU Baseline

V2 introduces GPU acceleration with secret sharing for linear operations. Nonlinear operations reconstruct plaintext on the server.

| Feature | Value |
|---------|-------|
| Model | Qwen 2.5 1.5B / 7B Instruct |
| Context | 32K tokens (1.5B) / 128K tokens (7B) |
| Compute | CUDA (H100 tensor cores) |
| Precision | BF16/FP16 |
| Attention | Grouped Query Attention (GQA) |
| Security | Secret sharing + server-side reconstruction |

### V3 - Reference Implementation

V3 represents the production-ready implementation with multiple security variants:

| Variant | Security Model | Accuracy | Performance |
|---------|---------------|----------|-------------|
| **V3** | Secret sharing + client reconstruction | 100% | +11.8% vs V2 |
| **V3-CC** | H100 Confidential Computing (hardware encryption) | 100% | +18.7% vs V2 |
| **V3-MPC** | Beaver triple-based MPC (no plaintext reconstruction) | 98-99.5% | +24.5% vs V2 |
| **V3-OT** | Oblivious Transfer for nonlinear functions | 100% | +12.2% vs V2 |

**V3-CC (Confidential Computing):**
- Hardware-encrypted GPU memory (H100 CC mode)
- Remote attestation for client verification
- Server cannot access plaintext even with root access

**V3-MPC (Multi-Party Computation):**
- Beaver triple-based secure multiplication
- Polynomial approximations for nonlinear functions
- Server never sees plaintext at any point

**V3-OT (Oblivious Transfer):**
- Precomputed function tables for nonlinear operations
- Information-theoretic security (no computational assumptions)
- Constant 48 KB overhead per nonlinear evaluation

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARDLM_BIND_ADDR` | `0.0.0.0` | Server listen address |
| `SHARDLM_PORT` | `8080` | Server port |
| `SHARDLM_MODEL_DIR` | `qwen2.5-1.5b-instruct-weights` | Path to model weights |
| `SHARDLM_MAX_SESSIONS` | `1000` | Max concurrent sessions |
| `SHARDLM_SESSION_TTL_SECS` | `900` | Session TTL (seconds) |
| `SHARDLM_MAX_PROMPT_LEN` | `1024` | Max input tokens |
| `SHARDLM_CORS_ORIGINS` | `http://localhost:3000` | CORS origins (comma-separated) |

## API Endpoints

### V1 (CPU Baseline)

```bash
GET  /health                  # Basic health check
GET  /ready                   # Model loaded and ready
GET  /v1/info                 # Server info (model, dimensions)
POST /v1/session/new          # Create session
POST /v1/inference/forward    # Secure forward pass
```

### V2/V3 (GPU)

```bash
GET  /health                  # Basic health check
GET  /ready                   # Model loaded and ready
GET  /v2/info                 # Server info (model, GPU, dimensions)
POST /v2/session/new          # Create session
POST /v2/generate             # Plaintext generation (testing only)
```

### V3 Secure Inference

```bash
# V3-CC (Confidential Computing)
GET  /v3/cc/attestation       # Get hardware attestation
POST /v3/cc/verify            # Verify attestation
POST /v3/cc/prefill           # Secure prefill with CC

# V3-MPC (Beaver Triples)
GET  /v3/mpc/info             # MPC configuration info
POST /v3/mpc/prefill          # Secure prefill with MPC

# V3-OT (Oblivious Transfer)
GET  /v3/ot/info              # OT configuration info
GET  /v3/ot/tables            # Function lookup tables
POST /v3/ot/prefill           # Secure prefill with OT
```

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

### 4. Full Inference Pipeline

```
[Client] Tokenize → Embed → Share Generation
                         ↓
[Server] Linear Ops (on shares) → Secure Nonlinear → Linear Ops → ...
                         ↓
[Client] Reconstruct logits → Sample next token
```

## Development

### Run Tests

```bash
# All tests
cargo test --workspace

# Specific crate
cargo test -p shardlm-sharing

# v2 crates (CPU mode)
cargo test -p shardlm-v2-core -p shardlm-v2-model
```

### Build with CUDA

```bash
# Requires CUDA toolkit installed

# Build v2 server with CUDA
cargo build -p shardlm-v2-server --features cuda --release

# Build v2 client
cargo build -p shardlm-v2-client --release

# Build all v2 crates with CUDA
cargo build -p shardlm-v2-core -p shardlm-v2-model -p shardlm-v2-server --features cuda --release
```

### Common Pitfalls

#### GPU Memory Requirements

For optimal performance, use GPUs with sufficient VRAM to keep model weights resident:
- **Qwen 2.5 1.5B**: ~3GB VRAM (fits on most GPUs)
- **Qwen 2.5 7B**: ~14GB VRAM (A10G, RTX 4090, or better)

#### Axum Handlers and Lock Guards

**Problem**: Axum async handlers must return `Send` futures. Holding `RwLockReadGuard` or `RwLockWriteGuard` (from `parking_lot` or `std`) across `.await` points causes compilation errors like:

```
error: future cannot be sent between threads safely
  --> src/routes/inference.rs
   |
   | #[axum::debug_handler]
   | ^^^^^^^^^^^^^^^^^^^^^^ future returned by `handler` is not `Send`
   |
   = help: within `impl Future<...>`, the trait `Send` is not implemented for `*mut ()`
note: future is not `Send` as this value is used across an await
   |
   | let guard = state.get_lock()?;
   |     ----- has type `RwLockReadGuard<...>` which is not `Send`
   | ...
   | .await
   |  ^^^^^ await occurs here, with `guard` maybe used later
```

**Solution**: Scope lock guards so they're dropped before any `.await`:

```rust
// BAD - guard held across await
let guard = state.get_tokenizer()?;
let tokens = guard.encode(&prompt)?;
drop(guard);  // explicit drop doesn't help!
let result = some_async_fn().await;  // ERROR: guard "maybe used later"

// GOOD - scope block ensures guard is dropped
let tokens = {
    let guard = state.get_tokenizer()?;
    guard.encode(&prompt)?
}; // guard dropped here
let result = some_async_fn().await;  // OK

// GOOD - closure for complex logic with early returns
let result = (|| -> Result<Data, Error> {
    let guard = state.get_tokenizer()?;
    // ... complex logic ...
    Ok(data)
})(); // guard dropped when closure returns
let result = result?;
let async_result = some_async_fn().await;  // OK
```

## Deployment

### GPU Instance Setup

| GPU | VRAM | RAM | vCPU | Use Case |
|-----|------|-----|------|----------|
| NVIDIA RTX 4090 | 24 GB | 41 GB | 12 | Qwen 2.5 1.5B, 7B (V2/V3/V3-MPC/V3-OT) |
| NVIDIA H100 SXM | 80 GB | 125 GB | 20 | Qwen 2.5 7B (V3-CC) |
| 2x NVIDIA H100 SXM | 160 GB | 250 GB | 40 | 70B+ models (tensor parallelism) |

### Automated Setup (RunPod, Lambda Labs, etc.)

Use the setup script to automatically configure a cloud GPU machine:

```bash
# Clone the repository
git clone https://github.com/conf-anonymous/shardlm.git
cd shardlm

# Run the setup script (interactive mode)
./scripts/setup_nvidia_machine.sh
```

The script will:
1. Detect your GPU and available features
2. Install Rust and dependencies
3. Build the server and client with CUDA support
4. Prompt you to select a model to download

**Non-interactive mode:**

```bash
./scripts/setup_nvidia_machine.sh --1.5b    # Install Qwen 2.5 1.5B only (~3 GB)
./scripts/setup_nvidia_machine.sh --7b      # Install Qwen 2.5 7B only (~14 GB)
./scripts/setup_nvidia_machine.sh --all     # Install both models
./scripts/setup_nvidia_machine.sh --skip    # Skip model download
```

After setup, start the server:

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
./target/release/shardlm-v2-server
```

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

## Supported Models

ShardLM is designed to work with any open-weights transformer model. The architecture is model-agnostic and supports standard transformer components (attention, FFN, RMSNorm, RoPE).

**Tested Models:**

| Model | Parameters | API Version | Status |
|-------|------------|-------------|--------|
| Qwen 2.5 1.5B Instruct | 1.5B | V1 (CPU), V2/V3 (GPU) | Fully tested |
| Qwen 2.5 7B Instruct | 7B | V2/V3 (GPU) | Fully tested |

**In Progress:**

| Model | Parameters | Notes |
|-------|------------|-------|
| Qwen 2.5 72B Instruct | 72B | Multi-GPU tensor parallelism |
| Llama 3.1 70B | 70B | GQA architecture validation |

## Documentation

- [V3_REFERENCE_IMPLEMENTATION.md](docs/V3_REFERENCE_IMPLEMENTATION.md) - Comprehensive V3 technical documentation
- [V2_SECURE_INFERENCE_SPEC.md](docs/V2_SECURE_INFERENCE_SPEC.md) - V2 secure inference specification
- [V2_SECURITY_MODEL.md](docs/V2_SECURITY_MODEL.md) - Security model and threat analysis
- [V3_OT_BENCHMARK_REPORT.md](docs/V3_OT_BENCHMARK_REPORT.md) - Performance benchmarks
- [SCALING_ANALYSIS.md](docs/SCALING_ANALYSIS.md) - Scaling analysis across model sizes
- [HARDWARE_REQUIREMENTS.md](docs/HARDWARE_REQUIREMENTS.md) - Hardware requirements guide
- [MULTI_GPU_INFERENCE_STATUS.md](docs/MULTI_GPU_INFERENCE_STATUS.md) - Multi-GPU inference status

## License

Apache-2.0
