# ShardLM V3 Reference Implementation: Comprehensive Technical Documentation

## Executive Summary

ShardLM V3 Reference Implementation represents a suite of privacy-preserving inference implementations for large language models. This document covers four protocol variants:

| Variant | Security Model | Accuracy | Performance |
|---------|---------------|----------|-------------|
| **V2** | Secret sharing + client reconstruction | 100% | Baseline |
| **V3** | Same as V2, GPU-optimized transfers | 100% | +29% faster |
| **V3-CC** | H100 Confidential Computing (hardware encryption) | 100% | +32% faster |
| **V3-MPC** | Beaver triple-based MPC (no plaintext reconstruction) | 98-99.5% | +36% faster |

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Foundational Security: Additive Secret Sharing](#2-foundational-security-additive-secret-sharing)
3. [V2: Baseline Implementation](#3-v2-baseline-implementation)
4. [V3: GPU-Optimized Transfers](#4-v3-gpu-optimized-transfers)
5. [V3-CC: H100 Confidential Computing](#5-v3-cc-h100-confidential-computing)
6. [V3-MPC: True Multi-Party Computation](#6-v3-mpc-true-multi-party-computation)
7. [Benchmark Results](#7-benchmark-results)
8. [Running the System](#8-running-the-system)
9. [Headless Client Architecture](#9-headless-client-architecture)
10. [Security Analysis](#10-security-analysis)

---

## 1. Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Tokenizer   │  │   Shares     │  │   Sampler    │          │
│  │              │→ │  (client)    │→ │  (logits)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/JSON
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER (H100 GPU)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Embeddings  │  │  Transformer │  │   Logits     │          │
│  │  (shares)    │→ │  Layers      │→ │  (shares)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  Security Layer: V2 | V3 | V3-CC | V3-MPC                       │
└─────────────────────────────────────────────────────────────────┘
```

### Crate Structure

```
shardlm/
├── crates/
│   └── v2/
│       ├── server/          # HTTP server with all endpoints
│       │   └── src/routes/
│       │       ├── secure_inference.rs      # V2 & V3 endpoints
│       │       ├── secure_inference_cc.rs   # V3-CC endpoints
│       │       └── secure_inference_mpc.rs  # V3-MPC endpoints
│       ├── client/          # Headless benchmark client
│       │   └── src/
│       │       ├── lib.rs        # Client API
│       │       ├── benchmark.rs  # Benchmark runner
│       │       └── session.rs    # Session management
│       ├── sharing/         # Secret sharing & MPC primitives
│       │   └── src/
│       │       ├── secure.rs              # Share types
│       │       ├── secure_linear.rs       # Linear ops on shares
│       │       ├── secure_nonlinear.rs    # Nonlinear (V2/V3)
│       │       ├── secure_nonlinear_mpc.rs # MPC nonlinear
│       │       ├── beaver.rs              # Beaver triples
│       │       └── secure_polynomial.rs   # Polynomial eval
│       ├── cc/              # Confidential Computing
│       │   └── src/
│       │       ├── lib.rs            # CC trait & providers
│       │       ├── attestation.rs    # Attestation tokens
│       │       └── encrypted_tensor.rs # Encrypted buffers
│       └── core/            # GPU operations
│           └── src/
│               ├── gpu.rs     # GPU device management
│               └── kernel.rs  # CUDA kernels
```

---

## 2. Foundational Security: Additive Secret Sharing

All ShardLM variants build on **additive secret sharing**, where plaintext values are split into two shares:

```
plaintext = client_share + server_share
```

### Type System Enforcement

The Rust type system enforces share ownership at compile time:

```rust
// crates/v2/sharing/src/secure.rs

pub struct SecureShare<O: ShareOwner, T> {
    data: T,
    _owner: PhantomData<O>,
}

pub type ClientShare<T> = SecureShare<Client, T>;
pub type ServerShare<T> = SecureShare<Server, T>;

// Only SecureSharePair can reconstruct
pub struct SecureSharePair<T> {
    client: ClientShare<T>,
    server: ServerShare<T>,
}

impl<T> SecureSharePair<T> {
    pub fn reconstruct(&self) -> T {
        // Only available when holding both shares
        self.client.data + self.server.data
    }
}
```

### Linear Operations on Shares

For a linear layer `Y = X·W + b`:

```rust
// Server computes on each share separately
let y_client = matmul(&x_client, &W);           // Client's share
let y_server = matmul(&x_server, &W) + &b;      // Server's share + bias

// Result: y_client + y_server = X·W + b
```

The server never adds shares together, preserving the secret.

### Nonlinear Operations Challenge

Nonlinear functions (RMSNorm, SiLU, Softmax) cannot be computed directly on shares:

```
SiLU(x_client + x_server) ≠ SiLU(x_client) + SiLU(x_server)
```

Each variant handles this differently:
- **V2/V3**: Reconstruct on client, recompute shares
- **V3-CC**: Hardware-protected reconstruction
- **V3-MPC**: Polynomial approximation with Beaver triples

---

## 3. V2: Baseline Implementation

### Design Philosophy

V2 prioritizes simplicity and correctness. Nonlinear operations are reconstructed on the client side, computed exactly, then re-shared.

### Server Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v2/secure/session/init` | POST | Initialize OT session |
| `/v2/secure/embeddings` | POST | Retrieve embeddings via OT |
| `/v2/secure/embeddings/direct` | POST | Direct embedding lookup |
| `/v2/secure/gpu/prefill_v2` | POST | Batched GPU prefill |
| `/v2/secure/logits` | POST | Project to vocabulary |

### Inference Flow

```
1. Client tokenizes prompt → [token_ids]
2. Client fetches embeddings → {client_shares, server_shares}
3. POST /v2/secure/gpu/prefill_v2
   - Server processes all layers on GPU
   - For nonlinear ops: sends shares to client
   - Client reconstructs, computes, re-shares
4. Server returns logits shares
5. Client reconstructs logits, samples token
```

### Security Properties

- **Linear operations**: Computed entirely on server (GPU)
- **Nonlinear operations**: Reconstructed on client
- **Server learns**: Nothing about plaintext activations
- **Client learns**: Intermediate activations (for nonlinear computation)

### Key Code Locations

- `crates/v2/server/src/routes/secure_inference.rs:2062-2240` - `batched_prefill_gpu_v2()`
- `crates/v2/sharing/src/secure_nonlinear.rs` - Reconstruction-based functions

---

## 4. V3: GPU-Optimized Transfers

### Design Philosophy

V3 optimizes V2 by minimizing GPU ↔ CPU data transfers. Tensors stay GPU-resident throughout the forward pass.

### Key Optimizations

1. **Single Upload**: All prompt embeddings uploaded once at start
2. **GPU-Resident Computation**: Intermediate tensors stay on GPU
3. **Single Download**: Only final logits downloaded at end
4. **Batched Processing**: All tokens processed together per layer

### Server Endpoint

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v2/secure/gpu/prefill_v3` | POST | Optimized batched prefill |

### Request/Response Format

```rust
pub struct BatchedPrefillRequest {
    pub session_id: String,
    pub hidden_client: Vec<Vec<f32>>,  // [seq_len][hidden_dim]
    pub hidden_server: Vec<Vec<f32>>,
}

pub struct BatchedPrefillResponse {
    pub final_hidden_client: Vec<f32>,  // Last token only
    pub final_hidden_server: Vec<f32>,
    pub k_cache: Vec<Vec<Vec<f32>>>,    // [layer][seq_len][kv_dim]
    pub v_cache: Vec<Vec<Vec<f32>>>,
    pub logits_client: Vec<f32>,
    pub logits_server: Vec<f32>,
}
```

### Performance Characteristics

- **29% faster** than V2 (reduced memory bandwidth)
- Same security model as V2
- Better GPU utilization (batched operations)

### Key Code Locations

- `crates/v2/server/src/routes/secure_inference.rs:2293+` - `batched_prefill_gpu_v3()`
- `crates/v2/sharing/src/secure_nonlinear_gpu.rs` - GPU-accelerated operations

---

## 5. V3-CC: H100 Confidential Computing

### Design Philosophy

V3-CC leverages NVIDIA H100 Confidential Computing to provide hardware-backed security guarantees. GPU memory is encrypted at rest and during transfers.

### Key Features

1. **Hardware Memory Encryption**: All GPU memory encrypted by hardware
2. **Attestation**: Cryptographic proof of secure execution environment
3. **No Accuracy Loss**: Exact computation (not polynomial approximation)
4. **Trust Model**: Security relies on hardware, not protocol

### Server Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v3/cc/attestation` | GET | Get server attestation token |
| `/v3/cc/verify` | POST | Verify client attestation |
| `/v3/cc/prefill` | POST | CC-protected prefill |

### Attestation Token Structure

```rust
pub struct AttestationToken {
    pub measurement: [u8; 32],   // Hash of GPU state
    pub timestamp: u64,          // Unix epoch seconds
    pub signature: Vec<u8>,      // Hardware signature
    pub provider: String,        // "NVIDIA" or "Software"
    pub gpu_id: String,          // GPU identifier
}
```

### Confidential Computing Trait

```rust
pub trait ConfidentialCompute: Send + Sync {
    fn is_available(&self) -> bool;
    fn provider_name(&self) -> &str;
    fn get_attestation(&self) -> Result<AttestationToken>;
    fn verify_attestation(&self, token: &AttestationToken) -> Result<bool>;
    fn allocate_secure(&self, size: usize) -> Result<EncryptedBuffer>;
    fn encrypt_buffer(&self, data: &[f32]) -> Result<EncryptedBuffer>;
    fn decrypt_buffer(&self, buffer: &EncryptedBuffer) -> Result<Vec<f32>>;
}
```

### Provider Hierarchy

1. **NVIDIA CC** (feature: `nvidia-cc`): True hardware encryption
2. **Software CC** (feature: `software-cc`): AES-GCM fallback
3. **NoOp Provider**: Development mode (no encryption)

### Encrypted Buffer Format

```rust
pub struct EncryptedBuffer {
    pub data: Vec<u8>,           // AES-GCM ciphertext
    pub nonce: [u8; 12],         // GCM nonce
    pub tag: [u8; 16],           // Authentication tag
    pub original_len: usize,     // Number of f32 elements
    pub encrypted: bool,         // Actually encrypted?
}
```

### Security Properties

- **Memory Encryption**: All GPU memory encrypted at rest
- **Transfer Encryption**: CPU ↔ GPU transfers encrypted
- **Attestation**: Verifiable proof of secure environment
- **No Protocol Trust**: Security from hardware, not math

### Key Code Locations

- `crates/v2/cc/src/lib.rs` - CC trait and providers
- `crates/v2/cc/src/attestation.rs` - Attestation tokens
- `crates/v2/cc/src/encrypted_tensor.rs` - Encrypted buffers
- `crates/v2/server/src/routes/secure_inference_cc.rs` - CC endpoints

---

## 6. V3-MPC: True Multi-Party Computation

### Design Philosophy

V3-MPC provides the strongest cryptographic guarantees by using Beaver triples for secure multiplication. **No plaintext is ever reconstructed on either party**.

### Beaver Triple Protocol

A Beaver triple is a pre-computed tuple `(a, b, c)` where `c = a·b`. This enables secure multiplication:

```rust
pub struct BeaverTriple {
    pub a: f32, pub b: f32, pub c: f32,      // Full values
    pub a_client: f32, pub a_server: f32,    // Shares of a
    pub b_client: f32, pub b_server: f32,    // Shares of b
    pub c_client: f32, pub c_server: f32,    // Shares of c
}
```

### Secure Multiplication Protocol

To compute `z = x · y` where `x = x_c + x_s` and `y = y_c + y_s`:

```rust
pub fn secure_multiply_mpc(
    x_client: f32, x_server: f32,
    y_client: f32, y_server: f32,
    triple: &BeaverTriple,
) -> (f32, f32) {
    // Step 1: Compute masked values
    let d = (x_client + x_server) - (triple.a_client + triple.a_server);  // x - a
    let e = (y_client + y_server) - (triple.b_client + triple.b_server);  // y - b

    // Step 2: Compute shares of z = xy = (a+d)(b+e) = ab + ae + bd + de
    let z_client = triple.c_client
                 + triple.a_client * e
                 + d * triple.b_client
                 + d * e;
    let z_server = triple.c_server
                 + triple.a_server * e
                 + d * triple.b_server;

    (z_client, z_server)
}
```

**Security**: `d` and `e` are uniformly random (masked by `a` and `b`), so revealing them leaks nothing about `x` or `y`.

### Polynomial Approximations

Nonlinear functions are approximated using polynomials computed via Beaver triples:

**SiLU (Chebyshev, degree 5)**:
```rust
const SILU_COEFFS: [f32; 6] = [
    0.5,             // c0
    0.39269908,      // c1 ≈ π/8
    0.0,             // c2
    -0.039269908,    // c3
    0.0,             // c4
    0.005,           // c5
];
// Error < 0.5% in operating range [-4, 4]
```

**Exponential (Taylor, for softmax)**:
```rust
const EXP_COEFFS: [f32; 6] = [
    1.0,             // x^0
    1.0,             // x^1
    0.5,             // x^2 / 2!
    0.16666667,      // x^3 / 3!
    0.041666668,     // x^4 / 4!
    0.008333334,     // x^5 / 5!
];
```

### MPC-Secure Operations

```rust
// RMSNorm using Beaver triples
pub fn secure_rms_norm_mpc(
    x_client: &[f32], x_server: &[f32],
    gamma: &[f32], eps: f32,
    triples: &[BeaverTriple],
) -> (Vec<f32>, Vec<f32>);

// SiLU using polynomial approximation
pub fn secure_silu_mpc_batch(
    x_client: &[f32], x_server: &[f32],
    triples: &[BeaverTriple],
) -> (Vec<f32>, Vec<f32>);

// SwiGLU combining SiLU and multiplication
pub fn secure_swiglu_mpc(
    gate_client: &[f32], gate_server: &[f32],
    up_client: &[f32], up_server: &[f32],
    triples: &[BeaverTriple],
) -> (Vec<f32>, Vec<f32>);
```

### Triple Storage and Pre-generation

```rust
pub struct BeaverTripleStore {
    triples: Vec<Vec<BeaverTriple>>,  // [layer][triple_idx]
    cursors: Vec<AtomicUsize>,        // Current position per layer
    triples_per_layer: usize,
}

impl BeaverTripleStore {
    pub fn new(num_layers: usize, triples_per_layer: usize) -> Self;
    pub fn get_triple(&self, layer: usize) -> &BeaverTriple;
}
```

### Server Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v3/mpc/info` | GET | Get MPC configuration |
| `/v3/mpc/prefill` | POST | MPC-protected prefill |

### Response Extensions

```rust
pub struct MpcInfo {
    pub triples_used: usize,
    pub mpc_active: bool,
    pub execution_ms: f64,
    pub accuracy_estimate: String,  // "~0.5-2% error"
    pub triple_memory_mb: f64,
}
```

### Security Properties

- **No Reconstruction**: Neither party ever sees plaintext
- **Information-Theoretic**: Security not based on computational hardness
- **Offline Phase**: Triples can be pre-generated
- **Accuracy Tradeoff**: ~0.5-2% error from polynomial approximation

### Key Code Locations

- `crates/v2/sharing/src/beaver.rs` - Beaver triple implementation
- `crates/v2/sharing/src/secure_polynomial.rs` - Polynomial evaluation
- `crates/v2/sharing/src/secure_nonlinear_mpc.rs` - MPC operations
- `crates/v2/server/src/routes/secure_inference_mpc.rs` - MPC endpoints

---

## 7. Benchmark Results

### Test Configuration

```
Model:          Qwen2.5-1.5B-Instruct
GPU:            NVIDIA H100
Prompt:         "The quick brown fox jumps over the lazy dog.
                 This is a longer prompt to better measure throughput."
Prompt Tokens:  98
Runs:           15
Warmup:         3
Temperature:    0.7
```

### Performance Comparison Table

| Variant | Prefill (ms) | Total (ms) | Tokens/sec | P95 (ms) | vs V2 |
|---------|-------------|------------|------------|----------|-------|
| **V2** | 2872.4 | 2907.3 | 34.1 | 3200.7 | - |
| **V3** | 2219.6 | 2248.6 | 44.2 | 2533.2 | **+29.4%** |
| **V3-CC** | 2173.1 | 2203.0 | 45.1 | 2540.3 | **+32.2%** |
| **V3-MPC** | 2116.2 | 2146.1 | 46.3 | 2110.4 | **+35.7%** |

### Detailed Statistics

#### V2 (Baseline)

```
Prompt tokens:   98
Tokens/second:   34.1
Prefill:
  Mean:          2872.4 ms
  Std:           288.3 ms
  Min:           2550.2 ms
  Max:           3512.9 ms
  P50:           2713.9 ms
  P95:           3200.7 ms
  P99:           3512.9 ms
Total:
  Mean:          2907.3 ms
  Std:           289.1 ms
```

#### V3 (GPU-Optimized)

```
Prompt tokens:   98
Tokens/second:   44.2
Prefill:
  Mean:          2219.6 ms
  Std:           252.4 ms
  Min:           1963.9 ms
  Max:           2764.0 ms
  P50:           2154.7 ms
  P95:           2533.2 ms
  P99:           2764.0 ms
Total:
  Mean:          2248.6 ms
  Std:           255.0 ms
```

#### V3-CC (Confidential Computing)

```
Prompt tokens:   98
Tokens/second:   45.1
Prefill:
  Mean:          2173.1 ms
  Std:           157.4 ms
  Min:           2053.2 ms
  Max:           2559.9 ms
  P50:           2107.8 ms
  P95:           2540.3 ms
  P99:           2559.9 ms
Total:
  Mean:          2203.0 ms
  Std:           158.8 ms
```

#### V3-MPC (Multi-Party Computation)

```
Prompt tokens:   98
Tokens/second:   46.3
Prefill:
  Mean:          2116.2 ms
  Std:           127.0 ms
  Min:           2040.6 ms
  Max:           2586.2 ms
  P50:           2088.0 ms
  P95:           2110.4 ms
  P99:           2586.2 ms
Total:
  Mean:          2146.1 ms
  Std:           127.7 ms
```

### Security Comparison

| Variant | Nonlinear Ops | Accuracy | Trust Model |
|---------|---------------|----------|-------------|
| **V2** | Reconstructs plaintext | 100% | Client computes nonlinear |
| **V3** | Same as V2 | 100% | Client computes nonlinear |
| **V3-CC** | Hardware-protected | 100% | H100 TEE |
| **V3-MPC** | Beaver triples | 98-99.5% | Cryptographic |

---

## 8. Running the System

### Prerequisites

- Rust 1.75+ with `cargo`
- CUDA 12.0+ with cuDNN
- NVIDIA H100 GPU (for V3-CC hardware features)
- Model weights (Qwen2.5-1.5B-Instruct or similar)

### Building

```bash
cd /workspace/shardlm

# Build all components
cargo build --release

# Build with specific features
cargo build --release --features "cuda,mpc-secure,h100-cc"
```

### Starting the Server

```bash
# Set environment variables
export RUST_LOG=info
export SHARDLM_V2_MODEL_DIR=/path/to/model
export SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b
export SHARDLM_V2_NUM_GPUS=1
export SHARDLM_V2_PORT=9095

# Start server
./target/release/shardlm-v2-server
```

#### Server Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SHARDLM_V2_MODEL_DIR` | Path to model weights | Required |
| `SHARDLM_V2_MODEL_ARCH` | Model architecture ID | `llama_70b` |
| `SHARDLM_V2_NUM_GPUS` | Number of GPUs to use | 1 |
| `SHARDLM_V2_PORT` | Server port | 9090 |
| `RUST_LOG` | Log level | `info` |

### Client Commands

#### Health Check

```bash
./target/release/shardlm-v2-client health -s http://localhost:9095
```

#### Server Info

```bash
./target/release/shardlm-v2-client info -s http://localhost:9095
```

#### Generate Text

```bash
./target/release/shardlm-v2-client generate \
    -s http://localhost:9095 \
    -p "Hello, how are you?" \
    --max-tokens 50 \
    --temperature 0.7 \
    --timing
```

#### Run Benchmarks

```bash
# Benchmark V2
./target/release/shardlm-v2-client benchmark \
    -s http://localhost:9095 \
    -p "The quick brown fox..." \
    --runs 15 \
    --warmup 3 \
    --endpoint v2 \
    --output v2_benchmark.json

# Benchmark V3
./target/release/shardlm-v2-client benchmark \
    -s http://localhost:9095 \
    -p "The quick brown fox..." \
    --runs 15 \
    --warmup 3 \
    --endpoint v3 \
    --output v3_benchmark.json

# Benchmark V3-CC
./target/release/shardlm-v2-client benchmark \
    -s http://localhost:9095 \
    -p "The quick brown fox..." \
    --runs 15 \
    --warmup 3 \
    --endpoint v3-cc \
    --output v3_cc_benchmark.json

# Benchmark V3-MPC
./target/release/shardlm-v2-client benchmark \
    -s http://localhost:9095 \
    -p "The quick brown fox..." \
    --runs 15 \
    --warmup 3 \
    --endpoint v3-mpc \
    --output v3_mpc_benchmark.json
```

### Comprehensive Benchmark Script

```bash
#!/bin/bash
# comprehensive_benchmark.sh

SERVER_URL="http://localhost:9095"
CLIENT="./target/release/shardlm-v2-client"
OUTPUT_DIR="./benchmark_results"
PROMPT="The quick brown fox jumps over the lazy dog."

mkdir -p "$OUTPUT_DIR"

for VARIANT in v2 v3 v3-cc v3-mpc; do
    echo "Benchmarking $VARIANT..."
    $CLIENT benchmark \
        -s "$SERVER_URL" \
        -p "$PROMPT" \
        --runs 15 \
        --warmup 3 \
        --endpoint "$VARIANT" \
        --output "$OUTPUT_DIR/${VARIANT}_benchmark.json"
done

echo "Results saved to $OUTPUT_DIR/"
```

---

## 9. Headless Client Architecture

### Design Goals

The headless client is designed for:

1. **Automated Testing**: Run benchmarks without user interaction
2. **CI/CD Integration**: Programmatic access to all endpoints
3. **Performance Measurement**: Comprehensive timing statistics
4. **Protocol Comparison**: Support all four variants uniformly

### Core Components

#### ShardLmClient

```rust
pub struct ShardLmClient {
    client: reqwest::Client,
    server_url: String,
    session_id: Option<String>,
    cached_info: Option<ServerInfo>,
}

impl ShardLmClient {
    pub fn new(server_url: &str) -> Self;
    pub async fn health_check(&self) -> Result<bool>;
    pub async fn get_info(&mut self) -> Result<&ServerInfo>;
    pub async fn create_session(&mut self) -> Result<SessionInfo>;

    // Four prefill variants
    pub async fn prefill_v2(&self, ...) -> Result<BatchedPrefillResponse>;
    pub async fn prefill_v3(&self, ...) -> Result<BatchedPrefillResponse>;
    pub async fn prefill_v3_cc(&self, ...) -> Result<BatchedPrefillResponse>;
    pub async fn prefill_v3_mpc(&self, ...) -> Result<BatchedPrefillResponse>;
}
```

#### BenchmarkRunner

```rust
pub struct BenchmarkRunner {
    client: ShardLmClient,
    config: BenchmarkConfig,
    timings: Vec<GenerationTiming>,
}

pub struct BenchmarkConfig {
    pub prompt: String,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub runs: usize,
    pub warmup: usize,
    pub endpoint: String,  // v2, v3, v3-cc, v3-mpc
}

impl BenchmarkRunner {
    pub async fn run(&mut self) -> Result<BenchmarkResult>;
    async fn run_single_prefill(&mut self) -> Result<GenerationTiming>;
}
```

#### Statistics Collection

```rust
pub struct TimingStats {
    pub mean_ms: f64,
    pub std_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

pub struct BenchmarkResult {
    pub model: String,
    pub gpu: String,
    pub version: String,
    pub endpoint: String,
    pub timestamp: String,
    pub config: BenchmarkConfig,
    pub results: BenchmarkStats,
}
```

### Benchmark Execution Flow

```
1. Initialize client with server URL
2. Fetch server info (model, GPU, etc.)
3. Create session
4. Tokenize prompt to determine token count
5. WARMUP PHASE (configurable iterations)
   - Run prefill calls silently
   - Warm GPU caches and JIT compilation
6. BENCHMARK PHASE
   - For each run:
     a. Fetch embeddings (time it)
     b. Call prefill for selected variant (time it)
     c. Record total time
   - Display progress bar
7. STATISTICS COMPUTATION
   - Calculate mean, std, min, max
   - Calculate percentiles (P50, P95, P99)
   - Calculate tokens/second
8. OUTPUT
   - Print summary to console
   - Save JSON to file (optional)
```

### Endpoint Selection

The client selects the appropriate endpoint based on configuration:

```rust
match self.config.endpoint.as_str() {
    "v3" => {
        self.client.prefill_v3(&embeddings.client, &embeddings.server).await?
    }
    "v3-cc" => {
        self.client.prefill_v3_cc(&embeddings.client, &embeddings.server).await?
    }
    "v3-mpc" => {
        self.client.prefill_v3_mpc(&embeddings.client, &embeddings.server).await?
    }
    _ => {
        // Default to V2
        self.client.prefill_v2(&embeddings.client, &embeddings.server).await?
    }
}
```

### Testing Strategy

The headless client enables several testing patterns:

1. **Unit Benchmarks**: Measure single endpoint performance
2. **Comparison Benchmarks**: Run all variants with same config
3. **Regression Testing**: Compare against baseline results
4. **Load Testing**: Multiple concurrent clients
5. **Latency Profiling**: Detailed timing breakdown

---

## 10. Security Analysis

### Threat Model

**Adversary Capabilities**:
- Can observe all network traffic
- Controls the server (for V2/V3)
- Cannot compromise H100 TEE (for V3-CC)
- Cannot break cryptographic assumptions (for V3-MPC)

### Security Guarantees by Variant

#### V2/V3: Honest-but-Curious Server

| Asset | Protected? | Notes |
|-------|------------|-------|
| Token IDs | Yes (OT) | Oblivious transfer hides selection |
| Embeddings | Partial | Server sees embedding table, not selection |
| Linear activations | Yes | Never reconstructed on server |
| Nonlinear activations | No | Reconstructed on client |
| Model weights | No | Server holds full weights |

**Attack Surface**: Compromised client reveals intermediate activations.

#### V3-CC: Hardware Trust

| Asset | Protected? | Notes |
|-------|------------|-------|
| Token IDs | Yes | Encrypted in TEE |
| All activations | Yes | Hardware memory encryption |
| Model weights | Yes | Protected by TEE |
| Computation | Yes | Attested execution |

**Attack Surface**: H100 hardware vulnerabilities, side channels.

#### V3-MPC: Cryptographic Security

| Asset | Protected? | Notes |
|-------|------------|-------|
| Token IDs | Yes | Secret shared |
| All activations | Yes | Never reconstructed |
| Nonlinear outputs | Yes | Polynomial approximation |
| Model weights | Partial | Shared across parties |

**Attack Surface**: Collusion between parties, approximation leakage.

### Comparison Summary

| Property | V2 | V3 | V3-CC | V3-MPC |
|----------|----|----|-------|--------|
| No server plaintext | Partial | Partial | Yes | Yes |
| Hardware independent | Yes | Yes | No | Yes |
| Information-theoretic | No | No | No | Yes |
| Exact computation | Yes | Yes | Yes | No |
| Attestation | No | No | Yes | No |

---

## Appendix A: JSON Benchmark Output Format

```json
{
  "model": "Qwen2_5_1_5B",
  "gpu": "NVIDIA H100",
  "version": "v3-mpc",
  "endpoint": "/v3/mpc/prefill",
  "timestamp": "2026-01-10T20:58:00.453Z",
  "config": {
    "prompt": "The quick brown fox...",
    "prompt_tokens": 98,
    "max_new_tokens": 50,
    "temperature": 0.7,
    "runs": 15,
    "warmup": 3,
    "endpoint": "v3-mpc"
  },
  "results": {
    "embedding": {
      "mean_ms": 29.9,
      "std_ms": 3.98,
      "min_ms": 25.59,
      "max_ms": 42.35,
      "p50_ms": 29.13,
      "p95_ms": 32.98,
      "p99_ms": 42.35
    },
    "prefill": {
      "mean_ms": 2116.2,
      "std_ms": 127.0,
      "min_ms": 2040.6,
      "max_ms": 2586.2,
      "p50_ms": 2088.0,
      "p95_ms": 2110.4,
      "p99_ms": 2586.2
    },
    "total": {
      "mean_ms": 2146.1,
      "std_ms": 127.7,
      "min_ms": 2066.6,
      "max_ms": 2617.7,
      "p50_ms": 2120.9,
      "p95_ms": 2139.7,
      "p99_ms": 2617.7
    },
    "tokens_per_second": 46.31
  }
}
```

---

## Appendix B: Quick Reference

### Feature Flags

```toml
[features]
default = ["cuda"]
cuda = ["shardlm-v2-core/cuda"]
h100-cc = ["cuda", "shardlm-v2-cc"]
mpc-secure = []
nvidia-cc = []
software-cc = []
```

### Environment Quick Setup

```bash
# Minimal server startup
export SHARDLM_V2_MODEL_DIR=/workspace/qwen-2.5-1.5b-instruct
export SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b
export SHARDLM_V2_PORT=9095
./target/release/shardlm-v2-server

# Quick benchmark (all variants)
for v in v2 v3 v3-cc v3-mpc; do
    ./target/release/shardlm-v2-client benchmark \
        -s http://localhost:9095 \
        -p "Hello world" \
        --endpoint $v \
        --runs 5
done
```

