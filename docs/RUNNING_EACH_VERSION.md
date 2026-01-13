# Running Each ShardLM Version

This guide provides step-by-step instructions for running each ShardLM protocol version with example prompts and expected outputs.

## Prerequisites

Before running any version, ensure you have:

1. **Built the server and client**:
   ```bash
   cargo build -p shardlm-v2-server --features cuda --release
   cargo build -p shardlm-v2-client --release
   ```

2. **Downloaded model weights** (Qwen 2.5 1.5B recommended for testing):
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='/workspace/qwen2.5-1.5b-instruct-weights')"
   ```

## Version Overview

| Version | Security Model | GPU Required | Use Case |
|---------|---------------|--------------|----------|
| **V2** | Secret sharing + server reconstruction | Any CUDA GPU | Development, baseline benchmarks |
| **V3** | Secret sharing + client reconstruction | Any CUDA GPU | Production baseline |
| **V3-CC** | Hardware TEE (H100 Confidential Computing) | H100 only | Maximum hardware security |
| **V3-MPC** | Beaver triples MPC | Any CUDA GPU | Cryptographic security (slight accuracy loss) |
| **V3-OT** | Oblivious Transfer | Any CUDA GPU | Information-theoretic security |

---

## V2: GPU-Accelerated Secret Sharing

V2 uses additive secret sharing for linear operations with server-side reconstruction for nonlinear operations. This is the fastest version but provides partial security.

### Start the Server

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
./target/release/shardlm-v2-server
```

### Run Generation

```bash
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "What is the capital of France?" \
  --max-tokens 30 \
  --timing \
  --endpoint v2 \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Expected Output:**
```
Generating from: "What is the capital of France?"
Server: http://localhost:9090
Endpoint: v2

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Generated: The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.

Timing:
  Embedding:      45.2 ms
  Prefill:       382.1 ms
  Decode:       2450.3 ms
  Total:        2877.6 ms
  Tokens:     30
  Speed:      12.24 tok/s
```

### Run Benchmark

```bash
./target/release/shardlm-v2-client benchmark \
  -s http://localhost:9090 \
  --runs 10 \
  --warmup 2 \
  --endpoint v2
```

---

## V3: Optimized Secret Sharing

V3 minimizes GPU-to-CPU transfers by keeping intermediate states on the GPU. Security properties are similar to V2.

### Start the Server

Same as V2 (the server supports all V3 endpoints):

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
./target/release/shardlm-v2-server
```

### Run Generation

```bash
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "Explain photosynthesis in simple terms." \
  --max-tokens 50 \
  --timing \
  --endpoint v3 \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Expected Output:**
```
Generating from: "Explain photosynthesis in simple terms."
Server: http://localhost:9090
Endpoint: v3

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Generated: Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose (sugar) and oxygen. Think of it as the plant's way of making its own food using energy from the sun.

Timing:
  Embedding:      43.8 ms
  Prefill:       395.2 ms
  Decode:       4125.7 ms
  Total:        4564.7 ms
  Tokens:     50
  Speed:      12.12 tok/s
```

### Run Benchmark

```bash
./target/release/shardlm-v2-client benchmark \
  -s http://localhost:9090 \
  --runs 10 \
  --warmup 2 \
  --endpoint v3
```

---

## V3-CC: Confidential Computing (H100 Only)

V3-CC uses NVIDIA H100 Confidential Computing mode for hardware-encrypted GPU memory. The server cannot access plaintext even with root privileges.

### Requirements

- **NVIDIA H100 GPU** with Confidential Computing support
- CC mode enabled in GPU firmware

### Start the Server

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
SHARDLM_V2_CC_MODE=1 \
./target/release/shardlm-v2-server
```

### Verify Attestation

Before running inference, verify the hardware attestation:

```bash
# Get attestation report
curl -s http://localhost:9090/v3/cc/attestation | jq .

# Verify attestation (returns verification result)
curl -X POST http://localhost:9090/v3/cc/verify \
  -H "Content-Type: application/json" \
  -d '{"attestation_report": "<report_from_above>"}'
```

### Run Generation

```bash
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "What are the security benefits of confidential computing?" \
  --max-tokens 250 \
  --timing \
  --endpoint v3-cc \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Expected Output:**
```
Generating from: "What are the security benefits of confidential computing?"
Server: http://localhost:9090
Endpoint: v3-cc

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Generated: Confidential computing provides hardware-level protection for data in use. Key benefits include:
1. Memory encryption prevents even privileged software from accessing plaintext data
2. Remote attestation allows clients to verify the integrity of the execution environment
3. Protection against physical attacks on GPU memory

Timing:
  Embedding:      48.3 ms
  Prefill:       412.5 ms
  Decode:       5230.8 ms
  Total:        5691.6 ms
  Tokens:     60
  Speed:      11.47 tok/s
```

### Run Benchmark

```bash
./target/release/shardlm-v2-client benchmark \
  -s http://localhost:9090 \
  --runs 10 \
  --warmup 2 \
  --endpoint v3-cc
```

---

## V3-MPC: Beaver Triple-Based MPC

V3-MPC uses Beaver triples for secure multiplication, ensuring the server never sees plaintext activations. Uses polynomial approximations for nonlinear functions (slight accuracy loss).

### Start the Server

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
./target/release/shardlm-v2-server
```

### Run Generation

```bash
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "Write a haiku about privacy." \
  --max-tokens 250 \
  --timing \
  --endpoint v3-mpc \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Expected Output:**
```
Generating from: "Write a haiku about privacy."
Server: http://localhost:9090
Endpoint: v3-mpc

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Generated: Secrets kept within
Whispers shared with trusted few
Silence holds the key

Timing:
  Embedding:      44.1 ms
  Prefill:       425.8 ms
  Decode:       2680.4 ms
  Total:        3150.3 ms
  Tokens:     20
  Speed:      7.46 tok/s
```

### Run Benchmark

```bash
./target/release/shardlm-v2-client benchmark \
  -s http://localhost:9090 \
  --runs 10 \
  --warmup 2 \
  --endpoint v3-mpc
```

---

## V3-OT: Oblivious Transfer

V3-OT uses Oblivious Transfer for nonlinear operations via precomputed function tables. Provides information-theoretic security with constant 48 KB overhead.

### Start the Server

```bash
SHARDLM_V2_MODEL_DIR=/workspace/qwen2.5-1.5b-instruct-weights \
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b \
SHARDLM_V2_NUM_GPUS=1 \
SHARDLM_V2_PORT=9090 \
./target/release/shardlm-v2-server
```

### Check OT Configuration

```bash
curl -s http://localhost:9090/v3/ot/info | jq .
```

### Run Generation

```bash
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "Explain oblivious transfer in two sentences." \
  --max-tokens 250 \
  --timing \
  --endpoint v3-ot \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Expected Output:**
```
Generating from: "Explain oblivious transfer in two sentences."
Server: http://localhost:9090
Endpoint: v3-ot

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Generated: Oblivious transfer is a cryptographic protocol where a sender transfers one of several pieces of information to a receiver, but remains oblivious to which piece was transferred.

Timing:
  Embedding:      45.6 ms
  Prefill:       398.2 ms
  Decode:       3520.1 ms
  Total:        3963.9 ms
  Tokens:     40
  Speed:      11.35 tok/s
```

### Run Benchmark

```bash
./target/release/shardlm-v2-client benchmark \
  -s http://localhost:9090 \
  --runs 10 \
  --warmup 2 \
  --endpoint v3-ot \
  --output v3_ot_results.json
```

---

## Interactive Chat Mode

All versions support interactive chat. Use `--endpoint` to select the protocol version:

```bash
# Chat using V2 (default)
./target/release/shardlm-v2-client chat \
  -s http://localhost:9090 \
  --max-tokens 100 \
  --endpoint v2 \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json

# Chat using V3-OT (full cryptographic security)
./target/release/shardlm-v2-client chat \
  -s http://localhost:9090 \
  --max-tokens 100 \
  --endpoint v3-ot \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
```

**Example Session (V3-OT):**
```
ShardLM V2 Interactive Chat
Type 'quit' or 'exit' to end the session.
Endpoint: v3-ot

Loaded tokenizer from: /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json
Detected ChatML format (Qwen)
Session created: sess_abc123
Model: qwen2_5_1_5b

You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence where computers learn patterns from data without being explicitly programmed. Instead of following hard-coded rules, ML algorithms improve their performance through experience.
  [32 tokens] (2845ms, 11.2 tok/s)

You: Give me an example.
Assistant: A common example is email spam filtering. The system learns from millions of emails labeled as "spam" or "not spam" and develops patterns to classify new emails automatically.
  [38 tokens] (3210ms, 11.8 tok/s)

You: quit
Goodbye!
```

---

## Comparison Summary

| Version | Security | Accuracy | Speed (tok/s) | Hardware |
|---------|----------|----------|---------------|----------|
| V2 | Partial | 100% | ~12 | Any CUDA |
| V3 | Partial | 100% | ~12 | Any CUDA |
| V3-CC | Full (HW) | 100% | ~11.5 | H100 only |
| V3-MPC | Full (Crypto) | 98-99.5% | ~7-8 | Any CUDA |
| V3-OT | Full (Crypto) | 100% | ~11 | Any CUDA |

**Recommendations:**
- **Development/Testing**: Use V2 for fastest iteration
- **Production (any GPU)**: Use V3-OT for full security with minimal overhead
- **Production (H100)**: Use V3-CC for hardware-backed security
- **Maximum security**: Use V3-MPC (accept slight accuracy tradeoff)

---

## Troubleshooting

### Server Won't Start

```bash
# Check CUDA is available
nvidia-smi

# Check model path exists
ls -la /workspace/qwen2.5-1.5b-instruct-weights/config.json
```

### Connection Refused

```bash
# Ensure server is running
curl http://localhost:9090/health

# Check server logs for errors
```

### Garbled Output

Ensure you're using the `--tokenizer` flag:
```bash
./target/release/shardlm-v2-client generate \
  -s http://localhost:9090 \
  -p "Hello!" \
  --tokenizer /workspace/qwen2.5-1.5b-instruct-weights/tokenizer.json  # Required!
```

### Out of Memory

Reduce batch size or use a smaller model:
```bash
# For limited VRAM, use 1.5B model
SHARDLM_V2_MODEL_ARCH=qwen2_5_1_5b
```
