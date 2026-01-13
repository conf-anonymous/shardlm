# ShardLM V2 Security Model

**Version**: v2-baseline
**Status**: Deprecated (see V3 for reference implementation)
**Date**: 2026-01-10

## Overview

V2 implements a **hybrid secret sharing** model for secure inference. While it uses additive secret sharing for linear operations, it **reconstructs plaintext** during nonlinear operations.

## Security Guarantees (V2)

### What V2 Provides
- **Embedding Privacy**: Token embeddings are split into additive shares
- **Linear Operation Security**: Matrix multiplications preserve secret sharing
- **Transport Encryption**: HTTPS for client-server communication

### What V2 Does NOT Provide
- **Nonlinear Operation Security**: Server sees plaintext during RMSNorm, SiLU, Softmax
- **GPU Confidential Computing**: Standard CUDA memory (not encrypted)
- **Hardware Attestation**: No cryptographic proof of secure execution

## Technical Details

### Secret Sharing Scheme
```
plaintext = client_share + server_share
```

Both shares are random; neither reveals the plaintext alone.

### Linear Operations (Secure)
For weight matrix W and secret-shared input x:
```
y_client = W @ x_client
y_server = W @ x_server
# Reconstruction: y = y_client + y_server = W @ (x_client + x_server) = W @ x
```

### Nonlinear Operations (INSECURE in V2)

**RMSNorm** (`crates/v2/sharing/src/secure_nonlinear.rs:115`):
```rust
// RECONSTRUCTS PLAINTEXT
let x = x_client[i] + x_server[i];
let norm = (sum_sq / n as f32).sqrt();
```

**SiLU/SwiGLU** (`crates/v2/sharing/src/secure_nonlinear.rs:53`):
```rust
// RECONSTRUCTS PLAINTEXT
let x = x_client + x_server;
let silu = x / (1.0 + (-x).exp());
let z_server = 0.0;  // All output on client - defeats purpose
```

**Softmax** (`crates/v2/sharing/src/secure_nonlinear.rs`):
```rust
// RECONSTRUCTS PLAINTEXT for exp() and normalization
let x = x_client[i] + x_server[i];
let exp_x = (x - max_val).exp();
```

### GPU Operations (No CC)
```rust
// Standard CUDA allocation - memory visible to host
let tensor = device.htod_sync_copy(&data)?;
```

## Attack Vectors

1. **Server Memory Inspection**: Server can read reconstructed plaintext during nonlinear ops
2. **GPU Memory Dump**: No encryption on GPU memory
3. **Side-Channel**: Timing attacks on nonlinear operations

## Comparison: V2 vs V3

| Feature | V2 | V3 (Reference) |
|---------|----|--------------------|
| Linear Ops | Secret-shared | Secret-shared |
| Nonlinear Ops | Plaintext reconstruction | MPC with Beaver triples |
| GPU Memory | Standard CUDA | H100 CC encrypted |
| Attestation | None | Hardware attestation |
| Serialization | JSON | Binary (bincode) |
| Server Learns Input | Yes (during nonlinear) | No |
| Server Learns Output | Yes (during softmax) | No |

## Recommendation

V2 should only be used for:
- Performance baseline comparisons
- Development/debugging
- Scenarios where server is fully trusted

For production secure inference, use **V3 Reference Implementation**.

## Files Reference

- `crates/v2/sharing/src/secure_nonlinear.rs` - CPU nonlinear ops (reconstructs plaintext)
- `crates/v2/sharing/src/secure_nonlinear_gpu.rs` - GPU nonlinear ops (reconstructs plaintext)
- `crates/v2/server/src/routes/secure_inference.rs` - V2 endpoints
