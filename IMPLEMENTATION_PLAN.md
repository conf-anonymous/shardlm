# Implementation Plan: Real V3-OT and V3-MPC Integration

## Executive Summary

This plan addresses the critical issue where V3-OT and V3-MPC endpoints currently **simulate** security overhead rather than actually implementing the secure protocols. The cryptographic primitives exist but are not wired into the inference endpoints.

## Current State

### What EXISTS and is REAL:
1. **IKNP OT Extension** (`crates/ot/src/iknp.rs`) - Full cryptographic implementation with:
   - Curve25519 Diffie-Hellman for base OT
   - AES-128 for PRF
   - SHA-256 for key derivation
   - Proper zeroization of sensitive data

2. **Beaver Triple MPC** (`crates/v2/sharing/src/beaver.rs`) - Complete implementation with:
   - `secure_multiply_mpc()` for secure multiplication
   - `BeaverTripleStore` for pre-generated triples

3. **MPC-Secure Nonlinear Operations** (`crates/v2/sharing/src/secure_nonlinear_mpc.rs`):
   - `secure_silu_mpc_batch()` - SiLU using polynomial approximation + Beaver triples
   - `secure_rms_norm_mpc()` - RMSNorm using Beaver triples
   - `secure_swiglu_mpc()` - SwiGLU using Beaver triples
   - `secure_softmax_mpc()` - Softmax using Beaver triples

### What DOESN'T work (the problem):
- V3-OT endpoint (`secure_inference_ot.rs`): Calls base V3 and adds simulated overhead
- V3-MPC endpoint (`secure_inference_mpc.rs`): Calls base V3 and generates unused triples

---

## Implementation Plan

### Part 1: V3-MPC Integration (Easier)

The MPC-secure nonlinear operations already exist. We need to wire them into the inference pipeline.

#### Step 1.1: Create GPU-compatible MPC operations
**File:** `crates/v2/sharing/src/secure_nonlinear_mpc_gpu.rs` (new file)

```rust
// Wrap CPU MPC operations with GPU data transfer
pub fn secure_rms_norm_mpc_gpu(
    x_client: &CudaTensor,
    x_server: &CudaTensor,
    gamma: &CudaTensor,
    eps: f32,
    triples: &[BeaverTriple],
    kernels: &KernelContext,
    device: &GpuDevice,
) -> Result<(CudaTensor, CudaTensor), String> {
    // 1. Download shares from GPU to CPU
    // 2. Call secure_rms_norm_mpc
    // 3. Upload results back to GPU
}
```

#### Step 1.2: Modify `secure_inference_mpc.rs`
Replace the simulated call to `batched_prefill_gpu_v3` with actual MPC operations:

```rust
pub async fn mpc_prefill(...) -> Result<Json<MpcPrefillResponse>> {
    // Get pre-generated triples from store
    let store = TRIPLE_STORE.get()...;

    // For each layer:
    for layer_idx in 0..num_layers {
        // 1. RMSNorm using secure_rms_norm_mpc (with Beaver triples)
        // 2. QKV projection (linear - stays the same)
        // 3. Attention softmax using secure_softmax_mpc
        // 4. O projection (linear - stays the same)
        // 5. RMSNorm again using secure_rms_norm_mpc
        // 6. FFN gate/up projection (linear - stays the same)
        // 7. SwiGLU using secure_swiglu_mpc
        // 8. FFN down projection (linear - stays the same)
    }
}
```

#### Step 1.3: Pre-generate Beaver triples on server startup
**File:** `crates/v2/server/src/state.rs`

Add triple store initialization during model loading:
```rust
// Calculate triples needed per inference
let triples_per_layer = triples_needed_per_layer(hidden_dim, max_seq_len);
let store = create_inference_triple_store(num_layers, hidden_dim, max_seq_len);
```

---

### Part 2: V3-OT Integration (More Complex)

OT is used for **function table lookups** - the client looks up nonlinear function values without revealing which values it needs.

#### Step 2.1: Create OT-based function evaluation module
**File:** `crates/v2/sharing/src/secure_nonlinear_ot.rs` (new file)

```rust
use shardlm_ot::iknp::IknpOtExtension;
use super::secure_inference_ot::OtFunctionTables;

pub struct OtFunctionEvaluator {
    client_ot: IknpOtExtension,  // For client
    server_ot: IknpOtExtension,  // For server
    tables: &'static OtFunctionTables,
}

impl OtFunctionEvaluator {
    /// Evaluate SiLU on shares using OT table lookup
    pub fn silu_ot(
        &mut self,
        x_client: f32,
        x_server: f32,
        session_id: &[u8],
        ctr: u64,
    ) -> Result<(f32, f32), OtError> {
        // 1. Client reconstructs x = x_client + x_server
        // 2. Client computes table index for x
        // 3. Client generates OT query for this index
        // 4. Server processes query against silu table
        // 5. Client decodes to get silu(x)
        // 6. Client re-shares the result
    }
}
```

#### Step 2.2: Establish OT session per inference session
**File:** `crates/v2/server/src/routes/secure_inference_ot.rs`

```rust
// Per-session OT state (stored in SecureSession)
pub struct OtSession {
    server_ot: IknpOtExtension,
    base_ot_complete: bool,
}

// New endpoint for OT session initialization
pub async fn init_ot_session(...) -> Result<Json<OtSessionResponse>> {
    let mut server_ot = IknpOtExtension::new_server();
    // Base OT handshake
    let msg1 = ... // receive from client
    let msg2 = server_ot.process_base_ot_receiver(&msg1)?;
    // Return msg2 to client
}
```

#### Step 2.3: Replace simulated OT with real OT in prefill
**File:** `crates/v2/server/src/routes/secure_inference_ot.rs`

```rust
pub async fn ot_prefill(...) -> Result<Json<OtPrefillResponse>> {
    // For each nonlinear operation:
    // 1. Client sends OT query (indices encrypted)
    // 2. Server processes against function table
    // 3. Server returns masked table entries
    // 4. Client unmasks and re-shares

    // Linear operations remain unchanged (on GPU)
}
```

#### Step 2.4: Client-side OT integration
**File:** `crates/v2/client/src/ot_client.rs` (new file)

The client needs to:
1. Initialize OT session (base OT handshake)
2. For each nonlinear operation during inference:
   - Reconstruct x from shares
   - Generate OT query for table index
   - Receive and unmask server response
   - Re-share the result

---

### Part 3: Testing Strategy

#### 3.1 Unit Tests
- Verify MPC nonlinear operations produce correct output (within approximation error)
- Verify OT table lookups return correct values
- Verify security properties (server never sees plaintext in MPC/OT modes)

#### 3.2 Integration Tests
```rust
#[tokio::test]
async fn test_v3_mpc_inference_correctness() {
    // Run same prompt through V3 and V3-MPC
    // Verify outputs are similar (allow for approximation error)
}

#[tokio::test]
async fn test_v3_ot_inference_correctness() {
    // Run same prompt through V3 and V3-OT
    // Verify outputs match (OT should be exact)
}
```

#### 3.3 Security Verification Tests
```rust
#[test]
fn test_mpc_no_plaintext_reconstruction() {
    // Use SecurityAuditor to verify no plaintext leaks
}

#[test]
fn test_ot_indices_encrypted() {
    // Verify OT queries don't contain plaintext indices
}
```

---

## Estimated Effort

| Task | Complexity | Est. Time |
|------|------------|-----------|
| V3-MPC GPU wrappers | Medium | 2-3 hours |
| V3-MPC endpoint integration | Medium | 3-4 hours |
| V3-OT function evaluator | High | 4-5 hours |
| V3-OT session management | Medium | 2-3 hours |
| V3-OT endpoint integration | High | 4-5 hours |
| Client-side OT support | Medium | 3-4 hours |
| Testing & debugging | High | 5-6 hours |
| **Total** | | **23-30 hours** |

---

## Dependencies

1. **For V3-MPC:**
   - `shardlm-v2-sharing` crate (already has all MPC primitives)
   - Feature flag: `mpc-secure`

2. **For V3-OT:**
   - `shardlm-ot` crate (IKNP implementation)
   - Need to add dependency in `crates/v2/server/Cargo.toml`
   - Need to add dependency in `crates/v2/client/Cargo.toml`

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MPC polynomial approximations may have higher error than documented | Medium | Test extensively, document actual error bounds |
| OT overhead may be higher than expected | Medium | Batch OT queries, use async processing |
| GPU-CPU transfers for MPC may cause bottleneck | Medium | Consider keeping more operations on CPU or pipelining |
| Client needs significant changes for OT | High | Phase the rollout - MPC first, then OT |

---

## Recommended Approach

**Phase 1: V3-MPC (Simpler, Self-contained)**
1. Wire existing MPC primitives into the server endpoint
2. No client changes needed (shares work the same way)
3. Accept ~0.5-2% accuracy loss from polynomial approximations

**Phase 2: V3-OT (Requires client changes)**
1. Add OT session management
2. Implement client-side OT protocol
3. Achieve exact accuracy with OT table lookups

This phased approach allows incremental delivery and testing.
