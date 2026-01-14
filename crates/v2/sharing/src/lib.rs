//! ShardLM v2 Secret Sharing - CUDA-accelerated operations
//!
//! This crate provides GPU-accelerated secret sharing for privacy-preserving
//! inference with Llama 70B models.
//!
//! # CRITICAL SECURITY INVARIANT
//!
//! **THE SERVER MUST NEVER RECONSTRUCT PLAINTEXT USER DATA.**
//!
//! This crate enforces this through Rust's type system:
//! - `SecureShare<Server, T>` cannot be reconstructed alone
//! - `SecureSharePair<T>` (required for reconstruction) can only exist on client
//! - `ServerContext` provides server-safe computation methods
//!
//! # Key Features
//!
//! - **Type-safe shares**: Compile-time prevention of server-side reconstruction
//! - **GPU-accelerated operations**: Parallelized across CUDA cores
//! - **Secure linear layers**: Server computes on shares, never plaintext
//! - **Client-side nonlinear ops**: Softmax, SiLU, RMSNorm on client only

pub mod attention;
pub mod beaver;
pub mod error;
pub mod secure;
pub mod secure_attention;
pub mod secure_ffn;
pub mod secure_linear;
pub mod secure_nonlinear;
pub mod secure_nonlinear_mpc;
pub mod secure_polynomial;
#[cfg(feature = "cuda")]
pub mod secure_nonlinear_gpu;
#[cfg(feature = "cuda")]
pub mod secure_nonlinear_mpc_gpu;
pub mod secure_ot;
pub mod secure_nonlinear_ot;
pub mod security_assertions;
pub mod share;

pub use attention::GpuAttention;
pub use error::{Result, SharingError};
pub use secure::{
    ClientShare, SecureShare, SecureSharePair, ServerComputeResult, ServerContext, ServerShare,
};
pub use secure_attention::SecureAttention;
pub use secure_ffn::SecureFFN;
pub use secure_linear::SecureLinear;

// GPU-accelerated secure operations
#[cfg(feature = "cuda")]
pub use secure_attention::gpu::{GpuSecureAttention, QkvTensorResult};
#[cfg(feature = "cuda")]
pub use secure_ffn::gpu::{GpuSecureFFN, GateUpTensorResult};
#[cfg(feature = "cuda")]
pub use secure_linear::gpu::GpuSecureLinear;
pub use secure_nonlinear::{
    secure_attention_approx, secure_rms_norm_approx, secure_silu_approx,
    secure_softmax_approx, secure_swiglu_approx,
};
pub use beaver::{BeaverTriple, BeaverTripleStore, secure_multiply_mpc};
pub use secure_polynomial::{secure_polynomial_eval, secure_silu_mpc, secure_exp_mpc, secure_rsqrt_mpc};
pub use secure_nonlinear_mpc::{
    secure_silu_mpc_batch, secure_rms_norm_mpc, secure_swiglu_mpc, secure_softmax_mpc,
    create_inference_triple_store, triples_needed_per_layer,
};
#[cfg(feature = "cuda")]
pub use secure_nonlinear_gpu::{
    secure_silu_gpu, secure_rms_norm_gpu, secure_mul_gpu,
    secure_swiglu_gpu, secure_attention_gpu, secure_add_gpu,
    secure_rms_norm_gpu_vec, secure_swiglu_gpu_vec,
};
#[cfg(feature = "cuda")]
pub use secure_nonlinear_mpc_gpu::{
    secure_rms_norm_mpc_gpu, secure_swiglu_mpc_gpu, secure_softmax_mpc_gpu,
    secure_silu_mpc_gpu, triples_for_layer_gpu, MpcInferenceState,
    secure_rms_norm_mpc_gpu_vec, secure_swiglu_mpc_gpu_vec, secure_softmax_mpc_gpu_vec,
};
pub use secure_ot::{OtEmbeddingResult, SecureOtServer};
pub use secure_nonlinear_ot::{
    OtFunctionTable, OtFunctionEvaluator, OtFunctionServer, StandardOtTables,
    secure_silu_ot, secure_rms_norm_ot, secure_swiglu_ot, secure_softmax_ot,
};
pub use security_assertions::{SecurityAuditor, SecurityViolation};
pub use share::{Share, SharePair};
