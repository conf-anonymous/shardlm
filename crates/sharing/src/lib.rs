//! ShardLM Secret Sharing
//!
//! Additive secret sharing primitives for secure computation.
//! X = X_c + X_s where client holds X_c and server holds X_s.

mod error;
mod share;
mod matrix;
mod linear;
mod kv_cache;
mod attention;
mod projection;
mod ffn;
mod rmsnorm;
mod rope;
mod transformer;

#[cfg(test)]
mod linear_adversarial_tests;

pub use error::{SharingError, Result};
pub use share::{Share, SharePair};
pub use matrix::{SharedMatrix, SharedVector};
pub use linear::{
    LinearClient, LinearServer, LinearRequestMsg, LinearResponseMsg,
    secure_linear, plaintext_linear,
    secure_linear_batch, plaintext_linear_batch,
    secure_linear_batch_timed, compare_batch_outputs,
    BatchLinearResult,
    // GEMM-based batch operations
    secure_linear_gemm, plaintext_linear_gemm,
    secure_linear_gemm_timed, GemmBatchResult,
    // Hybrid GEMM/matvec strategy
    GEMM_CROSSOVER_L,
    secure_linear_hybrid, secure_linear_hybrid_timed, HybridLinearResult,
    // Precomputed transpose for GEMM
    LinearWeights, secure_linear_gemm_pretransposed, secure_linear_hybrid_pretransposed,
};
pub use kv_cache::{KvCacheClient, KvCacheServer, share_kv, reconstruct_kv};
pub use attention::{
    AttentionClient, AttentionServer, AttentionOutput,
    compute_attention, compute_attention_with_rope,
};
pub use projection::{
    ProjectionClient, ProjectionServer, QkvProjection, OutputProjection,
    compute_qkv_projection, compute_output_projection,
};
pub use ffn::{
    FfnClient, FfnServer, FfnOutput,
    compute_ffn,
};
pub use rmsnorm::{
    RmsNormClient, RmsNormConfig,
    compute_rmsnorm,
};
pub use rope::{
    RopeFrequencies,
    apply_rope, apply_rope_to_q, apply_rope_to_k,
    apply_rope_to_q_shared, apply_rope_to_k_shared,
};
pub use transformer::{
    TransformerLayerClient, TransformerLayerServer, TransformerLayerConfig,
    TransformerLayerOutput, PrivateDecoder,
};
