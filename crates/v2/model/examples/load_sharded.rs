//! Example: Load Llama 70B model across 4 GPUs with tensor parallelism + INT8 quantization
//!
//! This example demonstrates how to:
//! 1. Initialize multi-GPU context
//! 2. Create a sharding plan for Llama 70B
//! 3. Load model weights with INT8 quantization (50% memory reduction)
//! 4. Shard weights across GPUs with tensor parallelism
//! 5. **Weight caching for fast subsequent loads (~30-60s vs ~20-30min)**
//!
//! Run with:
//!   cargo run -p shardlm-v2-model --features cuda --release --example load_sharded -- /path/to/model
//!
//! Requirements:
//! - 4x GPUs with ~24GB VRAM each (e.g., A10G, RTX 3090)
//! - Llama 70B weights in safetensor format (~140GB)
//!
//! Memory usage with INT8:
//! - Original BF16: ~140GB total, ~35GB per GPU (OOM on A10G)
//! - With INT8: ~70GB total, ~17.5GB per GPU (fits on A10G!)
//!
//! Weight caching:
//! - First load: Quantizes weights and saves to .shardlm-cache/ (~20-30min)
//! - Subsequent loads: Reads from cache, bypassing quantization (~30-60s)

use std::env;
use std::time::Instant;

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_model::ShardedModelLoader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [num_gpus]", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  model_dir  Path to Llama 70B weights (safetensor format)");
        eprintln!("  num_gpus   Number of GPUs to use (default: 4)");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} /data/llama-70b-instruct-weights 4", args[0]);
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let num_gpus: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4);

    println!("==============================================");
    println!("  ShardLM v2 - Sharded Model Loading Example");
    println!("  INT8 Quantization + Tensor Parallelism + Caching");
    println!("==============================================\n");

    println!("Model directory: {}", model_dir);
    println!("Number of GPUs: {}", num_gpus);
    println!();

    // Create sharded model loader with INT8 quantization for 50% memory reduction
    println!("[1/5] Creating sharded model loader (INT8 quantization)...");
    let start = Instant::now();

    // Use new_int8() for INT8 quantization - critical for fitting on 4x A10G
    let mut loader = ShardedModelLoader::new_int8(model_dir, ModelArchitecture::Llama3_1_70B, num_gpus)?;

    println!("  Created in {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    println!("  Quantization: {:?}", loader.quantization());
    println!("  Cache directory: {}", loader.cache_dir().display());
    println!("  Cache exists: {}", loader.has_cache());
    println!();

    // Print model configuration
    let config = loader.config();
    println!("[2/5] Model Configuration:");
    println!("  Hidden dim:       {}", config.hidden_dim);
    println!("  Attention heads:  {} (Q) / {} (KV)", config.num_heads, config.num_kv_heads);
    println!("  Layers:           {}", config.num_layers);
    println!("  Vocabulary:       {}", config.vocab_size);
    println!("  Intermediate:     {}", config.intermediate_dim);
    println!("  Max context:      {}", config.max_seq_len);
    println!();

    // Print sharding plan
    let plan = loader.sharding_plan();
    println!("[3/5] Sharding Plan:");
    println!("  GPUs:             {}", plan.num_gpus);
    println!("  Architecture:     {:?}", plan.architecture);
    println!("  Memory per GPU:   {:?} GB", plan.memory_per_gpu_gb());
    println!();

    // Load safetensor files (only needed for first load, skip if loading from cache)
    if !loader.has_cache() {
        println!("[4/5] Memory-mapping model files (needed for first load)...");
        let start = Instant::now();

        loader.load_all_files()?;

        let load_time = start.elapsed();
        let total_size = loader.total_size_gb();
        let throughput = total_size / load_time.as_secs_f64();

        println!();
        println!("Memory-mapped {} safetensor files", loader.list_safetensor_files()?.len());
        println!("Total size: {:.2} GB", total_size);
        println!("Load time: {:.2}s ({:.2} GB/s)", load_time.as_secs_f64(), throughput);
        println!();
    } else {
        println!("[4/5] Skipping safetensor loading - using cache instead");
        println!();
    }

    // Now load weights into GPU memory with sharding
    #[cfg(feature = "cuda")]
    {
        println!("[5/5] Loading weights into GPU memory...");
        let start = Instant::now();

        let weights = loader.load_all_weights()?;

        let gpu_load_time = start.elapsed();
        println!();
        println!("GPU loading complete!");
        println!("  Layers loaded:      {}", weights.layers.len());
        println!("  Embed tokens shards: {}", weights.embed_tokens.len());
        println!("  LM head shards:      {}", weights.lm_head.len());
        println!("  GPU load time:       {:.2}s", gpu_load_time.as_secs_f64());
        println!();

        // Print GPU memory usage
        let gpu_ctx = loader.gpu_ctx();
        println!("GPU Memory Usage:");
        for i in 0..num_gpus {
            let device = gpu_ctx.device(i);
            let (used, total) = device.memory_info();
            let used_gb = used as f64 / (1024.0 * 1024.0 * 1024.0);
            let total_gb = total as f64 / (1024.0 * 1024.0 * 1024.0);
            println!(
                "  GPU {}: {:.2} GB / {:.2} GB ({:.1}%)",
                i,
                used_gb,
                total_gb,
                (used as f64 / total as f64) * 100.0
            );
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("Note: CUDA feature not enabled. Skipping GPU loading.");
        println!("Rebuild with: cargo run -p shardlm-v2-model --features cuda --example load_sharded");
    }

    println!();
    println!("Model loading complete!");
    println!("==============================================");

    Ok(())
}
