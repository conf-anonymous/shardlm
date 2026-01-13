//! Example: Run inference on Llama 70B across 4 GPUs
//!
//! This example demonstrates:
//! 1. Loading sharded INT8 weights (from cache if available)
//! 2. Creating a DistributedEngine for tensor-parallel inference
//! 3. Running a forward pass and generating text
//!
//! Run with:
//!   cargo run -p shardlm-v2-model --features cuda --release --example inference -- /path/to/model
//!
//! Requirements:
//! - 4x GPUs with ~24GB VRAM each (e.g., A10G, RTX 3090)
//! - Llama 70B weights in safetensor format (~140GB)
//! - First run: Quantization takes ~20-30min, saves to cache
//! - Subsequent runs: Loads from cache in ~30-60s

use std::env;
use std::time::Instant;

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_model::{DistributedEngine, ShardedModelLoader, Tokenizer};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <model_dir> [prompt]", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  model_dir  Path to Llama 70B weights (safetensor format)");
        eprintln!("  prompt     Optional prompt (default: 'Hello, I am')");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} /data/llama-70b-instruct-weights 'What is 2+2?'", args[0]);
        std::process::exit(1);
    }

    let model_dir = &args[1];
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello, I am");
    let num_gpus: usize = 4;

    println!("=============================================");
    println!("  ShardLM v2 - Distributed Inference Example");
    println!("=============================================\n");

    println!("Model directory: {}", model_dir);
    println!("Prompt: \"{}\"", prompt);
    println!("Number of GPUs: {}", num_gpus);
    println!();

    // Step 1: Load tokenizer
    println!("[1/4] Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&format!("{}/tokenizer.json", model_dir))?;
    println!("  Vocabulary size: {}", tokenizer.vocab_size());

    // Step 2: Create sharded model loader
    println!("\n[2/4] Loading sharded model weights...");
    let load_start = Instant::now();

    let mut loader = ShardedModelLoader::new_int8(model_dir, ModelArchitecture::Llama3_1_70B, num_gpus)?;

    println!("  Cache exists: {}", loader.has_cache());
    if !loader.has_cache() {
        println!("  First run: will quantize weights and save to cache (~20-30min)");
    }

    // Load weights (from cache or fresh)
    if !loader.has_cache() {
        loader.load_all_files()?;
    }
    let weights = loader.load_all_weights()?;

    println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    // Step 3: Create distributed engine
    println!("\n[3/4] Creating distributed inference engine...");
    let config = loader.config().clone();
    let gpu_ctx = shardlm_v2_core::gpu::MultiGpuContext::new(num_gpus)?;

    let mut engine = DistributedEngine::new(config.clone(), weights, gpu_ctx)?;

    println!("  Engine ready: {:?}", engine);
    println!("  Layers: {}", engine.config().num_layers);
    println!("  Heads per GPU: {}", engine.config().num_heads / num_gpus);

    // Step 4: Run inference
    println!("\n[4/4] Running inference...");

    // Tokenize prompt
    let token_ids = tokenizer.encode(prompt, true)?;
    println!("  Prompt tokens: {:?}", token_ids);

    // Generate
    let gen_start = Instant::now();
    let max_new_tokens = 32;
    let temperature = 0.7;

    #[cfg(feature = "cuda")]
    {
        println!("  Generating {} tokens with temperature {}...", max_new_tokens, temperature);

        let generated_ids = engine.generate(&token_ids, max_new_tokens, temperature)?;

        let gen_time = gen_start.elapsed();
        let new_tokens = generated_ids.len() - token_ids.len();
        let tokens_per_sec = new_tokens as f64 / gen_time.as_secs_f64();

        // Decode output
        let output = tokenizer.decode(&generated_ids, true)?;

        println!();
        println!("=============================================");
        println!("  Generation Complete");
        println!("=============================================");
        println!("Prompt: {}", prompt);
        println!("Output: {}", output);
        println!();
        println!("Stats:");
        println!("  New tokens: {}", new_tokens);
        println!("  Generation time: {:.2}s", gen_time.as_secs_f64());
        println!("  Tokens/sec: {:.2}", tokens_per_sec);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (engine, gen_start, max_new_tokens, temperature, token_ids, tokenizer);
        println!("Note: CUDA feature not enabled. Skipping inference.");
        println!("Rebuild with: cargo run -p shardlm-v2-model --features cuda --example inference");
    }

    Ok(())
}
