//! Example: Load Llama 70B with INT8 quantization + pipeline parallelism
//!
//! This example demonstrates memory-optimized loading for 4x A10G (24GB each):
//! - INT8 quantization: 50% memory reduction
//! - Pipeline parallelism: 20 layers per GPU
//! - Tensor parallelism: weights sharded across GPUs
//!
//! Run with:
//!   cargo run -p shardlm-v2-model --features cuda --example load_optimized -- /path/to/model
//!
//! Requirements:
//! - 4x GPUs with ~24GB VRAM each (e.g., A10G, RTX 3090)
//! - Llama 70B weights in safetensor format (~140GB)

use std::env;
use std::time::Instant;

use shardlm_v2_core::config::ModelArchitecture;
use shardlm_v2_core::memory::{MemoryConfig, MemoryEstimator};
use shardlm_v2_model::OptimizedModelLoader;

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

    println!("================================================");
    println!("  ShardLM v2 - Optimized Model Loading Example");
    println!("  INT8 Quantization + Pipeline Parallelism");
    println!("================================================\n");

    println!("Model directory: {}", model_dir);
    println!("Number of GPUs: {}", num_gpus);
    println!();

    // Create memory configuration for the target hardware
    let memory_config = if num_gpus == 4 {
        println!("Using optimized config for 4x A10G (24GB each):");
        MemoryConfig::for_4x_a10g()
    } else if num_gpus == 2 {
        println!("Using optimized config for 2x H100 (80GB each):");
        MemoryConfig::for_2x_h100()
    } else {
        println!("Using custom config for {} GPUs:", num_gpus);
        MemoryConfig {
            pipeline_stages: num_gpus,
            pipeline_parallel: true,
            quantization: shardlm_v2_core::quantization::QuantizationMode::Int8PerChannel,
            ..Default::default()
        }
    };

    println!("  Quantization:      {:?}", memory_config.quantization);
    println!("  Pipeline parallel: {}", memory_config.pipeline_parallel);
    println!("  Pipeline stages:   {}", memory_config.pipeline_stages);
    println!();

    // Show memory estimates before loading
    println!("[1/4] Memory Estimation:");
    let estimator = MemoryEstimator::new(ModelArchitecture::Llama3_1_70B, memory_config.clone());
    let summary = estimator.summary(num_gpus);
    println!("  Total weight memory: {:.2} GB", summary.total_weight_memory_gb);
    println!("  Per-GPU memory:      {:.2} GB", summary.per_gpu_memory_gb);
    println!("  Usable GPU memory:   {:.2} GB", summary.usable_gpu_memory_gb);
    println!("  Fits in memory:      {}", if summary.fits { "YES ✓" } else { "NO ✗" });
    println!();

    if !summary.fits {
        eprintln!("WARNING: Model may not fit in GPU memory!");
        eprintln!("Consider enabling CPU offloading or using larger GPUs.");
    }

    // Create optimized model loader
    println!("[2/4] Creating optimized model loader...");
    let start = Instant::now();

    let mut loader = OptimizedModelLoader::new(
        model_dir,
        ModelArchitecture::Llama3_1_70B,
        memory_config,
    )?;

    println!("  Created in {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    println!();

    // Load safetensor files (memory-mapped)
    println!("[3/4] Memory-mapping model files...");
    let start = Instant::now();

    loader.load_all_files()?;

    let mmap_time = start.elapsed();
    println!("  Memory-mapped in {:.2}s", mmap_time.as_secs_f64());
    println!();

    // Load weights with INT8 quantization
    println!("[4/4] Loading weights with INT8 quantization...");
    let start = Instant::now();

    let weights = loader.load_all_weights()?;

    let load_time = start.elapsed();
    println!();
    println!("Loading complete!");
    println!("  Layers loaded:       {}", weights.layers.len());
    println!("  Offloaded layers:    {}", weights.num_offloaded_layers());
    println!("  Total weight memory: {:.2} GB", weights.total_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("  Load time:           {:.2}s", load_time.as_secs_f64());
    println!();

    // Print per-layer info
    println!("Layer Distribution (Pipeline Parallelism):");
    if let Some(ref pipeline) = weights.pipeline_config {
        for stage in 0..pipeline.num_stages {
            let layers = &pipeline.stage_layers[stage];
            let gpu = pipeline.stage_gpus[stage];
            println!("  GPU {}: layers {:?}", gpu,
                if layers.len() > 4 {
                    format!("{}..{} ({} layers)", layers[0], layers[layers.len()-1], layers.len())
                } else {
                    format!("{:?}", layers)
                });
        }
    }
    println!();

    println!("================================================");
    println!("  Model ready for inference!");
    println!("================================================");

    Ok(())
}
