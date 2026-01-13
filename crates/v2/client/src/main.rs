//! ShardLM V2 Headless Client CLI
//!
//! A command-line tool for running ShardLM V2 inference and benchmarks.
//!
//! # Usage
//!
//! ```bash
//! # Health check
//! shardlm-v2-client health --server-url http://localhost:9090
//!
//! # Single generation
//! shardlm-v2-client generate --server-url http://localhost:9090 --prompt "Hello"
//!
//! # Benchmark mode
//! shardlm-v2-client benchmark --server-url http://localhost:9090 --runs 10
//!
//! # Interactive chat
//! shardlm-v2-client chat --server-url http://localhost:9090
//! ```

use anyhow::Result;
use clap::{Parser, Subcommand};
use console::style;
use shardlm_v2_client::{BenchmarkConfig, BenchmarkRunner, ShardLmClient};
use std::io::{self, Write};

#[derive(Parser)]
#[command(name = "shardlm-v2-client")]
#[command(author = "ShardLM Contributors")]
#[command(version)]
#[command(about = "ShardLM V2 Headless Client for benchmarking and inference")]
#[command(long_about = r#"
ShardLM V2 Headless Client

A native Rust client for running privacy-preserving inference benchmarks.
Designed for H100 instances and paper artifact evaluation.

Examples:
  # Check server health
  shardlm-v2-client health -s http://localhost:9090

  # Run a single generation
  shardlm-v2-client generate -s http://localhost:9090 -p "Hello, world!"

  # Run benchmarks
  shardlm-v2-client benchmark -s http://localhost:9090 --runs 10 --warmup 2

  # Interactive chat
  shardlm-v2-client chat -s http://localhost:9090
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Check server health
    Health {
        /// Server URL
        #[arg(short, long, default_value = "http://localhost:9090")]
        server_url: String,
    },

    /// Get server information
    Info {
        /// Server URL
        #[arg(short, long, default_value = "http://localhost:9090")]
        server_url: String,
    },

    /// Generate text from a prompt
    Generate {
        /// Server URL
        #[arg(short, long, default_value = "http://localhost:9090")]
        server_url: String,

        /// Input prompt
        #[arg(short, long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short, long, default_value = "50")]
        max_tokens: usize,

        /// Temperature for sampling (0.0 = greedy)
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Show timing information
        #[arg(long, default_value = "false")]
        timing: bool,

        /// Path to tokenizer.json file (from model directory)
        #[arg(long)]
        tokenizer: Option<String>,
    },

    /// Run performance benchmarks
    Benchmark {
        /// Server URL
        #[arg(short, long, default_value = "http://localhost:9090")]
        server_url: String,

        /// Input prompt
        #[arg(short, long, default_value = "Hello, how are you?")]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(short, long, default_value = "50")]
        max_tokens: usize,

        /// Number of benchmark runs
        #[arg(short, long, default_value = "10")]
        runs: usize,

        /// Number of warmup runs
        #[arg(short, long, default_value = "2")]
        warmup: usize,

        /// Endpoint version (v2 or v3)
        #[arg(short, long, default_value = "v2")]
        endpoint: String,

        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<String>,

        /// Include raw timing data in output
        #[arg(long, default_value = "false")]
        raw: bool,
    },

    /// Interactive chat mode
    Chat {
        /// Server URL
        #[arg(short, long, default_value = "http://localhost:9090")]
        server_url: String,

        /// Maximum tokens to generate per response
        #[arg(short, long, default_value = "100")]
        max_tokens: usize,

        /// Temperature for sampling
        #[arg(short, long, default_value = "0.7")]
        temperature: f32,

        /// Path to tokenizer.json file (from model directory)
        #[arg(long)]
        tokenizer: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Health { server_url } => {
            run_health(&server_url).await?;
        }
        Commands::Info { server_url } => {
            run_info(&server_url).await?;
        }
        Commands::Generate {
            server_url,
            prompt,
            max_tokens,
            temperature,
            timing,
            tokenizer,
        } => {
            run_generate(&server_url, &prompt, max_tokens, temperature, timing, tokenizer.as_deref()).await?;
        }
        Commands::Benchmark {
            server_url,
            prompt,
            max_tokens,
            runs,
            warmup,
            endpoint,
            output,
            raw,
        } => {
            run_benchmark(
                &server_url,
                &prompt,
                max_tokens,
                runs,
                warmup,
                &endpoint,
                output,
                raw,
            )
            .await?;
        }
        Commands::Chat {
            server_url,
            max_tokens,
            temperature,
            tokenizer,
        } => {
            run_chat(&server_url, max_tokens, temperature, tokenizer.as_deref()).await?;
        }
    }

    Ok(())
}

async fn run_health(server_url: &str) -> Result<()> {
    println!("Checking server health at {}...", server_url);

    let client = ShardLmClient::new(server_url);
    match client.health_check().await {
        Ok(true) => {
            println!("{} Server is healthy", style("[OK]").green().bold());
        }
        Ok(false) => {
            println!("{} Server returned unhealthy status", style("[WARN]").yellow().bold());
        }
        Err(e) => {
            println!("{} Failed to connect: {}", style("[ERROR]").red().bold(), e);
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_info(server_url: &str) -> Result<()> {
    println!("Fetching server info from {}...\n", server_url);

    let mut client = ShardLmClient::new(server_url);
    match client.get_info().await {
        Ok(info) => {
            println!("Server Information:");
            println!("  Version:      {}", info.version);
            println!("  Model:        {}", info.model_name);
            println!("  Hidden Dim:   {}", info.hidden_dim);
            println!("  Layers:       {}", info.num_layers);
            println!("  Heads:        {}", info.num_heads);
            println!("  Vocab Size:   {}", info.vocab_size);
            println!("  CUDA:         {}", if info.cuda_available { "Yes" } else { "No" });
            if let Some(gpu) = info.gpu_name {
                println!("  GPU:          {}", gpu);
            }
        }
        Err(e) => {
            println!("{} Failed to get info: {}", style("[ERROR]").red().bold(), e);
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_generate(
    server_url: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    show_timing: bool,
    tokenizer_path: Option<&str>,
) -> Result<()> {
    println!("Generating from: \"{}\"", prompt);
    println!("Server: {}", server_url);
    println!();

    let mut client = ShardLmClient::new(server_url);

    // Load tokenizer if path provided
    if let Some(path) = tokenizer_path {
        client.load_tokenizer(path)?;
        println!("Loaded tokenizer from: {}", path);
    }

    match client.generate(prompt, max_tokens, temperature).await {
        Ok(result) => {
            println!("{} {}", style("Generated:").cyan().bold(), result.text);

            if show_timing {
                println!();
                println!("Timing:");
                println!("  Embedding:  {:>8.1} ms", result.timing.embedding_ms);
                println!("  Prefill:    {:>8.1} ms", result.timing.prefill_ms);
                println!("  Decode:     {:>8.1} ms", result.timing.decode_ms);
                println!("  Total:      {:>8.1} ms", result.timing.total_ms);
                println!("  Tokens:     {}", result.timing.tokens_generated);
            }
        }
        Err(e) => {
            println!("{} Generation failed: {}", style("[ERROR]").red().bold(), e);
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_benchmark(
    server_url: &str,
    prompt: &str,
    max_tokens: usize,
    runs: usize,
    warmup: usize,
    endpoint: &str,
    output: Option<String>,
    include_raw: bool,
) -> Result<()> {
    println!("{}", style("ShardLM V2 Benchmark").cyan().bold());
    println!("========================================\n");

    let client = ShardLmClient::new(server_url);
    let config = BenchmarkConfig {
        prompt: prompt.to_string(),
        prompt_tokens: 0, // Will be computed
        max_new_tokens: max_tokens,
        temperature: 0.7,
        runs,
        warmup,
        endpoint: endpoint.to_string(),
    };

    let mut runner = BenchmarkRunner::new(client, config);

    match runner.run().await {
        Ok(mut result) => {
            // Clear raw timings if not requested
            if !include_raw {
                result.raw_timings = None;
            }

            result.print_summary();

            // Save to file if requested
            if let Some(path) = output {
                result.save(&path)?;
                println!("Results saved to: {}", path);
            }
        }
        Err(e) => {
            println!("{} Benchmark failed: {}", style("[ERROR]").red().bold(), e);
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_chat(server_url: &str, max_tokens: usize, temperature: f32, tokenizer_path: Option<&str>) -> Result<()> {
    println!("{}", style("ShardLM V2 Interactive Chat").cyan().bold());
    println!("Type 'quit' or 'exit' to end the session.\n");

    let mut client = ShardLmClient::new(server_url);

    // Load tokenizer if path provided
    if let Some(path) = tokenizer_path {
        client.load_tokenizer(path)?;
        println!("Loaded tokenizer from: {}", path);
    }

    // Create session
    match client.create_session().await {
        Ok(session) => {
            println!("Session created: {}", session.session_id);
            println!("Model: {}", session.model_name);
            println!();
        }
        Err(e) => {
            println!("{} Failed to create session: {}", style("[ERROR]").red().bold(), e);
            std::process::exit(1);
        }
    }

    loop {
        print!("{} ", style("You:").green().bold());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }

        match client.generate(input, max_tokens, temperature).await {
            Ok(result) => {
                println!("{} {}", style("Assistant:").blue().bold(), result.text);
                println!(
                    "  {} ({}ms)",
                    style(format!("[{} tokens]", result.timing.tokens_generated)).dim(),
                    result.timing.total_ms as u64
                );
                println!();
            }
            Err(e) => {
                println!("{} {}", style("[Error]").red().bold(), e);
            }
        }
    }

    Ok(())
}
