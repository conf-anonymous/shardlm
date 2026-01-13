//! Benchmark runner for ShardLM V2 performance measurement
//!
//! Produces structured JSON output suitable for paper artifact.

use crate::inference::GenerationTiming;
use crate::{Result, ShardLmClient};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Prompt to use for benchmarking
    pub prompt: String,
    /// Number of tokens in prompt (computed)
    #[serde(default)]
    pub prompt_tokens: usize,
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Number of benchmark runs
    pub runs: usize,
    /// Number of warmup runs (excluded from results)
    pub warmup: usize,
    /// Endpoint to use (v1, v2, v3)
    #[serde(default = "default_endpoint")]
    pub endpoint: String,
}

fn default_endpoint() -> String {
    "v2".to_string()
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            prompt: "Hello, how are you?".to_string(),
            prompt_tokens: 0,
            max_new_tokens: 50,
            temperature: 0.7,
            runs: 10,
            warmup: 2,
            endpoint: "v2".to_string(),
        }
    }
}

/// Statistical summary of timing measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingStats {
    /// Mean value in milliseconds
    pub mean_ms: f64,
    /// Standard deviation in milliseconds
    pub std_ms: f64,
    /// Minimum value in milliseconds
    pub min_ms: f64,
    /// Maximum value in milliseconds
    pub max_ms: f64,
    /// 50th percentile (median) in milliseconds
    pub p50_ms: f64,
    /// 95th percentile in milliseconds
    pub p95_ms: f64,
    /// 99th percentile in milliseconds
    pub p99_ms: f64,
}

impl TimingStats {
    /// Compute statistics from a list of timing values (in ms)
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean_ms: 0.0,
                std_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
            };
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        Self {
            mean_ms: mean,
            std_ms: std,
            min_ms: min,
            max_ms: max,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
        }
    }
}

/// Compute percentile from sorted values
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Benchmark results for a single phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseResults {
    /// Embedding fetch timing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<TimingStats>,
    /// Prefill timing (prompt processing)
    pub prefill: TimingStats,
    /// Decode timing (token generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<TimingStats>,
    /// Total end-to-end timing
    pub total: TimingStats,
    /// Tokens per second (for generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
}

/// Complete benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Model name
    pub model: String,
    /// GPU name
    pub gpu: String,
    /// ShardLM version (v1, v2, v3)
    pub version: String,
    /// Endpoint used
    pub endpoint: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Benchmark configuration
    pub config: BenchmarkConfig,
    /// Results
    pub results: PhaseResults,
    /// Raw timing data (for detailed analysis)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_timings: Option<Vec<GenerationTiming>>,
}

impl BenchmarkResult {
    /// Save to JSON file
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)
    }

    /// Print summary to console
    pub fn print_summary(&self) {
        println!("\n========================================");
        println!("Benchmark Results");
        println!("========================================");
        println!("Model:     {}", self.model);
        println!("GPU:       {}", self.gpu);
        println!("Version:   {}", self.version);
        println!("Endpoint:  {}", self.endpoint);
        println!("Timestamp: {}", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
        println!();
        println!("Configuration:");
        println!("  Prompt tokens:   {}", self.config.prompt_tokens);
        println!("  Max new tokens:  {}", self.config.max_new_tokens);
        println!("  Temperature:     {:.2}", self.config.temperature);
        println!("  Runs:            {}", self.config.runs);
        println!("  Warmup:          {}", self.config.warmup);
        println!();
        println!("Prefill (prompt processing):");
        println!("  Mean:   {:>8.1} ms", self.results.prefill.mean_ms);
        println!("  Std:    {:>8.1} ms", self.results.prefill.std_ms);
        println!("  Min:    {:>8.1} ms", self.results.prefill.min_ms);
        println!("  Max:    {:>8.1} ms", self.results.prefill.max_ms);
        println!("  P50:    {:>8.1} ms", self.results.prefill.p50_ms);
        println!("  P95:    {:>8.1} ms", self.results.prefill.p95_ms);
        println!("  P99:    {:>8.1} ms", self.results.prefill.p99_ms);
        println!();
        println!("Total (end-to-end):");
        println!("  Mean:   {:>8.1} ms", self.results.total.mean_ms);
        println!("  Std:    {:>8.1} ms", self.results.total.std_ms);
        println!("  P95:    {:>8.1} ms", self.results.total.p95_ms);
        if let Some(tps) = self.results.tokens_per_second {
            println!("  Tokens/sec: {:.1}", tps);
        }
        println!("========================================\n");
    }
}

/// Benchmark runner
pub struct BenchmarkRunner {
    client: ShardLmClient,
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(client: ShardLmClient, config: BenchmarkConfig) -> Self {
        Self { client, config }
    }

    /// Run the benchmark
    pub async fn run(&mut self) -> Result<BenchmarkResult> {
        // Get server info
        let info = self.client.get_info().await?;
        let gpu = info.gpu_name.unwrap_or_else(|| "Unknown GPU".to_string());

        // Create session
        self.client.create_session().await?;

        // Tokenize prompt to get token count
        let token_ids = self.client.simple_tokenize(&self.config.prompt);
        let mut config = self.config.clone();
        config.prompt_tokens = token_ids.len();

        println!("Starting benchmark...");
        println!("  Server:  {}", self.client.server_url());
        println!("  Model:   {}", info.model_name);
        println!("  GPU:     {}", gpu);
        println!("  Prompt:  {} tokens", config.prompt_tokens);
        println!("  Runs:    {} (+ {} warmup)", config.runs, config.warmup);
        println!();

        // Warmup runs
        if config.warmup > 0 {
            println!("Running {} warmup iterations...", config.warmup);
            for i in 0..config.warmup {
                print!("  Warmup {}/{}\r", i + 1, config.warmup);
                let _ = self.run_single_prefill().await?;
            }
            println!("  Warmup complete.        ");
        }

        // Benchmark runs
        println!("Running {} benchmark iterations...", config.runs);
        let mut timings = Vec::with_capacity(config.runs);
        let progress = indicatif::ProgressBar::new(config.runs as u64);
        progress.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for _ in 0..config.runs {
            let timing = self.run_single_prefill().await?;
            timings.push(timing);
            progress.inc(1);
        }
        progress.finish_with_message("Done");

        // Compute statistics
        let embedding_times: Vec<f64> = timings.iter().map(|t| t.embedding_ms).collect();
        let prefill_times: Vec<f64> = timings.iter().map(|t| t.prefill_ms).collect();
        let total_times: Vec<f64> = timings.iter().map(|t| t.total_ms).collect();

        // Compute tokens per second (prefill throughput)
        let prefill_stats = TimingStats::from_values(&prefill_times);
        let tokens_per_second = if prefill_stats.mean_ms > 0.0 {
            (config.prompt_tokens as f64) / (prefill_stats.mean_ms / 1000.0)
        } else {
            0.0
        };

        let results = PhaseResults {
            embedding: Some(TimingStats::from_values(&embedding_times)),
            prefill: prefill_stats,
            decode: None, // Prefill-only benchmark
            total: TimingStats::from_values(&total_times),
            tokens_per_second: Some(tokens_per_second),
        };

        // Determine the correct endpoint path based on version
        let endpoint_path = match config.endpoint.as_str() {
            "v3" => "/v2/secure/gpu/prefill_v3".to_string(),
            "v3-cc" => "/v3/cc/prefill".to_string(),
            "v3-mpc" => "/v3/mpc/prefill".to_string(),
            "v3-ot" => "/v3/ot/prefill".to_string(),
            _ => format!("/v2/secure/gpu/prefill_{}", config.endpoint),
        };

        let benchmark_result = BenchmarkResult {
            model: info.model_name,
            gpu,
            version: config.endpoint.clone(),
            endpoint: endpoint_path,
            timestamp: Utc::now(),
            config,
            results,
            raw_timings: Some(timings),
        };

        Ok(benchmark_result)
    }

    /// Run a single prefill benchmark iteration
    async fn run_single_prefill(&mut self) -> Result<GenerationTiming> {
        let token_ids = self.client.simple_tokenize(&self.config.prompt);

        let mut timing = GenerationTiming::default();
        let start = Instant::now();

        // Fetch embeddings
        let embed_start = Instant::now();
        let embeddings = self.client.fetch_embeddings(&token_ids).await?;
        timing.embedding_ms = embed_start.elapsed().as_secs_f64() * 1000.0;

        // Prefill based on endpoint version
        let prefill_start = Instant::now();
        match self.config.endpoint.as_str() {
            "v3" => {
                let _ = self
                    .client
                    .prefill_v3(&embeddings.client, &embeddings.server)
                    .await?;
            }
            "v3-cc" => {
                let _ = self
                    .client
                    .prefill_v3_cc(&embeddings.client, &embeddings.server)
                    .await?;
            }
            "v3-mpc" => {
                let _ = self
                    .client
                    .prefill_v3_mpc(&embeddings.client, &embeddings.server)
                    .await?;
            }
            "v3-ot" => {
                let _ = self
                    .client
                    .prefill_v3_ot(&embeddings.client, &embeddings.server)
                    .await?;
            }
            _ => {
                // Default to v2
                let _ = self
                    .client
                    .prefill_v2(&embeddings.client, &embeddings.server)
                    .await?;
            }
        }
        timing.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        timing.total_ms = start.elapsed().as_secs_f64() * 1000.0;
        timing.tokens_generated = token_ids.len();

        Ok(timing)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_stats() {
        let values = vec![100.0, 110.0, 105.0, 95.0, 108.0];
        let stats = TimingStats::from_values(&values);

        assert!((stats.mean_ms - 103.6).abs() < 0.1);
        assert!(stats.min_ms == 95.0);
        assert!(stats.max_ms == 110.0);
    }

    #[test]
    fn test_percentile() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!((percentile(&sorted, 50.0) - 5.0).abs() < 0.1);
        assert!((percentile(&sorted, 95.0) - 10.0).abs() < 0.1);
    }
}
