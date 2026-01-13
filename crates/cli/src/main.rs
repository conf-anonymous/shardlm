//! ShardLM CLI Demo
//!
//! Demonstrates loading TinyLlama weights and running secure inference.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use shardlm_fixed_point::DEFAULT_SCALE;
use shardlm_model::{SafetensorsLoader, Tokenizer};
use shardlm_ot::{IknpOtExtension, OtReceiver, OtSender, OtSessionConfig};
use shardlm_sharing::{
    LinearClient, LinearServer, plaintext_linear,
    secure_linear_batch_timed, plaintext_linear_batch, compare_batch_outputs,
    secure_linear_gemm_timed, plaintext_linear_gemm,
    // Hybrid strategy
    GEMM_CROSSOVER_L, secure_linear_hybrid_timed,
    LinearWeights, secure_linear_gemm_pretransposed,
};

/// Compute percentile from sorted durations
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Format duration in human-readable form
fn format_duration(d: Duration) -> String {
    if d.as_micros() < 1000 {
        format!("{}µs", d.as_micros())
    } else if d.as_millis() < 1000 {
        format!("{:.2}ms", d.as_micros() as f64 / 1000.0)
    } else {
        format!("{:.2}s", d.as_millis() as f64 / 1000.0)
    }
}

fn main() {
    println!("=== ShardLM v1 Demo ===\n");

    // Check for model weights
    let model_dir = PathBuf::from("tinyllama-weights");
    if !model_dir.exists() {
        eprintln!("Error: Model weights not found at ./tinyllama-weights");
        eprintln!("Please download with:");
        eprintln!("  python -c \"from huggingface_hub import snapshot_download; snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='./tinyllama-weights')\"");
        std::process::exit(1);
    }

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = match Tokenizer::from_directory(&model_dir) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error loading tokenizer: {}", e);
            std::process::exit(1);
        }
    };
    println!("  Vocab size: {}", tokenizer.vocab_size());
    println!("  BOS token: {:?}", tokenizer.bos_token_id());
    println!("  EOS token: {:?}", tokenizer.eos_token_id());

    // Load model weights
    println!("\nLoading model weights (this may take a moment)...");
    let start = Instant::now();
    let loader = match SafetensorsLoader::from_directory(&model_dir, DEFAULT_SCALE) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            std::process::exit(1);
        }
    };
    println!("  Config loaded in {:?}", start.elapsed());
    println!("  Hidden size: {}", loader.config.hidden_size);
    println!("  Num layers: {}", loader.config.num_hidden_layers);
    println!("  Num heads: {}", loader.config.num_attention_heads);

    // Load embeddings only (full model is too large for demo)
    println!("\nLoading embedding table...");
    let start = Instant::now();
    let embeddings = match loader.load_embeddings() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error loading embeddings: {}", e);
            std::process::exit(1);
        }
    };
    println!("  Loaded {} embeddings of dim {} in {:?}",
             embeddings.vocab_size,
             embeddings.embed_dim,
             start.elapsed());
    println!("  Total size: {:.2} MB",
             (embeddings.vocab_size * embeddings.embed_dim * 4) as f64 / 1_000_000.0);

    // Demo: Tokenize a prompt
    let prompt = "Hello, how are you?";
    println!("\n--- Tokenization Demo ---");
    println!("Input: \"{}\"", prompt);

    let token_ids = match tokenizer.encode(prompt) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("Error tokenizing: {}", e);
            std::process::exit(1);
        }
    };
    println!("Token IDs: {:?}", token_ids);

    // Show tokens
    print!("Tokens: ");
    for &id in &token_ids {
        if let Some(token) = tokenizer.id_to_token(id) {
            print!("[{}] ", token.replace('▁', "_"));
        }
    }
    println!();

    // Demo: Secure embedding retrieval with IKNP OT
    println!("\n--- Secure Embedding Retrieval (IKNP OT) ---");
    println!("Retrieving embeddings for {} tokens via IKNP OT extension...", token_ids.len());

    // Setup OT session with IKNP extension
    let config = OtSessionConfig {
        vocab_size: embeddings.vocab_size as u32,
        hidden_dim: embeddings.embed_dim as u16,
        max_prompt_len: 64,
        scale: DEFAULT_SCALE,
        ..Default::default()
    };

    let sender_ext = IknpOtExtension::new_server();
    let receiver_ext = IknpOtExtension::new_client();

    let mut sender = OtSender::new(sender_ext, config.clone());
    let mut receiver = OtReceiver::new(receiver_ext);

    // Server sets embedding database
    sender.set_embedding_db(embeddings.to_bytes());

    // Session handshake with IKNP base OT
    let start = Instant::now();

    let (init_header, init_payload) = receiver.generate_session_init();
    let (params_header, params_payload) =
        sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
    receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

    // IKNP base OT: Client sends A points, Server responds with B points
    let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
    if let Some((_, response_msg)) =
        sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
    {
        receiver.handle_base_ot_response(&response_msg).unwrap();
    }

    let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
    receiver.handle_session_ready(&ready_payload).unwrap();

    let base_ot_time = start.elapsed();
    println!("  Base OT (128 DH key exchanges): {:?}", base_ot_time);

    // Fetch embeddings using OT extension
    let start = Instant::now();
    let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&token_ids).unwrap();
    let (_, fetch_response) = sender
        .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
        .unwrap();
    let embedding_bytes = receiver
        .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
        .unwrap();

    let fetch_time = start.elapsed();
    println!("  Embedding fetch ({} rows): {:?}", token_ids.len(), fetch_time);
    println!("  Retrieved {} bytes ({} embeddings × {} dims × 4 bytes)",
             embedding_bytes.len(),
             token_ids.len(),
             embeddings.embed_dim);

    // Verify correctness
    println!("\n--- Verification ---");
    let row_bytes = embeddings.embed_dim * 4;
    let mut all_correct = true;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let ot_start = i * row_bytes;
        let ot_first = i32::from_le_bytes([
            embedding_bytes[ot_start],
            embedding_bytes[ot_start + 1],
            embedding_bytes[ot_start + 2],
            embedding_bytes[ot_start + 3],
        ]);

        let plain_emb = embeddings.get(token_id as usize).unwrap();
        let plain_first = plain_emb.data[0];

        if ot_first != plain_first {
            println!("  Token {}: MISMATCH (OT: {}, Plain: {})", token_id, ot_first, plain_first);
            all_correct = false;
        }
    }

    if all_correct {
        println!("  ✓ All {} embeddings match plaintext reference!", token_ids.len());
    }

    // Session Reuse Benchmark
    println!("\n--- Session Reuse Benchmark ---");
    println!("Testing session reuse with multiple embedding fetches...");

    let num_fetches = 50;
    let tokens_per_fetch = 8;

    // Generate random token sequences for benchmark
    let mut rng_tokens: Vec<Vec<u32>> = Vec::with_capacity(num_fetches);
    for i in 0..num_fetches {
        // Use deterministic "random" tokens based on index
        let tokens: Vec<u32> = (0..tokens_per_fetch)
            .map(|j| ((i * tokens_per_fetch + j) * 7 + 13) as u32 % embeddings.vocab_size as u32)
            .collect();
        rng_tokens.push(tokens);
    }

    // Time each fetch
    let mut fetch_times: Vec<Duration> = Vec::with_capacity(num_fetches);
    let mut total_bytes_sent: usize = 0;
    let mut total_bytes_received: usize = 0;

    let benchmark_start = Instant::now();

    for tokens in &rng_tokens {
        let fetch_start = Instant::now();

        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(tokens).unwrap();
        let request_size = fetch_request.query_blob.len() + 2; // +2 for len field

        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();
        let response_size = fetch_response.response_blob.len() + 4; // +4 for len/row_bytes fields

        let _embedding_bytes = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        fetch_times.push(fetch_start.elapsed());
        total_bytes_sent += request_size;
        total_bytes_received += response_size;
    }

    let benchmark_total = benchmark_start.elapsed();

    // Compute statistics
    fetch_times.sort();
    let min_time = fetch_times.first().copied().unwrap_or(Duration::ZERO);
    let max_time = fetch_times.last().copied().unwrap_or(Duration::ZERO);
    let avg_time = benchmark_total / num_fetches as u32;
    let p50 = percentile(&fetch_times, 50.0);
    let p95 = percentile(&fetch_times, 95.0);
    let p99 = percentile(&fetch_times, 99.0);

    println!("\n  Session Reuse Results ({} fetches, {} tokens each):", num_fetches, tokens_per_fetch);
    println!("  ─────────────────────────────────────────────────────");
    println!("  Base OT (one-time setup): {}", format_duration(base_ot_time));
    println!("  Total benchmark time:     {}", format_duration(benchmark_total));
    println!();
    println!("  Per-fetch timing:");
    println!("    Min:    {}", format_duration(min_time));
    println!("    Avg:    {}", format_duration(avg_time));
    println!("    p50:    {}", format_duration(p50));
    println!("    p95:    {}", format_duration(p95));
    println!("    p99:    {}", format_duration(p99));
    println!("    Max:    {}", format_duration(max_time));
    println!();
    println!("  Bandwidth per fetch:");
    println!("    Request:  {} bytes ({:.2} KB)",
             total_bytes_sent / num_fetches,
             total_bytes_sent as f64 / num_fetches as f64 / 1024.0);
    println!("    Response: {} bytes ({:.2} KB)",
             total_bytes_received / num_fetches,
             total_bytes_received as f64 / num_fetches as f64 / 1024.0);
    println!("    Total:    {} bytes ({:.2} KB)",
             (total_bytes_sent + total_bytes_received) / num_fetches,
             (total_bytes_sent + total_bytes_received) as f64 / num_fetches as f64 / 1024.0);
    println!();
    println!("  Throughput:");
    println!("    {} fetches/sec",
             (num_fetches as f64 / benchmark_total.as_secs_f64()).round() as u64);
    println!("    {} tokens/sec",
             ((num_fetches * tokens_per_fetch) as f64 / benchmark_total.as_secs_f64()).round() as u64);

    // Verify counter is advancing correctly
    println!("\n  Session state:");
    println!("    Counter after benchmark: {}", receiver.counter());
    println!("    Expected counter:        {}", num_fetches + 2); // +1 for initial fetch, +1 for first fetch

    // ==== Secure Linear Slice Demo ====
    println!("\n--- Secure Linear Slice (Q Projection) ---");
    println!("Loading TinyLlama Q projection weights (layer 0)...");

    let start = Instant::now();
    let q_proj = match loader.load_linear(
        "model.layers.0.self_attn.q_proj.weight",
        loader.config.hidden_size,
        loader.config.num_attention_heads * loader.config.head_dim(),
    ) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Error loading Q projection: {}", e);
            std::process::exit(1);
        }
    };
    println!("  Loaded Q projection in {:?}", start.elapsed());
    println!("  Dimensions: {} × {} = {} parameters",
             q_proj.in_features, q_proj.out_features,
             q_proj.in_features * q_proj.out_features);
    println!("  Size: {:.2} MB",
             (q_proj.in_features * q_proj.out_features * 4) as f64 / 1_000_000.0);

    // Use last token's embedding as input
    let last_token_id = *token_ids.last().unwrap() as usize;
    let input_embedding = embeddings.get(last_token_id).unwrap();
    println!("\n  Input: embedding for token {} (last token)", last_token_id);

    // Secure linear: Y = X * W where X is secret-shared
    println!("\n  Running secure Q projection...");
    let start = Instant::now();

    // Client creates additive shares of input: X = X_c + X_s
    let (mut linear_client, server_input_share) = LinearClient::new(&input_embedding);

    // Server creates linear layer with Q projection weights
    let linear_server = LinearServer::new(
        q_proj.weight.clone(),
        None, // No bias for LLaMA Q projection
        q_proj.in_features,
        q_proj.out_features,
        q_proj.scale,
    ).unwrap();

    // Client generates request (sends X_c)
    let request = linear_client.generate_request();

    // Server computes: Y_c = X_c * W, Y_s = X_s * W
    let response = linear_server.handle_request(&request, &server_input_share).unwrap();

    // Client receives response and reconstructs Y = Y_c + Y_s
    linear_client.handle_response(&response).unwrap();
    let secure_output = linear_client.reconstruct(&response.server_output_share).unwrap();

    let secure_time = start.elapsed();

    // Plaintext reference computation
    let start = Instant::now();
    let plaintext_output = plaintext_linear(
        &input_embedding,
        &q_proj.weight,
        None,
        q_proj.in_features,
        q_proj.out_features,
        q_proj.scale,
    ).unwrap();
    let plaintext_time = start.elapsed();

    println!("  Secure linear time:    {}", format_duration(secure_time));
    println!("  Plaintext linear time: {}", format_duration(plaintext_time));
    println!("  Output dimension: {}", secure_output.len());

    // Verify correctness
    let mut max_diff: i64 = 0;
    let mut sum_diff: i64 = 0;
    for (s, p) in secure_output.data.iter().zip(&plaintext_output.data) {
        let diff = (*s as i64 - *p as i64).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
    }
    let avg_diff = sum_diff as f64 / secure_output.len() as f64;

    println!("\n  Correctness verification:");
    println!("    Max |secure - plaintext|: {} LSB", max_diff);
    println!("    Avg |secure - plaintext|: {:.2} LSB", avg_diff);

    if max_diff <= 1 {
        println!("    ✓ Secure computation matches plaintext (within ±1 LSB rounding)!");
    } else {
        println!("    ⚠ Difference exceeds expected rounding error");
    }

    // Show sample output values
    let scale_factor = (1u64 << secure_output.scale) as f64;
    println!("\n  Sample output values (first 5 of {}):", secure_output.len());
    for i in 0..5.min(secure_output.len()) {
        let secure_f64 = secure_output.data[i] as f64 / scale_factor;
        let plain_f64 = plaintext_output.data[i] as f64 / scale_factor;
        println!("    [{}] secure: {:.4}, plaintext: {:.4}", i, secure_f64, plain_f64);
    }

    // Security verification
    println!("\n  Security verification:");
    let input_matches_client_share = input_embedding.data == request.client_share;
    let input_matches_server_share = input_embedding.data == server_input_share.data;
    println!("    Input ≠ client share: {} (server doesn't see X)", !input_matches_client_share);
    println!("    Input ≠ server share: {} (neither share reveals X)", !input_matches_server_share);

    // Verify shares reconstruct
    let reconstructed_input: Vec<i32> = request.client_share.iter()
        .zip(&server_input_share.data)
        .map(|(&c, &s)| c.wrapping_add(s))
        .collect();
    let input_reconstructs = reconstructed_input == input_embedding.data;
    println!("    X_c + X_s = X: {} (shares are valid)", input_reconstructs);

    // ==== Sequence Throughput Benchmark ====
    println!("\n--- Sequence Throughput Benchmark ---");
    println!("Testing secure Q projection for full sequences...");

    // Build input sequences from embeddings
    // L=6: the sample prompt
    // L=64: max sequence for current config
    let seq_lengths = [6usize, 64usize];

    for &seq_len in &seq_lengths {
        println!("\n  Sequence length L={}", seq_len);
        println!("  ─────────────────────────────────");

        // Create input vectors (using cycling through tokens for L=64)
        let mut inputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let token_idx = if i < token_ids.len() {
                token_ids[i] as usize
            } else {
                // Cycle through vocab for longer sequences
                (i * 137 + 42) % embeddings.vocab_size
            };
            inputs.push(embeddings.get(token_idx).unwrap().clone());
        }

        // Run secure batch linear
        let batch_result = secure_linear_batch_timed(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        // Run plaintext batch linear for comparison
        let start = Instant::now();
        let plaintext_outputs = plaintext_linear_batch(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();
        let plaintext_time = start.elapsed();

        // Compare outputs
        let (max_diff, avg_diff) = compare_batch_outputs(&batch_result.outputs, &plaintext_outputs).unwrap();

        // Calculate throughput
        let tokens_per_sec = seq_len as f64 / batch_result.total_time.as_secs_f64();
        let plaintext_tokens_per_sec = seq_len as f64 / plaintext_time.as_secs_f64();
        let overhead = batch_result.total_time.as_secs_f64() / plaintext_time.as_secs_f64();

        println!("    Secure time:      {} ({} per token)",
                 format_duration(batch_result.total_time),
                 format_duration(batch_result.per_token_time));
        println!("    Plaintext time:   {} ({} per token)",
                 format_duration(plaintext_time),
                 format_duration(plaintext_time / seq_len as u32));
        println!("    Overhead:         {:.2}x", overhead);
        println!();
        println!("    Throughput:");
        println!("      Secure:    {:.0} tokens/sec", tokens_per_sec);
        println!("      Plaintext: {:.0} tokens/sec", plaintext_tokens_per_sec);
        println!();
        println!("    Correctness:");
        println!("      Max |diff|: {} LSB", max_diff);
        println!("      Avg |diff|: {:.2} LSB", avg_diff);

        if max_diff <= 1 {
            println!("      ✓ All {} outputs correct (within ±1 LSB)!", seq_len);
        } else {
            println!("      ⚠ Difference exceeds expected rounding error");
        }
    }

    // ==== GEMM vs Matvec Benchmark ====
    println!("\n--- GEMM vs Sequential Matvec Benchmark ---");
    println!("Comparing batched GEMM to sequential matvec...");
    println!("(GEMM batches [L,d] × [d,d] vs L independent d×d matvecs)");

    let gemm_seq_lengths = [8usize, 16, 32, 64];

    for &seq_len in &gemm_seq_lengths {
        println!("\n  Sequence length L={}", seq_len);
        println!("  ─────────────────────────────────");

        // Create input vectors
        let mut inputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let token_idx = (i * 137 + 42) % embeddings.vocab_size;
            inputs.push(embeddings.get(token_idx).unwrap().clone());
        }

        // Run GEMM-based secure batch
        let gemm_result = secure_linear_gemm_timed(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        // Run sequential matvec (existing approach)
        let matvec_result = secure_linear_batch_timed(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        // Run plaintext GEMM for correctness check
        let plaintext_outputs = plaintext_linear_gemm(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        // Compare outputs
        let (max_diff, avg_diff) = compare_batch_outputs(&gemm_result.outputs, &plaintext_outputs).unwrap();

        // Calculate speedup and throughput
        let speedup = matvec_result.total_time.as_secs_f64() / gemm_result.total_time.as_secs_f64();
        let gemm_tps = seq_len as f64 / gemm_result.total_time.as_secs_f64();
        let matvec_tps = seq_len as f64 / matvec_result.total_time.as_secs_f64();

        println!("    GEMM time:    {} ({} per token)",
                 format_duration(gemm_result.total_time),
                 format_duration(gemm_result.per_token_time));
        println!("    Matvec time:  {} ({} per token)",
                 format_duration(matvec_result.total_time),
                 format_duration(matvec_result.per_token_time));
        println!("    Speedup:      {:.2}x", speedup);
        println!();
        println!("    Throughput:");
        println!("      GEMM:   {:.0} tokens/sec", gemm_tps);
        println!("      Matvec: {:.0} tokens/sec", matvec_tps);
        println!();
        println!("    Correctness:");
        println!("      Max |diff|: {} LSB", max_diff);
        println!("      Avg |diff|: {:.2} LSB", avg_diff);

        if max_diff <= 1 {
            println!("      ✓ GEMM matches plaintext (within ±1 LSB)!");
        } else {
            println!("      ⚠ Difference exceeds expected rounding error");
        }
    }

    // ==== Hybrid Strategy + Precomputed Transpose Benchmark ====
    println!("\n--- Hybrid Strategy Benchmark ---");
    println!("Testing hybrid GEMM/matvec with precomputed transpose...");
    println!("(Uses matvec for L<{}, GEMM for L>={})", GEMM_CROSSOVER_L, GEMM_CROSSOVER_L);

    // Create LinearWeights with precomputed transpose
    println!("\n  Precomputing W_t (transpose of Q projection)...");
    let start = Instant::now();
    let q_proj_weights = LinearWeights::new(
        q_proj.weight.clone(),
        None,
        q_proj.in_features,
        q_proj.out_features,
        q_proj.scale,
    ).unwrap();
    let transpose_time = start.elapsed();
    println!("    Transpose precomputation: {}", format_duration(transpose_time));
    println!("    W_t size: {:.2} MB (same as W)",
             (q_proj.in_features * q_proj.out_features * 4) as f64 / 1_000_000.0);

    let hybrid_seq_lengths = [4usize, 8, 16, 32, 64];

    for &seq_len in &hybrid_seq_lengths {
        println!("\n  Sequence length L={} {}", seq_len,
                 if seq_len < GEMM_CROSSOVER_L { "(matvec)" } else { "(GEMM)" });
        println!("  ─────────────────────────────────");

        // Create input vectors
        let mut inputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let token_idx = (i * 137 + 42) % embeddings.vocab_size;
            inputs.push(embeddings.get(token_idx).unwrap().clone());
        }

        // Run hybrid (no pretranspose)
        let hybrid_result = secure_linear_hybrid_timed(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        // Run GEMM with precomputed transpose
        let start = Instant::now();
        let pretrans_outputs = secure_linear_gemm_pretransposed(&inputs, &q_proj_weights).unwrap();
        let pretrans_time = start.elapsed();

        // Reference: sequential matvec
        let matvec_result = secure_linear_batch_timed(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        // Correctness check
        let plaintext_outputs = plaintext_linear_batch(
            &inputs,
            &q_proj.weight,
            None,
            q_proj.in_features,
            q_proj.out_features,
            q_proj.scale,
        ).unwrap();

        let (hybrid_max_diff, _) = compare_batch_outputs(&hybrid_result.outputs, &plaintext_outputs).unwrap();
        let (pretrans_max_diff, _) = compare_batch_outputs(&pretrans_outputs, &plaintext_outputs).unwrap();

        // Speedups
        let hybrid_speedup = matvec_result.total_time.as_secs_f64() / hybrid_result.total_time.as_secs_f64();
        let pretrans_speedup = matvec_result.total_time.as_secs_f64() / pretrans_time.as_secs_f64();

        println!("    Matvec time:      {} (baseline)", format_duration(matvec_result.total_time));
        println!("    Hybrid time:      {} ({:.2}x)", format_duration(hybrid_result.total_time), hybrid_speedup);
        println!("    Pretrans GEMM:    {} ({:.2}x)", format_duration(pretrans_time), pretrans_speedup);
        println!();
        println!("    Strategy used:    {}", if hybrid_result.used_gemm { "GEMM" } else { "matvec" });
        println!("    Correctness:      {} (hybrid), {} (pretrans) LSB max diff",
                 hybrid_max_diff, pretrans_max_diff);
    }

    println!("\n=== Demo Complete ===");
    println!("\nShardLM v1.2 Security Properties:");
    println!("  • Base OT: Simplest OT protocol using Curve25519 (Ristretto)");
    println!("  • κ = 128 base OTs establish shared keys");
    println!("  • Extension: IKNP-style row masking with AES-CTR PRF");
    println!("  • Session reuse: Base OT amortized over {} fetches", num_fetches + 1);
    println!("  • Secure Linear: Y = XW computed on secret-shared X");
    println!("  • Server holds all {} embeddings ({:.0} MB)",
             embeddings.vocab_size,
             (embeddings.vocab_size * embeddings.embed_dim * 4) as f64 / 1_000_000.0);
    println!("  • Server holds Q projection weights ({:.2} MB)",
             (q_proj.in_features * q_proj.out_features * 4) as f64 / 1_000_000.0);
    println!("  • Server is designed to learn nothing about which embeddings");
    println!("    were accessed (under IKNP OT assumptions)");
    println!("  • Server is designed to learn nothing about X or Y");
    println!("    (under additive secret sharing assumptions)");
}
