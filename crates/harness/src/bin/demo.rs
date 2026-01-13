//! End-to-end secure inference demo with full transformer
//!
//! This demo shows the full pipeline:
//! 1. Load model weights (all layers)
//! 2. Tokenize input
//! 3. Fetch embeddings
//! 4. Run full transformer forward pass
//! 5. Decode and display output

use shardlm_fixed_point::DEFAULT_SCALE;
use shardlm_model::{compute_logits, KVCache, SafetensorsLoader, Tokenizer, TransformerState};
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ShardLM Full Transformer Inference Demo ===\n");

    // Get model directory from env or use default
    let model_dir = env::var("SHARDLM_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("tinyllama-weights"));

    println!("Loading model from: {:?}", model_dir);

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_directory(&model_dir)?;
    println!("  Vocab size: {}", tokenizer.vocab_size());

    // Load model weights (all layers)
    println!("Loading model weights...");
    let loader = SafetensorsLoader::from_directory(&model_dir, DEFAULT_SCALE)?;
    let weights = loader.load_model_weights()?;
    let embeddings = loader.load_embeddings()?;

    println!("  Hidden size: {}", weights.config.hidden_size);
    println!("  Vocab size: {}", weights.config.vocab_size);
    println!("  Num layers: {}", weights.num_layers());
    println!("  Num heads: {}", weights.config.num_attention_heads);
    println!("  Scale: {}", weights.scale);

    // Create transformer state
    let transformer = TransformerState::new(weights.config.clone(), weights.scale);

    // Input prompt
    let prompt = "The capital of France is";
    println!("\n--- Input ---");
    println!("Prompt: \"{}\"", prompt);

    // Tokenize
    println!("\n--- Tokenization ---");
    let token_ids = tokenizer.encode(prompt)?;
    println!("Token IDs: {:?}", token_ids);
    println!("Num tokens: {}", token_ids.len());

    // Show what each token represents
    print!("Tokens: ");
    for &id in &token_ids {
        if let Some(token) = tokenizer.id_to_token(id) {
            print!("[{}]", token.replace('▁', "_"));
        }
    }
    println!();

    // Create KV cache
    let mut kv_cache = KVCache::new(
        weights.num_layers(),
        weights.config.max_position_embeddings,
        weights.config.num_key_value_heads,
        weights.config.head_dim(),
    );

    // Process prompt tokens through transformer
    println!("\n--- Transformer Forward Pass ---");
    println!("Processing {} prompt tokens through {} layers...", token_ids.len(), weights.num_layers());

    let mut hidden_state = vec![0i32; weights.config.hidden_size];

    for (pos, &token_id) in token_ids.iter().enumerate() {
        print!("  Token {} (pos {})... ", token_id, pos);
        io::stdout().flush()?;

        // Get embedding for this token
        let start = (token_id as usize) * embeddings.embed_dim;
        let end = start + embeddings.embed_dim;

        if end <= embeddings.data.len() {
            hidden_state.copy_from_slice(&embeddings.data[start..end]);
        }

        // Run through transformer
        hidden_state = transformer.forward_single(&weights, &hidden_state, pos, &mut kv_cache);
        println!("done");
    }

    // Compute logits from final hidden state
    println!("\n--- Logit Computation ---");
    let logits = compute_logits(&hidden_state, &weights.lm_head, weights.scale);
    println!("Computed {} logits", logits.len());

    // Find top-k tokens
    println!("\n--- Token Prediction ---");
    let mut indexed: Vec<(usize, i32)> = logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
    indexed.sort_by(|a, b| b.1.cmp(&a.1));

    println!("Top 10 predicted tokens:");
    for (rank, &(token_id, logit)) in indexed.iter().take(10).enumerate() {
        let token_str = tokenizer
            .id_to_token(token_id as u32)
            .unwrap_or_else(|| format!("<unk:{}>", token_id));
        let logit_f64 = logit as f64 / (1u64 << weights.scale) as f64;
        println!(
            "  {}. Token {} = \"{}\" (logit: {:.4})",
            rank + 1,
            token_id,
            token_str.replace('▁', " "),
            logit_f64
        );
    }

    // Generate tokens autoregressively
    println!("\n--- Autoregressive Generation ---");
    let max_new_tokens = 20;
    let mut generated_tokens: Vec<u32> = Vec::new();
    let mut position = token_ids.len();

    print!("Generated: \"{}\"", prompt);
    io::stdout().flush()?;

    for _ in 0..max_new_tokens {
        // Get best token from current logits
        let (best_token, _) = indexed[0];
        let best_token_id = best_token as u32;

        // Check for EOS
        if best_token_id == 2 {
            break;
        }

        generated_tokens.push(best_token_id);

        // Decode and print
        let token_str = tokenizer
            .id_to_token(best_token_id)
            .unwrap_or_else(|| "?".to_string());
        // Handle special tokens: ▁ = space prefix, <0x0A> = newline
        let display_str = token_str
            .replace('▁', " ")
            .replace("<0x0A>", "\n");
        print!("{}", display_str);
        io::stdout().flush()?;

        // Get embedding for new token
        let start = (best_token_id as usize) * embeddings.embed_dim;
        let end = start + embeddings.embed_dim;

        if end <= embeddings.data.len() {
            hidden_state.copy_from_slice(&embeddings.data[start..end]);
        }

        // Run through transformer at new position
        hidden_state = transformer.forward_single(&weights, &hidden_state, position, &mut kv_cache);
        position += 1;

        // Recompute logits for next iteration
        let new_logits = compute_logits(&hidden_state, &weights.lm_head, weights.scale);
        indexed = new_logits.iter().enumerate().map(|(i, &l)| (i, l)).collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
    }
    println!("\"");

    // Final summary
    println!("\n--- Summary ---");
    println!("✓ Tokenized {} input tokens", token_ids.len());
    println!("✓ Loaded {} transformer layers", weights.num_layers());
    println!("✓ Ran full transformer forward pass");
    println!("✓ Computed {} logits", weights.config.vocab_size);
    println!("✓ Generated {} new tokens", generated_tokens.len());

    Ok(())
}
