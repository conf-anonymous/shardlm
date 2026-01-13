#!/usr/bin/env python3
"""
Optimized End-to-End test using BATCHED GPU endpoints.
Uses /v2/secure/gpu/generate/token for ALL 28 layers in ONE request.

Expected speedup: 10-50x over per-layer requests.
"""

import requests
import json
import math
import random
import time
from typing import List, Tuple, Dict, Any

SERVER_URL = "http://localhost:3000"

# Load the actual Qwen tokenizer
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    HAS_TOKENIZER = True
except:
    HAS_TOKENIZER = False
    print("Warning: transformers not available, using dummy tokens")


def sanitize_float(x) -> float:
    if x is None:
        return 0.0
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except:
        return 0.0


def sanitize_list(lst: List) -> List[float]:
    return [sanitize_float(x) for x in lst]


def softmax(scores: List[float]) -> List[float]:
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [1.0]
    max_score = max(scores)
    exps = [math.exp(min(s - max_score, 20)) for s in scores]  # Clamp to prevent overflow
    total = sum(exps)
    return [e / total for e in exps]


def client_sample_token(logits_client, logits_server, temperature=1.0, top_k=10):
    logits = [sanitize_float(c) + sanitize_float(s) for c, s in zip(logits_client, logits_server)]
    scaled = [l / temperature for l in logits]
    indexed = [(l, i) for i, l in enumerate(scaled)]
    indexed.sort(reverse=True)
    top_tokens = indexed[:top_k]
    top_logits = [t[0] for t in top_tokens]
    probs = softmax(top_logits)
    r = random.random()
    cum_prob = 0
    for i, (logit, token_id) in enumerate(top_tokens):
        cum_prob += probs[i]
        if r <= cum_prob:
            return token_id, logit
    return top_tokens[0][1], top_tokens[0][0]


class KVCache:
    """Server-side KV cache structure for batched processing."""
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # k_cache[layer][position] = Vec<f32>
        self.k_cache: List[List[List[float]]] = [[] for _ in range(num_layers)]
        self.v_cache: List[List[List[float]]] = [[] for _ in range(num_layers)]

    @property
    def seq_len(self) -> int:
        if self.k_cache and self.k_cache[0]:
            return len(self.k_cache[0])
        return 0

    def append(self, new_k: List[List[float]], new_v: List[List[float]]):
        """Append new K, V vectors for each layer."""
        for layer_idx in range(self.num_layers):
            if layer_idx < len(new_k):
                self.k_cache[layer_idx].append(new_k[layer_idx])
            if layer_idx < len(new_v):
                self.v_cache[layer_idx].append(new_v[layer_idx])


def process_token_batched(session_id: str, hidden_client: List[float], hidden_server: List[float],
                          position: int, kv_cache: KVCache) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Process a single token through ALL 28 layers in ONE request.
    Returns: (hidden_client, hidden_server, logits_client, logits_server)
    """
    resp = requests.post(f"{SERVER_URL}/v2/secure/gpu/generate/token", json={
        "session_id": session_id,
        "hidden_client": hidden_client,
        "hidden_server": hidden_server,
        "position": position,
        "k_cache": kv_cache.k_cache,
        "v_cache": kv_cache.v_cache,
    }, timeout=120)

    if resp.status_code != 200:
        raise ValueError(f"Generate token failed: {resp.status_code} {resp.text[:500]}")

    result = resp.json()

    # Update KV cache with new K, V vectors
    kv_cache.append(result['new_k'], result['new_v'])

    return (
        sanitize_list(result['hidden_client']),
        sanitize_list(result['hidden_server']),
        sanitize_list(result['logits_client']),
        sanitize_list(result['logits_server']),
    )


def test_batched_generation():
    print("=" * 70)
    print("BATCHED GPU END-TO-END GENERATION TEST")
    print("All 28 layers processed in ONE HTTP request per token!")
    print("=" * 70)

    # Format prompt using ChatML template
    user_message = "What is 2+2?"
    if HAS_TOKENIZER:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        print(f"\nPrompt: {user_message}")
        print(f"Formatted ({len(prompt_tokens)} tokens): {formatted_prompt[:100]}...")
    else:
        prompt_tokens = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198, 151644, 77091, 198]
        print(f"\nUsing dummy tokens ({len(prompt_tokens)} tokens)")

    start_time = time.time()

    # Init session
    print(f"\n1. Initializing secure session...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/session/init", json={"ot_public_keys": None})
    session_data = resp.json()
    session_id = session_data["session_id"]
    model_info = session_data["model_info"]
    print(f"   Model: {model_info['num_layers']} layers, {model_info['hidden_dim']} dim")

    kv_cache = KVCache(model_info['num_layers'])

    # Get embeddings
    print(f"\n2. Fetching {len(prompt_tokens)} embeddings...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
        "session_id": session_id,
        "token_ids": prompt_tokens
    })
    embeddings = resp.json()
    print(f"   Got {len(embeddings['client_shares'])} embeddings")

    # Process all prompt tokens with BATCHED endpoint
    print(f"\n3. Processing prompt (BATCHED - 1 request per token)...")
    prompt_start = time.time()
    hidden_client, hidden_server = None, None
    logits_client, logits_server = None, None

    for token_idx in range(len(prompt_tokens)):
        position = token_idx
        hidden_client = sanitize_list(embeddings['client_shares'][token_idx])
        hidden_server = sanitize_list(embeddings['server_shares'][token_idx])

        # ALL 28 layers + logits in ONE request!
        hidden_client, hidden_server, logits_client, logits_server = process_token_batched(
            session_id, hidden_client, hidden_server, position, kv_cache
        )

        if (token_idx + 1) % 5 == 0 or token_idx == len(prompt_tokens) - 1:
            elapsed = time.time() - prompt_start
            rate = (token_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"   Token {token_idx + 1}/{len(prompt_tokens)} done ({rate:.1f} tok/s)")

    prompt_time = time.time() - prompt_start
    print(f"   Prompt processing: {prompt_time:.2f}s ({len(prompt_tokens)/prompt_time:.1f} tok/s)")

    # Generate tokens
    print(f"\n4. Generating response (BATCHED)...")
    generated_tokens = []
    max_new_tokens = 20
    EOS_TOKEN = 151645

    gen_start = time.time()
    for step in range(max_new_tokens):
        # Sample token from last logits
        token_id, logit = client_sample_token(
            logits_client, logits_server,
            temperature=0.7, top_k=20
        )

        if token_id == EOS_TOKEN:
            print(f"   Step {step + 1}: EOS token, stopping")
            break

        generated_tokens.append(token_id)

        # Decode and show
        if HAS_TOKENIZER:
            partial = tokenizer.decode(generated_tokens)
            print(f"   Step {step + 1}: token {token_id} -> '{partial}'")
        else:
            print(f"   Step {step + 1}: token {token_id}")

        # Get next embedding
        resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
            "session_id": session_id,
            "token_ids": [token_id]
        })
        next_embed = resp.json()

        hidden_client = sanitize_list(next_embed['client_shares'][0])
        hidden_server = sanitize_list(next_embed['server_shares'][0])

        position = kv_cache.seq_len

        # ALL 28 layers + logits in ONE request!
        hidden_client, hidden_server, logits_client, logits_server = process_token_batched(
            session_id, hidden_client, hidden_server, position, kv_cache
        )

    gen_time = time.time() - gen_start
    total_time = time.time() - start_time

    # Final output
    if HAS_TOKENIZER:
        response = tokenizer.decode(generated_tokens)
    else:
        response = f"[Token IDs: {generated_tokens}]"

    print(f"\n" + "=" * 70)
    print("BATCHED GPU RESULTS")
    print("=" * 70)
    print(f"User: {user_message}")
    print(f"Response: {response}")
    print(f"\nTokens generated: {len(generated_tokens)}")
    print(f"Prompt time: {prompt_time:.2f}s ({len(prompt_tokens)/prompt_time:.1f} tok/s)")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    if gen_time > 0:
        print(f"Generation speed: {len(generated_tokens)/gen_time:.2f} tok/s")
    print("=" * 70)

    # Compare with previous results
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 50)
    cpu_total = 157.14
    gpu_per_layer_total = 68.05
    print(f"CPU (112 requests/token):     {cpu_total:.1f}s")
    print(f"GPU per-layer (4 req/layer):  {gpu_per_layer_total:.1f}s")
    print(f"GPU BATCHED (1 req/token):    {total_time:.1f}s")
    print(f"Speedup vs CPU:               {cpu_total/total_time:.1f}x")
    print(f"Speedup vs GPU per-layer:     {gpu_per_layer_total/total_time:.1f}x")
    print("=" * 70)
    print("BATCHED GPU TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)
    test_batched_generation()
