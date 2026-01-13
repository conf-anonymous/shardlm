#!/usr/bin/env python3
"""
Test GPU Prefill V2 (fully GPU-accelerated nonlinear ops).

Compares:
- v1 prefill: GPU matmul, CPU RMSNorm/SwiGLU
- v2 prefill: GPU matmul, GPU RMSNorm/SwiGLU

Expected: Performance improvement from reduced CPU overhead.
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


def batched_prefill_v1(session_id: str, hidden_client: List[List[float]], hidden_server: List[List[float]]):
    """Original prefill (GPU matmul, CPU nonlinear)."""
    resp = requests.post(f"{SERVER_URL}/v2/secure/gpu/prefill", json={
        "session_id": session_id,
        "hidden_client": hidden_client,
        "hidden_server": hidden_server,
    }, timeout=300)

    if resp.status_code != 200:
        raise ValueError(f"V1 prefill failed: {resp.status_code} {resp.text[:500]}")

    result = resp.json()
    return (
        sanitize_list(result['final_hidden_client']),
        sanitize_list(result['final_hidden_server']),
        result['k_cache'],
        result['v_cache'],
        sanitize_list(result['logits_client']),
        sanitize_list(result['logits_server']),
    )


def batched_prefill_v2(session_id: str, hidden_client: List[List[float]], hidden_server: List[List[float]]):
    """New prefill (GPU matmul + GPU nonlinear)."""
    resp = requests.post(f"{SERVER_URL}/v2/secure/gpu/prefill_v2", json={
        "session_id": session_id,
        "hidden_client": hidden_client,
        "hidden_server": hidden_server,
    }, timeout=300)

    if resp.status_code != 200:
        raise ValueError(f"V2 prefill failed: {resp.status_code} {resp.text[:500]}")

    result = resp.json()
    return (
        sanitize_list(result['final_hidden_client']),
        sanitize_list(result['final_hidden_server']),
        result['k_cache'],
        result['v_cache'],
        sanitize_list(result['logits_client']),
        sanitize_list(result['logits_server']),
    )


def test_prefill_comparison():
    print("=" * 70)
    print("GPU PREFILL V1 vs V2 COMPARISON")
    print("V1: GPU matmul, CPU RMSNorm/SwiGLU")
    print("V2: GPU matmul, GPU RMSNorm/SwiGLU (fully GPU)")
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

    # Init session
    print(f"\n1. Initializing secure session...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/session/init", json={"ot_public_keys": None})
    session_data = resp.json()
    session_id = session_data["session_id"]
    model_info = session_data["model_info"]
    print(f"   Model: {model_info['num_layers']} layers, {model_info['hidden_dim']} dim")

    # Get embeddings
    print(f"\n2. Fetching {len(prompt_tokens)} embeddings...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
        "session_id": session_id,
        "token_ids": prompt_tokens
    })
    embeddings = resp.json()
    print(f"   Got {len(embeddings['client_shares'])} embeddings")

    hidden_client = [sanitize_list(e) for e in embeddings['client_shares']]
    hidden_server = [sanitize_list(e) for e in embeddings['server_shares']]

    # ===== TEST V1 PREFILL =====
    print(f"\n3. Testing V1 prefill (GPU matmul + CPU nonlinear)...")
    v1_times = []
    for run in range(3):
        start = time.time()
        final_hidden_client_v1, final_hidden_server_v1, k_cache_v1, v_cache_v1, logits_client_v1, logits_server_v1 = batched_prefill_v1(
            session_id, hidden_client, hidden_server
        )
        elapsed = time.time() - start
        v1_times.append(elapsed)
        print(f"   Run {run+1}: {elapsed:.3f}s")

    v1_avg = sum(v1_times) / len(v1_times)
    print(f"   V1 Average: {v1_avg:.3f}s")

    # ===== TEST V2 PREFILL =====
    print(f"\n4. Testing V2 prefill (GPU matmul + GPU nonlinear)...")
    v2_times = []
    for run in range(3):
        start = time.time()
        final_hidden_client_v2, final_hidden_server_v2, k_cache_v2, v_cache_v2, logits_client_v2, logits_server_v2 = batched_prefill_v2(
            session_id, hidden_client, hidden_server
        )
        elapsed = time.time() - start
        v2_times.append(elapsed)
        print(f"   Run {run+1}: {elapsed:.3f}s")

    v2_avg = sum(v2_times) / len(v2_times)
    print(f"   V2 Average: {v2_avg:.3f}s")

    # ===== COMPARE OUTPUTS =====
    print(f"\n5. Verifying output correctness...")
    # Compare final hidden states
    max_diff_hidden = max(abs(a - b) for a, b in zip(final_hidden_client_v1[:100], final_hidden_client_v2[:100]))
    print(f"   Max diff in hidden states (first 100): {max_diff_hidden:.6f}")

    # Compare logits (top-10)
    logits_v1 = [c + s for c, s in zip(logits_client_v1, logits_server_v1)]
    logits_v2 = [c + s for c, s in zip(logits_client_v2, logits_server_v2)]

    # Get top-10 tokens
    top_v1 = sorted(enumerate(logits_v1), key=lambda x: -x[1])[:10]
    top_v2 = sorted(enumerate(logits_v2), key=lambda x: -x[1])[:10]

    top_v1_ids = [t[0] for t in top_v1]
    top_v2_ids = [t[0] for t in top_v2]
    match_count = len(set(top_v1_ids) & set(top_v2_ids))
    print(f"   Top-10 token overlap: {match_count}/10")

    if HAS_TOKENIZER:
        top_token_v1 = tokenizer.decode([top_v1[0][0]])
        top_token_v2 = tokenizer.decode([top_v2[0][0]])
        print(f"   V1 top token: '{top_token_v1}' ({top_v1[0][0]})")
        print(f"   V2 top token: '{top_token_v2}' ({top_v2[0][0]})")

    # ===== SUMMARY =====
    print(f"\n" + "=" * 70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    print(f"V1 (GPU matmul + CPU nonlinear): {v1_avg:.3f}s")
    print(f"V2 (GPU matmul + GPU nonlinear): {v2_avg:.3f}s")

    if v2_avg < v1_avg:
        speedup = v1_avg / v2_avg
        print(f"\nV2 is {speedup:.2f}x FASTER than V1!")
    else:
        slowdown = v2_avg / v1_avg
        print(f"\nV2 is {slowdown:.2f}x SLOWER than V1")
        print("(GPU nonlinear might have too much transfer overhead)")

    print(f"\nPrompt tokens: {len(prompt_tokens)}")
    print(f"V1 throughput: {len(prompt_tokens)/v1_avg:.1f} tok/s")
    print(f"V2 throughput: {len(prompt_tokens)/v2_avg:.1f} tok/s")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)
    test_prefill_comparison()
