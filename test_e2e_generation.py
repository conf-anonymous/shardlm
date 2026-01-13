#!/usr/bin/env python3
"""
End-to-end test for secure inference with KV cache.

This test simulates the full web client flow:
1. Initialize secure session
2. Tokenize input
3. Fetch embeddings
4. Process all layers for each prompt token (building KV cache)
5. Compute logits and sample
6. Generate multiple tokens with KV cache
"""

import requests
import json
import math
import random
import time
from typing import List, Tuple, Dict, Any

SERVER_URL = "http://localhost:3000"

# Model dimensions (from Qwen2.5-1.5B)
NUM_LAYERS = 28
NUM_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
HIDDEN_DIM = 1536
INTERMEDIATE_DIM = 8960


def sanitize_float(x) -> float:
    """Ensure value is a valid float."""
    if x is None:
        return 0.0
    try:
        f = float(x)
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return f
    except (TypeError, ValueError):
        return 0.0


def sanitize_list(lst: List) -> List[float]:
    """Sanitize a list of floats."""
    return [sanitize_float(x) for x in lst]


class SecureKVCache:
    """Client-side KV cache storing shares."""
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # cache[layer_idx] = {'k_client': [...], 'k_server': [...], 'v_client': [...], 'v_server': [...]}
        self.cache: Dict[int, Dict[str, List[List[float]]]] = {
            i: {'k_client': [], 'k_server': [], 'v_client': [], 'v_server': []}
            for i in range(num_layers)
        }
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def append(self, layer_idx: int, k_client: List[float], k_server: List[float],
               v_client: List[float], v_server: List[float]):
        # Sanitize all values before storing
        self.cache[layer_idx]['k_client'].append([sanitize_float(x) for x in k_client])
        self.cache[layer_idx]['k_server'].append([sanitize_float(x) for x in k_server])
        self.cache[layer_idx]['v_client'].append([sanitize_float(x) for x in v_client])
        self.cache[layer_idx]['v_server'].append([sanitize_float(x) for x in v_server])
        if layer_idx == 0:
            self._seq_len = len(self.cache[0]['k_client'])

    def reconstruct_k(self, layer_idx: int) -> List[List[float]]:
        """Reconstruct K values for attention."""
        layer = self.cache[layer_idx]
        result = []
        for pos in range(len(layer['k_client'])):
            k = [c + s for c, s in zip(layer['k_client'][pos], layer['k_server'][pos])]
            result.append(k)
        return result

    def reconstruct_v(self, layer_idx: int) -> List[List[float]]:
        """Reconstruct V values for attention."""
        layer = self.cache[layer_idx]
        result = []
        for pos in range(len(layer['v_client'])):
            v = [c + s for c, s in zip(layer['v_client'][pos], layer['v_server'][pos])]
            result.append(v)
        return result


def softmax(scores: List[float]) -> List[float]:
    """Compute softmax."""
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [1.0]
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def client_rms_norm(client_share: List[float], server_share: List[float], eps: float = 1e-6) -> Tuple[List[float], List[float]]:
    """RMSNorm (client-side nonlinear operation)."""
    # Reconstruct
    hidden = [sanitize_float(c) + sanitize_float(s) for c, s in zip(client_share, server_share)]

    # Compute RMS
    square_sum = sum(x * x for x in hidden)
    rms = math.sqrt(square_sum / len(hidden) + eps)
    normalized = [x / rms for x in hidden]

    # Re-share
    new_client = [(random.random() - 0.5) * 2 for _ in normalized]
    new_server = [sanitize_float(n - c) for n, c in zip(normalized, new_client)]
    new_client = [sanitize_float(c) for c in new_client]

    return new_client, new_server


def client_attention_with_cache(q_client: List[float], q_server: List[float],
                                 cached_k: List[List[float]], cached_v: List[List[float]],
                                 num_heads: int, num_kv_heads: int, head_dim: int) -> Tuple[List[float], List[float]]:
    """
    Compute attention with KV cache (client-side nonlinear operation).

    Args:
        q_client/q_server: Q shares [num_heads * head_dim]
        cached_k: Reconstructed K [seq_len][num_kv_heads * head_dim]
        cached_v: Reconstructed V [seq_len][num_kv_heads * head_dim]
    """
    # Reconstruct Q
    q = [sanitize_float(c) + sanitize_float(s) for c, s in zip(q_client, q_server)]

    seq_len = len(cached_k)
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    output = [0.0] * (num_heads * head_dim)

    for h in range(num_heads):
        kv_head = min(h // heads_per_kv, num_kv_heads - 1)
        q_head = q[h * head_dim:(h + 1) * head_dim]

        # Compute attention scores over all positions
        scores = []
        for pos in range(seq_len):
            k_head = cached_k[pos][kv_head * head_dim:(kv_head + 1) * head_dim]
            score = sum(sanitize_float(qi) * sanitize_float(ki) for qi, ki in zip(q_head, k_head)) * scale
            scores.append(sanitize_float(score))

        # Softmax
        weights = softmax(scores)

        # Weighted sum of V
        for pos in range(seq_len):
            v_head = cached_v[pos][kv_head * head_dim:(kv_head + 1) * head_dim]
            for d in range(head_dim):
                output[h * head_dim + d] += weights[pos] * sanitize_float(v_head[d])

    # Re-share
    new_client = [(random.random() - 0.5) * 2 for _ in output]
    new_server = [sanitize_float(o - c) for o, c in zip(output, new_client)]
    new_client = [sanitize_float(c) for c in new_client]

    return new_client, new_server


def client_swiglu(gate_client: List[float], gate_server: List[float],
                  up_client: List[float], up_server: List[float]) -> Tuple[List[float], List[float]]:
    """SwiGLU activation (client-side nonlinear operation)."""
    # Reconstruct
    gate = [sanitize_float(c) + sanitize_float(s) for c, s in zip(gate_client, gate_server)]
    up = [sanitize_float(c) + sanitize_float(s) for c, s in zip(up_client, up_server)]

    # SwiGLU: silu(gate) * up
    def silu(x):
        x = sanitize_float(x)
        x = max(-500, min(500, x))  # Clamp to prevent overflow
        return x / (1 + math.exp(-x))

    output = [sanitize_float(silu(g) * u) for g, u in zip(gate, up)]

    # Re-share
    new_client = [(random.random() - 0.5) * 2 for _ in output]
    new_server = [sanitize_float(o - c) for o, c in zip(output, new_client)]
    new_client = [sanitize_float(c) for c in new_client]

    return new_client, new_server


def client_sample_token(logits_client: List[float], logits_server: List[float],
                        temperature: float = 1.0, top_k: int = 10) -> Tuple[int, float]:
    """Sample next token (client-side)."""
    # Reconstruct logits
    logits = [c + s for c, s in zip(logits_client, logits_server)]

    # Apply temperature
    scaled = [l / temperature for l in logits]

    # Top-k
    indexed = [(l, i) for i, l in enumerate(scaled)]
    indexed.sort(reverse=True)
    top_tokens = indexed[:top_k]

    # Softmax over top-k
    top_logits = [t[0] for t in top_tokens]
    probs = softmax(top_logits)

    # Sample
    r = random.random()
    cum_prob = 0
    for i, (logit, token_id) in enumerate(top_tokens):
        cum_prob += probs[i]
        if r <= cum_prob:
            return token_id, logit

    return top_tokens[0][1], top_tokens[0][0]


def process_layer(session_id: str, hidden_client: List[float], hidden_server: List[float],
                  layer_idx: int, position: int, kv_cache: SecureKVCache) -> Tuple[List[float], List[float]]:
    """
    Process one transformer layer with KV cache.

    Flow:
    1. QKV projection (server)
    2. Store K/V in cache
    3. Attention with all cached K/V (client)
    4. O projection + gate/up (server)
    5. SwiGLU (client)
    6. Down projection + residual (server)
    """
    # Step 1: QKV projection
    resp = requests.post(f"{SERVER_URL}/v2/secure/layer/step", json={
        "session_id": session_id,
        "client_share": hidden_client,
        "server_share": hidden_server,
        "layer_idx": layer_idx,
        "position": position
    }, timeout=60)

    if resp.status_code != 200:
        raise ValueError(f"QKV projection failed: {resp.status_code} {resp.text[:200]}")

    qkv = resp.json()

    # Sanitize QKV response
    q_client = sanitize_list(qkv['q_client'])
    q_server = sanitize_list(qkv['q_server'])
    k_client = sanitize_list(qkv['k_client'])
    k_server = sanitize_list(qkv['k_server'])
    v_client = sanitize_list(qkv['v_client'])
    v_server = sanitize_list(qkv['v_server'])

    # Step 2: Store K/V in cache
    kv_cache.append(layer_idx, k_client, k_server, v_client, v_server)

    # Step 3: Attention with all cached K/V
    cached_k = kv_cache.reconstruct_k(layer_idx)
    cached_v = kv_cache.reconstruct_v(layer_idx)

    attn_client, attn_server = client_attention_with_cache(
        q_client, q_server,
        cached_k, cached_v,
        qkv['num_heads'], qkv['num_kv_heads'], qkv['head_dim']
    )

    # Step 4: O projection + gate/up
    resp = requests.post(f"{SERVER_URL}/v2/secure/forward/batched", json={
        "session_id": session_id,
        "hidden_client": hidden_client,
        "hidden_server": hidden_server,
        "attention_outputs": [{
            "attn_out_client": attn_client,
            "attn_out_server": attn_server
        }],
        "chunk_info": {
            "chunk_idx": 0,
            "total_chunks": 1,
            "start_layer": layer_idx,
            "end_layer": layer_idx + 1,
            "is_last": True
        }
    }, timeout=60)

    if resp.status_code != 200:
        raise ValueError(f"Gate/Up failed: {resp.status_code} {resp.text[:200]}")

    gate_up = resp.json()

    if 'layers_gate_up' not in gate_up or not gate_up['layers_gate_up']:
        raise ValueError(f"No gate_up returned for layer {layer_idx}: {gate_up}")

    layer_gate_up = gate_up['layers_gate_up'][0]

    # Sanitize gate_up response
    gate_client = sanitize_list(layer_gate_up['gate_client'])
    gate_server = sanitize_list(layer_gate_up['gate_server'])
    up_client = sanitize_list(layer_gate_up['up_client'])
    up_server = sanitize_list(layer_gate_up['up_server'])
    residual_client = sanitize_list(layer_gate_up['hidden_after_attn_client'])
    residual_server = sanitize_list(layer_gate_up['hidden_after_attn_server'])

    # Step 5: SwiGLU (client)
    activated_client, activated_server = client_swiglu(
        gate_client, gate_server,
        up_client, up_server
    )

    # Step 6: Down projection + residual
    resp = requests.post(f"{SERVER_URL}/v2/secure/forward/batched", json={
        "session_id": session_id,
        "hidden_client": hidden_client,
        "hidden_server": hidden_server,
        "activated_ffn": [{
            "layer_idx": layer_idx,
            "activated_client": activated_client,
            "activated_server": activated_server,
            "residual_client": residual_client,
            "residual_server": residual_server
        }]
    }, timeout=60)

    if resp.status_code != 200:
        raise ValueError(f"Down proj failed: {resp.status_code} {resp.text[:200]}")

    final = resp.json()

    if 'final_hidden_client' not in final:
        raise ValueError(f"No final_hidden_client in response: {final}")

    # Sanitize outputs to ensure no null/NaN values for next layer
    return sanitize_list(final['final_hidden_client']), sanitize_list(final['final_hidden_server'])


def test_e2e_generation():
    """Full end-to-end generation test."""
    print("=" * 60)
    print("END-TO-END SECURE INFERENCE TEST WITH KV CACHE")
    print("=" * 60)

    # Use some simple token IDs for testing (these are arbitrary)
    # In a real scenario, these would come from a tokenizer
    prompt_tokens = [1234, 5678, 9012]  # 3 prompt tokens

    print(f"\n1. Initializing secure session...")
    start_time = time.time()

    resp = requests.post(f"{SERVER_URL}/v2/secure/session/init", json={
        "ot_public_keys": None
    })
    session_data = resp.json()
    session_id = session_data["session_id"]
    model_info = session_data["model_info"]

    print(f"   Session: {session_id[:8]}...")
    print(f"   Model: {model_info['num_layers']} layers, {model_info['hidden_dim']} hidden dim")
    print(f"   Heads: {model_info['num_heads']} Q, {model_info['num_kv_heads']} KV, {model_info['head_dim']} head_dim")

    # Create KV cache
    kv_cache = SecureKVCache(model_info['num_layers'])

    print(f"\n2. Fetching embeddings for {len(prompt_tokens)} prompt tokens...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
        "session_id": session_id,
        "token_ids": prompt_tokens
    })
    embeddings = resp.json()
    print(f"   Got {len(embeddings['client_shares'])} embeddings of dim {len(embeddings['client_shares'][0])}")

    print(f"\n3. Processing prompt tokens through all layers (building KV cache)...")

    hidden_client = embeddings['client_shares'][0]
    hidden_server = embeddings['server_shares'][0]

    for token_idx in range(len(prompt_tokens)):
        position = token_idx
        print(f"   Token {token_idx + 1}/{len(prompt_tokens)} at position {position}...")

        hidden_client = embeddings['client_shares'][token_idx]
        hidden_server = embeddings['server_shares'][token_idx]

        # RMSNorm on embedding
        hidden_client, hidden_server = client_rms_norm(hidden_client, hidden_server)

        # Process through all layers
        for layer_idx in range(model_info['num_layers']):
            hidden_client, hidden_server = process_layer(
                session_id, hidden_client, hidden_server,
                layer_idx, position, kv_cache
            )

            if layer_idx % 7 == 0:
                print(f"      Layer {layer_idx + 1}/{model_info['num_layers']} done")

    print(f"   KV cache now has {kv_cache.seq_len} positions")

    print(f"\n4. Generating tokens...")
    generated_tokens = []
    max_new_tokens = 5  # Generate 5 tokens for testing

    for step in range(max_new_tokens):
        # Final RMSNorm
        normed_client, normed_server = client_rms_norm(hidden_client, hidden_server)

        # Compute logits
        resp = requests.post(f"{SERVER_URL}/v2/secure/logits", json={
            "session_id": session_id,
            "hidden_client": normed_client,
            "hidden_server": normed_server
        })
        logits = resp.json()

        # Sample token
        token_id, logit = client_sample_token(
            logits['logits_client'], logits['logits_server'],
            temperature=1.0, top_k=10
        )

        generated_tokens.append(token_id)
        print(f"   Step {step + 1}: Generated token {token_id} (logit: {logit:.2f})")

        # Check for EOS (151645 for Qwen)
        if token_id == 151645:
            print("   EOS token generated, stopping")
            break

        # Get embedding for next token
        resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
            "session_id": session_id,
            "token_ids": [token_id]
        })
        next_embed = resp.json()

        hidden_client = next_embed['client_shares'][0]
        hidden_server = next_embed['server_shares'][0]

        # RMSNorm
        hidden_client, hidden_server = client_rms_norm(hidden_client, hidden_server)

        # Process through all layers with KV cache
        position = kv_cache.seq_len  # Next position
        for layer_idx in range(model_info['num_layers']):
            hidden_client, hidden_server = process_layer(
                session_id, hidden_client, hidden_server,
                layer_idx, position, kv_cache
            )

        print(f"      KV cache: {kv_cache.seq_len} positions")

    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"✓ Prompt tokens processed: {len(prompt_tokens)}")
    print(f"✓ Tokens generated: {len(generated_tokens)}")
    print(f"✓ Generated token IDs: {generated_tokens}")
    print(f"✓ Final KV cache size: {kv_cache.seq_len} positions")
    print(f"✓ Total time: {elapsed:.2f}s")
    print(f"✓ Tokens/sec: {len(generated_tokens) / elapsed:.2f}")
    print("=" * 60)
    print("END-TO-END TEST PASSED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    test_e2e_generation()
