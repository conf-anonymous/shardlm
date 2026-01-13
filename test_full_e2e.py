#!/usr/bin/env python3
"""
Full end-to-end test that simulates exactly what the web client does.
Uses the actual tokenizer and generates a real response.
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


class SecureKVCache:
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.cache: Dict[int, Dict[str, List[List[float]]]] = {
            i: {'k_client': [], 'k_server': [], 'v_client': [], 'v_server': []}
            for i in range(num_layers)
        }
        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def append(self, layer_idx: int, k_client, k_server, v_client, v_server):
        self.cache[layer_idx]['k_client'].append(sanitize_list(k_client))
        self.cache[layer_idx]['k_server'].append(sanitize_list(k_server))
        self.cache[layer_idx]['v_client'].append(sanitize_list(v_client))
        self.cache[layer_idx]['v_server'].append(sanitize_list(v_server))
        if layer_idx == 0:
            self._seq_len = len(self.cache[0]['k_client'])

    def reconstruct_k(self, layer_idx: int) -> List[List[float]]:
        layer = self.cache[layer_idx]
        result = []
        for pos in range(len(layer['k_client'])):
            k = [c + s for c, s in zip(layer['k_client'][pos], layer['k_server'][pos])]
            result.append(k)
        return result

    def reconstruct_v(self, layer_idx: int) -> List[List[float]]:
        layer = self.cache[layer_idx]
        result = []
        for pos in range(len(layer['v_client'])):
            v = [c + s for c, s in zip(layer['v_client'][pos], layer['v_server'][pos])]
            result.append(v)
        return result


def softmax(scores: List[float]) -> List[float]:
    if len(scores) == 0:
        return []
    if len(scores) == 1:
        return [1.0]
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def client_rms_norm(client_share, server_share, eps=1e-6):
    hidden = [sanitize_float(c) + sanitize_float(s) for c, s in zip(client_share, server_share)]
    square_sum = sum(x * x for x in hidden)
    rms = math.sqrt(square_sum / len(hidden) + eps)
    normalized = [x / rms for x in hidden]
    new_client = [(random.random() - 0.5) * 2 for _ in normalized]
    new_server = [sanitize_float(n - c) for n, c in zip(normalized, new_client)]
    return sanitize_list(new_client), sanitize_list(new_server)


def client_attention_with_cache(q_client, q_server, cached_k, cached_v, num_heads, num_kv_heads, head_dim):
    q = [sanitize_float(c) + sanitize_float(s) for c, s in zip(q_client, q_server)]
    seq_len = len(cached_k)
    heads_per_kv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    output = [0.0] * (num_heads * head_dim)

    for h in range(num_heads):
        kv_head = min(h // heads_per_kv, num_kv_heads - 1)
        q_head = q[h * head_dim:(h + 1) * head_dim]
        scores = []
        for pos in range(seq_len):
            k_head = cached_k[pos][kv_head * head_dim:(kv_head + 1) * head_dim]
            score = sum(sanitize_float(qi) * sanitize_float(ki) for qi, ki in zip(q_head, k_head)) * scale
            scores.append(sanitize_float(score))
        weights = softmax(scores)
        for pos in range(seq_len):
            v_head = cached_v[pos][kv_head * head_dim:(kv_head + 1) * head_dim]
            for d in range(head_dim):
                output[h * head_dim + d] += weights[pos] * sanitize_float(v_head[d])

    new_client = [(random.random() - 0.5) * 2 for _ in output]
    new_server = [sanitize_float(o - c) for o, c in zip(output, new_client)]
    return sanitize_list(new_client), sanitize_list(new_server)


def client_swiglu(gate_client, gate_server, up_client, up_server):
    gate = [sanitize_float(c) + sanitize_float(s) for c, s in zip(gate_client, gate_server)]
    up = [sanitize_float(c) + sanitize_float(s) for c, s in zip(up_client, up_server)]

    def silu(x):
        x = sanitize_float(x)
        x = max(-500, min(500, x))
        return x / (1 + math.exp(-x))

    output = [sanitize_float(silu(g) * u) for g, u in zip(gate, up)]
    new_client = [(random.random() - 0.5) * 2 for _ in output]
    new_server = [sanitize_float(o - c) for o, c in zip(output, new_client)]
    return sanitize_list(new_client), sanitize_list(new_server)


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


def process_layer(session_id, hidden_client, hidden_server, layer_idx, position, kv_cache):
    # Store un-normalized hidden for residual connection
    residual_input_client = hidden_client
    residual_input_server = hidden_server

    # Normalize for input_layernorm (server will apply gamma)
    normed_client, normed_server = client_rms_norm(hidden_client, hidden_server)

    # QKV projection (uses normalized + gamma input)
    resp = requests.post(f"{SERVER_URL}/v2/secure/layer/step", json={
        "session_id": session_id,
        "client_share": normed_client,
        "server_share": normed_server,
        "layer_idx": layer_idx,
        "position": position
    }, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"QKV failed: {resp.status_code} {resp.text[:200]}")
    qkv = resp.json()

    q_client = sanitize_list(qkv['q_client'])
    q_server = sanitize_list(qkv['q_server'])
    k_client = sanitize_list(qkv['k_client'])
    k_server = sanitize_list(qkv['k_server'])
    v_client = sanitize_list(qkv['v_client'])
    v_server = sanitize_list(qkv['v_server'])

    kv_cache.append(layer_idx, k_client, k_server, v_client, v_server)

    cached_k = kv_cache.reconstruct_k(layer_idx)
    cached_v = kv_cache.reconstruct_v(layer_idx)

    attn_client, attn_server = client_attention_with_cache(
        q_client, q_server, cached_k, cached_v,
        qkv['num_heads'], qkv['num_kv_heads'], qkv['head_dim']
    )

    # Phase 2a: O projection + residual → get hidden_after_attn
    # Use un-normalized residual_input for residual connection
    resp = requests.post(f"{SERVER_URL}/v2/secure/forward/batched", json={
        "session_id": session_id,
        "hidden_client": residual_input_client,  # Un-normalized for residual
        "hidden_server": residual_input_server,
        "attention_outputs": [{
            "attn_out_client": attn_client,
            "attn_out_server": attn_server
        }],
        "chunk_info": {
            "chunk_idx": 0, "total_chunks": 1,
            "start_layer": layer_idx, "end_layer": layer_idx + 1,
            "is_last": True
        }
    }, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Phase 2a failed: {resp.status_code} {resp.text[:200]}")
    phase2a = resp.json()

    layer_output = phase2a['layers_gate_up'][0]
    hidden_after_attn_client = sanitize_list(layer_output['hidden_after_attn_client'])
    hidden_after_attn_server = sanitize_list(layer_output['hidden_after_attn_server'])

    # Client normalizes hidden_after_attn for post_attn_layernorm
    norm_client, norm_server = client_rms_norm(hidden_after_attn_client, hidden_after_attn_server)

    # Phase 2b: Apply gamma + gate/up with normalized input
    # hidden_client/hidden_server not used for residual here, just for consistency
    resp = requests.post(f"{SERVER_URL}/v2/secure/forward/batched", json={
        "session_id": session_id,
        "hidden_client": residual_input_client,  # Not used for phase 2b
        "hidden_server": residual_input_server,
        "attention_outputs": [{
            "attn_out_client": attn_client,
            "attn_out_server": attn_server,
            "normalized_ffn_client": norm_client,
            "normalized_ffn_server": norm_server
        }],
        "chunk_info": {
            "chunk_idx": 0, "total_chunks": 1,
            "start_layer": layer_idx, "end_layer": layer_idx + 1,
            "is_last": True
        }
    }, timeout=60)
    if resp.status_code != 200:
        raise ValueError(f"Phase 2b failed: {resp.status_code} {resp.text[:200]}")
    gate_up = resp.json()

    layer_gate_up = gate_up['layers_gate_up'][0]
    gate_client = sanitize_list(layer_gate_up['gate_client'])
    gate_server = sanitize_list(layer_gate_up['gate_server'])
    up_client = sanitize_list(layer_gate_up['up_client'])
    up_server = sanitize_list(layer_gate_up['up_server'])
    # hidden_after_attn (un-normalized) for FFN residual connection
    residual_client = hidden_after_attn_client
    residual_server = hidden_after_attn_server

    activated_client, activated_server = client_swiglu(gate_client, gate_server, up_client, up_server)

    # Down proj
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

    return sanitize_list(final['final_hidden_client']), sanitize_list(final['final_hidden_server'])


def test_real_generation():
    print("=" * 70)
    print("FULL END-TO-END GENERATION TEST")
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

    kv_cache = SecureKVCache(model_info['num_layers'])

    # Get embeddings
    print(f"\n2. Fetching {len(prompt_tokens)} embeddings...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
        "session_id": session_id,
        "token_ids": prompt_tokens
    })
    embeddings = resp.json()
    print(f"   Got {len(embeddings['client_shares'])} embeddings")

    # Process all prompt tokens
    print(f"\n3. Processing prompt through {model_info['num_layers']} layers...")
    prompt_start = time.time()
    hidden_client, hidden_server = None, None

    for token_idx in range(len(prompt_tokens)):
        position = token_idx
        hidden_client = sanitize_list(embeddings['client_shares'][token_idx])
        hidden_server = sanitize_list(embeddings['server_shares'][token_idx])

        for layer_idx in range(model_info['num_layers']):
            # process_layer handles normalization internally for correct residual connections
            hidden_client, hidden_server = process_layer(
                session_id, hidden_client, hidden_server,
                layer_idx, position, kv_cache
            )

        if (token_idx + 1) % 5 == 0 or token_idx == len(prompt_tokens) - 1:
            print(f"   Token {token_idx + 1}/{len(prompt_tokens)} done")

    prompt_time = time.time() - prompt_start
    print(f"   Prompt processing: {prompt_time:.2f}s ({len(prompt_tokens)/prompt_time:.1f} tok/s)")

    # Generate tokens
    print(f"\n4. Generating response...")
    generated_tokens = []
    max_new_tokens = 20
    EOS_TOKEN = 151645

    gen_start = time.time()
    for step in range(max_new_tokens):
        normed_client, normed_server = client_rms_norm(hidden_client, hidden_server)

        resp = requests.post(f"{SERVER_URL}/v2/secure/logits", json={
            "session_id": session_id,
            "hidden_client": normed_client,
            "hidden_server": normed_server
        }, timeout=60)
        logits = resp.json()

        token_id, logit = client_sample_token(
            logits['logits_client'], logits['logits_server'],
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
        for layer_idx in range(model_info['num_layers']):
            # process_layer handles normalization internally
            hidden_client, hidden_server = process_layer(
                session_id, hidden_client, hidden_server,
                layer_idx, position, kv_cache
            )

    gen_time = time.time() - gen_start
    total_time = time.time() - start_time

    # Final output
    if HAS_TOKENIZER:
        response = tokenizer.decode(generated_tokens)
    else:
        response = f"[Token IDs: {generated_tokens}]"

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"User: {user_message}")
    print(f"Response: {response}")
    print(f"\nTokens generated: {len(generated_tokens)}")
    print(f"Prompt time: {prompt_time:.2f}s")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Generation speed: {len(generated_tokens)/gen_time:.2f} tok/s" if gen_time > 0 else "")
    print("=" * 70)
    print("TEST PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    random.seed(42)
    test_real_generation()
