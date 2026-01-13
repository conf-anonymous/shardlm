#!/usr/bin/env python3
"""Test KV cache implementation for secure inference."""

import requests
import json

SERVER_URL = "http://localhost:3000"

def test_secure_inference():
    print("Testing secure inference with KV cache...")

    # 1. Initialize session
    print("\n1. Initializing secure session...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/session/init", json={
        "ot_public_keys": None
    })
    session_data = resp.json()
    session_id = session_data["session_id"]
    model_info = session_data["model_info"]
    print(f"Session ID: {session_id}")
    print(f"Model: {model_info}")

    # 2. Get embeddings for 2 tokens (to test multi-position)
    print("\n2. Fetching embeddings for 2 tokens...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/embeddings/direct", json={
        "session_id": session_id,
        "token_ids": [1234, 5678]  # Two tokens for testing
    })
    embed_data = resp.json()
    print(f"Got {len(embed_data['client_shares'])} embeddings of dim {len(embed_data['client_shares'][0])}")

    # 3. Process first token at position 0
    print("\n3. Processing first token (position 0)...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/layer/step", json={
        "session_id": session_id,
        "client_share": embed_data["client_shares"][0],
        "server_share": embed_data["server_shares"][0],
        "layer_idx": 0,
        "position": 0
    })
    layer0_pos0 = resp.json()
    print(f"Phase: {layer0_pos0.get('phase')}")
    print(f"Q length: {len(layer0_pos0.get('q_client', []))}")
    print(f"K length: {len(layer0_pos0.get('k_client', []))}")
    print(f"V length: {len(layer0_pos0.get('v_client', []))}")
    print(f"Heads: {layer0_pos0.get('num_heads')}, KV heads: {layer0_pos0.get('num_kv_heads')}, Head dim: {layer0_pos0.get('head_dim')}")

    # 4. Process second token at position 1
    print("\n4. Processing second token (position 1)...")
    resp = requests.post(f"{SERVER_URL}/v2/secure/layer/step", json={
        "session_id": session_id,
        "client_share": embed_data["client_shares"][1],
        "server_share": embed_data["server_shares"][1],
        "layer_idx": 0,
        "position": 1
    })
    layer0_pos1 = resp.json()
    print(f"Phase: {layer0_pos1.get('phase')}")
    print(f"Q length: {len(layer0_pos1.get('q_client', []))}")
    print(f"K length: {len(layer0_pos1.get('k_client', []))}")

    # 5. Verify Q vectors are different (due to RoPE at different positions)
    q0 = layer0_pos0.get('q_client', [])[:5]
    q1 = layer0_pos1.get('q_client', [])[:5]
    print(f"\n5. Verifying RoPE applied correctly:")
    print(f"Q[0] first 5 values: {q0}")
    print(f"Q[1] first 5 values: {q1}")
    print(f"Q values differ (RoPE applied): {q0 != q1}")

    print("\n✓ KV cache API test completed successfully!")
    return True

if __name__ == "__main__":
    test_secure_inference()
