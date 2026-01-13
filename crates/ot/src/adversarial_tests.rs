//! Adversarial security tests for OT protocol
//!
//! These tests validate that the OT implementation correctly enforces
//! security properties against malicious clients or servers.

#[cfg(test)]
mod tests {
    use crate::extension::SimpleOtExtension;
    use crate::iknp::IknpOtExtension;
    use crate::receiver::OtReceiver;
    use crate::sender::OtSender;
    use crate::session::OtSessionConfig;
    use crate::OtError;

    /// Helper to set up a complete OT session
    fn setup_session() -> (OtSender<IknpOtExtension>, OtReceiver<IknpOtExtension>, Vec<u8>) {
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            max_prompt_len: 16,
            max_requests: 100,
            ..Default::default()
        };

        let sender_ext = IknpOtExtension::new_server();
        let receiver_ext = IknpOtExtension::new_client();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create embedding database
        let mut db = Vec::with_capacity(100 * 8 * 4);
        for row in 0..100u32 {
            for col in 0..8u32 {
                let val = (row * 8 + col) as i32;
                db.extend_from_slice(&val.to_le_bytes());
            }
        }
        sender.set_embedding_db(db.clone());

        // Complete session handshake
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) = sender
            .handle_session_init(&init_payload, init_header.client_nonce)
            .unwrap();
        receiver
            .handle_session_params(&params_payload, params_header.session_id)
            .unwrap();

        // Base OT
        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) = sender
            .handle_base_ot_msg(&base_msg, base_header.client_nonce)
            .unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        // Mark ready
        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        (sender, receiver, db)
    }

    // ==================== Counter Enforcement Tests ====================

    #[test]
    fn test_counter_replay_attack_rejected() {
        // Tests that replaying a request with the same counter is rejected
        let (mut sender, mut receiver, _) = setup_session();

        // First valid fetch
        let tokens = vec![5u32];
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&tokens).unwrap();
        let (_, _) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();

        // Attempt to replay with same counter
        let replay_result =
            sender.handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce);

        assert!(
            matches!(replay_result, Err(OtError::CounterMismatch { .. })),
            "Replay attack should be rejected with CounterMismatch"
        );
    }

    #[test]
    fn test_counter_skip_attack_rejected() {
        // Tests that skipping counters is rejected
        let (mut sender, mut receiver, _) = setup_session();

        // Generate fetch but with wrong counter
        let tokens = vec![5u32];
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&tokens).unwrap();

        // Try with skipped counter (expecting 1, sending 5)
        let skip_result = sender.handle_embed_fetch(&fetch_request, 5, fetch_header.client_nonce);

        assert!(
            matches!(
                skip_result,
                Err(OtError::CounterMismatch {
                    expected: 1,
                    got: 5
                })
            ),
            "Skipped counter should be rejected"
        );
    }

    #[test]
    fn test_counter_rollback_attack_rejected() {
        // Tests that sending a previous counter after advancing is rejected
        let (mut sender, mut receiver, _) = setup_session();

        // First fetch advances counter to 2
        let tokens = vec![5u32];
        let (fetch_header1, fetch_request1) = receiver.generate_embed_fetch(&tokens).unwrap();
        let (_, fetch_response1) = sender
            .handle_embed_fetch(&fetch_request1, fetch_header1.ctr, fetch_header1.client_nonce)
            .unwrap();
        receiver
            .handle_embed_fetch_response(&fetch_response1, fetch_header1.ctr)
            .unwrap();

        // Second fetch advances counter to 3
        let (fetch_header2, fetch_request2) = receiver.generate_embed_fetch(&tokens).unwrap();
        let (_, fetch_response2) = sender
            .handle_embed_fetch(&fetch_request2, fetch_header2.ctr, fetch_header2.client_nonce)
            .unwrap();
        receiver
            .handle_embed_fetch_response(&fetch_response2, fetch_header2.ctr)
            .unwrap();

        // Try to roll back to counter 1 (server is now at counter 3)
        let rollback_result =
            sender.handle_embed_fetch(&fetch_request1, 1, fetch_header1.client_nonce);

        assert!(
            matches!(rollback_result, Err(OtError::CounterMismatch { .. })),
            "Counter rollback should be rejected"
        );
    }

    #[test]
    fn test_counter_must_be_sequential() {
        // Tests that counters must be strictly sequential
        let (mut sender, mut receiver, _) = setup_session();

        for expected_ctr in 1..=10u64 {
            let tokens = vec![(expected_ctr % 100) as u32];
            let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&tokens).unwrap();

            assert_eq!(
                fetch_header.ctr, expected_ctr,
                "Counter should be sequential"
            );

            let result = sender.handle_embed_fetch(
                &fetch_request,
                fetch_header.ctr,
                fetch_header.client_nonce,
            );
            assert!(result.is_ok(), "Sequential counter {} should work", expected_ctr);

            let (_, response) = result.unwrap();
            receiver
                .handle_embed_fetch_response(&response, fetch_header.ctr)
                .unwrap();
        }
    }

    // ==================== Index Validation Tests ====================

    #[test]
    fn test_out_of_bounds_index_rejected() {
        // Tests that out-of-bounds token indices are rejected
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let sender_ext = IknpOtExtension::new_server();
        let receiver_ext = IknpOtExtension::new_client();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Setup session
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) = sender
            .handle_session_init(&init_payload, init_header.client_nonce)
            .unwrap();
        receiver
            .handle_session_params(&params_payload, params_header.session_id)
            .unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) = sender
            .handle_base_ot_msg(&base_msg, base_header.client_nonce)
            .unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Try to fetch out-of-bounds token (vocab_size = 100)
        let invalid_tokens = vec![150u32]; // > vocab_size
        let result = receiver.generate_embed_fetch(&invalid_tokens);

        assert!(
            matches!(result, Err(OtError::InvalidIndex { index: 150, max: 100 })),
            "Out-of-bounds index should be rejected"
        );
    }

    #[test]
    fn test_batch_size_exceeded_rejected() {
        // Tests that exceeding max batch size is rejected
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            max_prompt_len: 4, // Small for testing
            ..Default::default()
        };

        let sender_ext = IknpOtExtension::new_server();
        let receiver_ext = IknpOtExtension::new_client();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Setup session
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) = sender
            .handle_session_init(&init_payload, init_header.client_nonce)
            .unwrap();
        receiver
            .handle_session_params(&params_payload, params_header.session_id)
            .unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) = sender
            .handle_base_ot_msg(&base_msg, base_header.client_nonce)
            .unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Try to fetch more tokens than max_prompt_len
        let too_many_tokens: Vec<u32> = (0..10).collect(); // max_prompt_len = 4
        let result = receiver.generate_embed_fetch(&too_many_tokens);

        assert!(
            matches!(result, Err(OtError::BatchSizeExceeded { size: 10, max: 4 })),
            "Batch size exceeded should be rejected"
        );
    }

    // ==================== Session State Tests ====================

    #[test]
    fn test_fetch_before_ready_rejected() {
        // Tests that fetching before session is ready is rejected
        let _config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let receiver_ext = IknpOtExtension::new_client();
        let mut receiver = OtReceiver::new(receiver_ext);

        // Try to fetch without completing session setup
        let tokens = vec![5u32];
        let result = receiver.generate_embed_fetch(&tokens);

        assert!(
            matches!(result, Err(OtError::SessionNotInitialized)),
            "Fetch before ready should be rejected"
        );
    }

    #[test]
    fn test_double_session_init_rejected() {
        // Tests that initializing session twice is rejected
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let sender_ext = IknpOtExtension::new_server();
        let mut sender = OtSender::new(sender_ext, config);

        // First init
        let init = shardlm_protocol::OtSessionInitPayload::default();
        sender.handle_session_init(&init, [0u8; 8]).unwrap();

        // Second init should fail
        let result = sender.handle_session_init(&init, [0u8; 8]);
        assert!(
            matches!(result, Err(OtError::SessionAlreadyInitialized)),
            "Double session init should be rejected"
        );
    }

    #[test]
    fn test_max_requests_enforced() {
        // Tests that session expires after max_requests
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            max_prompt_len: 16,
            max_requests: 5, // Very low for testing
            ..Default::default()
        };

        let sender_ext = IknpOtExtension::new_server();
        let receiver_ext = IknpOtExtension::new_client();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create embedding database
        let mut db = Vec::with_capacity(100 * 8 * 4);
        for row in 0..100u32 {
            for col in 0..8u32 {
                let val = (row * 8 + col) as i32;
                db.extend_from_slice(&val.to_le_bytes());
            }
        }
        sender.set_embedding_db(db);

        // Setup session
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) = sender
            .handle_session_init(&init_payload, init_header.client_nonce)
            .unwrap();
        receiver
            .handle_session_params(&params_payload, params_header.session_id)
            .unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) = sender
            .handle_base_ot_msg(&base_msg, base_header.client_nonce)
            .unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Make max_requests fetches
        for i in 0..5 {
            let tokens = vec![(i % 100) as u32];
            let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&tokens).unwrap();
            let result = sender.handle_embed_fetch(
                &fetch_request,
                fetch_header.ctr,
                fetch_header.client_nonce,
            );
            assert!(result.is_ok(), "Request {} should succeed", i + 1);

            let (_, response) = result.unwrap();
            receiver
                .handle_embed_fetch_response(&response, fetch_header.ctr)
                .unwrap();
        }

        // Next request should fail due to expiration
        let tokens = vec![5u32];
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&tokens).unwrap();
        let result =
            sender.handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce);

        assert!(
            matches!(result, Err(OtError::SessionExpired)),
            "Session should expire after max_requests"
        );
    }

    // ==================== OT Correctness Tests ====================

    #[test]
    fn test_retrieved_embeddings_match_plaintext() {
        // Tests that OT-retrieved embeddings exactly match plaintext database
        let (mut sender, mut receiver, db) = setup_session();

        let token_ids: Vec<u32> = vec![0, 10, 50, 99]; // Various indices
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&token_ids).unwrap();
        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();
        let embedding_bytes = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        // Verify each retrieved embedding
        let row_bytes = 8 * 4; // hidden_dim * sizeof(i32)
        for (i, &token_id) in token_ids.iter().enumerate() {
            let ot_start = i * row_bytes;
            let db_start = token_id as usize * row_bytes;

            for j in 0..row_bytes {
                assert_eq!(
                    embedding_bytes[ot_start + j], db[db_start + j],
                    "Mismatch at token {} byte {}", token_id, j
                );
            }
        }
    }

    #[test]
    fn test_different_tokens_different_embeddings() {
        // Tests that different tokens retrieve different embeddings (unless they're the same token)
        let (mut sender, mut receiver, _) = setup_session();

        let tokens1 = vec![10u32];
        let (fetch_header1, fetch_request1) = receiver.generate_embed_fetch(&tokens1).unwrap();
        let (_, fetch_response1) = sender
            .handle_embed_fetch(&fetch_request1, fetch_header1.ctr, fetch_header1.client_nonce)
            .unwrap();
        let emb1 = receiver
            .handle_embed_fetch_response(&fetch_response1, fetch_header1.ctr)
            .unwrap();

        let tokens2 = vec![20u32];
        let (fetch_header2, fetch_request2) = receiver.generate_embed_fetch(&tokens2).unwrap();
        let (_, fetch_response2) = sender
            .handle_embed_fetch(&fetch_request2, fetch_header2.ctr, fetch_header2.client_nonce)
            .unwrap();
        let emb2 = receiver
            .handle_embed_fetch_response(&fetch_response2, fetch_header2.ctr)
            .unwrap();

        assert_ne!(emb1, emb2, "Different tokens should have different embeddings");
    }

    #[test]
    fn test_same_token_same_embedding() {
        // Tests that the same token retrieved multiple times gives the same embedding
        let (mut sender, mut receiver, _) = setup_session();

        let token = 42u32;

        let tokens1 = vec![token];
        let (fetch_header1, fetch_request1) = receiver.generate_embed_fetch(&tokens1).unwrap();
        let (_, fetch_response1) = sender
            .handle_embed_fetch(&fetch_request1, fetch_header1.ctr, fetch_header1.client_nonce)
            .unwrap();
        let emb1 = receiver
            .handle_embed_fetch_response(&fetch_response1, fetch_header1.ctr)
            .unwrap();

        let tokens2 = vec![token];
        let (fetch_header2, fetch_request2) = receiver.generate_embed_fetch(&tokens2).unwrap();
        let (_, fetch_response2) = sender
            .handle_embed_fetch(&fetch_request2, fetch_header2.ctr, fetch_header2.client_nonce)
            .unwrap();
        let emb2 = receiver
            .handle_embed_fetch_response(&fetch_response2, fetch_header2.ctr)
            .unwrap();

        assert_eq!(emb1, emb2, "Same token should always give same embedding");
    }

    // ==================== Simple OT Extension Tests (for comparison) ====================

    #[test]
    fn test_simple_ot_correctness() {
        // Tests that SimpleOtExtension also produces correct results
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create embedding database
        let mut db = Vec::with_capacity(100 * 8 * 4);
        for row in 0..100u32 {
            for col in 0..8u32 {
                let val = (row * 8 + col) as i32;
                db.extend_from_slice(&val.to_le_bytes());
            }
        }
        sender.set_embedding_db(db.clone());

        // Setup session
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) = sender
            .handle_session_init(&init_payload, init_header.client_nonce)
            .unwrap();
        receiver
            .handle_session_params(&params_payload, params_header.session_id)
            .unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) = sender
            .handle_base_ot_msg(&base_msg, base_header.client_nonce)
            .unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Fetch and verify
        let token_ids: Vec<u32> = vec![5, 10, 15];
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&token_ids).unwrap();
        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();
        let embedding_bytes = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        // Verify
        let row_bytes = 8 * 4;
        for (i, &token_id) in token_ids.iter().enumerate() {
            let ot_first = i32::from_le_bytes([
                embedding_bytes[i * row_bytes],
                embedding_bytes[i * row_bytes + 1],
                embedding_bytes[i * row_bytes + 2],
                embedding_bytes[i * row_bytes + 3],
            ]);
            let expected_first = (token_id * 8) as i32;
            assert_eq!(ot_first, expected_first, "SimpleOT should retrieve correct values");
        }
    }
}
