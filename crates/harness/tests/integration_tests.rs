//! ShardLM v1 Integration Tests
//!
//! These tests implement the validation requirements from TEST_PLAN.md

use shardlm_fixed_point::{Fixed, FixedVector, DEFAULT_SCALE};
use shardlm_harness::{PlaintextInference, SecureInference, ShardLmPipeline};
use shardlm_model::{EmbeddingTable, LinearWeights, ModelWeights, TinyLlamaConfig};
use shardlm_ot::{OtReceiver, OtSender, OtSessionConfig, SimpleOtExtension};
use shardlm_protocol::{
    EmbedFetchRequestPayload, EmbedFetchResponsePayload, ErrorCode, ErrorPayload, Frame, Header,
    Message, MsgType, OtBaseOtMsgPayload, OtSessionInitPayload, OtSessionParamsPayload,
    OtSessionReadyPayload, Payload,
};
use shardlm_sharing::{SharedMatrix, SharedVector, SharePair};

// =============================================================================
// Section 1: Protocol & Wire Format Tests
// =============================================================================

mod protocol_tests {
    use super::*;

    /// Test encode → decode round-trip for all message types
    #[test]
    fn test_frame_roundtrip_all_message_types() {
        // OT_SESSION_INIT
        let init = OtSessionInitPayload::default();
        let header = Header::pre_session();
        let frame = init.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::OtSessionInit);

        // OT_SESSION_PARAMS
        let params = OtSessionParamsPayload::default();
        let header = Header::new([0x11; 16], 0);
        let frame = params.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::OtSessionParams);

        // OT_BASE_OT_MSG
        let base_ot = OtBaseOtMsgPayload::new(1, false, vec![1, 2, 3]);
        let header = Header::new([0x22; 16], 0);
        let frame = base_ot.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::OtBaseOtMsg);

        // OT_SESSION_READY
        let ready = OtSessionReadyPayload::default();
        let header = Header::new([0x33; 16], 0);
        let frame = ready.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::OtSessionReady);

        // EMBED_FETCH_REQUEST
        let request = EmbedFetchRequestPayload::new(32, vec![4, 5, 6]).unwrap();
        let header = Header::new([0x44; 16], 1);
        let frame = request.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::EmbedFetchRequest);

        // EMBED_FETCH_RESPONSE
        let response = EmbedFetchResponsePayload::new(32, 8192, vec![7, 8, 9]);
        let header = Header::new([0x55; 16], 1);
        let frame = response.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::EmbedFetchResponse);

        // ERROR
        let error = ErrorPayload::new(ErrorCode::SessionNotFound, "Session not found");
        let header = Header::new([0x66; 16], 0);
        let frame = error.into_frame(header);
        let encoded = frame.encode();
        let decoded = Frame::decode(&encoded).unwrap();
        assert_eq!(decoded.msg_type, MsgType::Error);
    }

    /// Test corrupted header CRC is rejected
    #[test]
    fn test_header_crc_corruption() {
        let frame = Frame::new(MsgType::OtSessionInit, Header::pre_session(), vec![]);
        let mut encoded = frame.encode();

        // Corrupt the header CRC (bytes 14-17 in the frame)
        encoded[14] ^= 0xFF;

        let result = Frame::decode(&encoded);
        assert!(
            matches!(result, Err(shardlm_protocol::ProtocolError::HeaderCrcMismatch { .. })),
            "Expected HeaderCrcMismatch error"
        );
    }

    /// Test corrupted payload CRC is rejected
    #[test]
    fn test_payload_crc_corruption() {
        let frame = Frame::new(
            MsgType::OtSessionInit,
            Header::pre_session(),
            vec![1, 2, 3, 4, 5],
        );
        let mut encoded = frame.encode();

        // Corrupt the last byte of payload
        let last = encoded.len() - 1;
        encoded[last] ^= 0xFF;

        let result = Frame::decode(&encoded);
        assert!(
            matches!(result, Err(shardlm_protocol::ProtocolError::PayloadCrcMismatch { .. })),
            "Expected PayloadCrcMismatch error"
        );
    }

    /// Test version mismatch returns error
    #[test]
    fn test_version_mismatch() {
        let frame = Frame::new(MsgType::OtSessionInit, Header::pre_session(), vec![]);
        let mut encoded = frame.encode();

        // Corrupt version (bytes 4-5)
        encoded[4] = 0xFF;
        encoded[5] = 0xFF;

        let result = Frame::decode(&encoded);
        assert!(
            matches!(result, Err(shardlm_protocol::ProtocolError::VersionMismatch { .. })),
            "Expected VersionMismatch error"
        );
    }

    /// Test invalid magic bytes
    #[test]
    fn test_invalid_magic() {
        let frame = Frame::new(MsgType::OtSessionInit, Header::pre_session(), vec![]);
        let mut encoded = frame.encode();

        // Corrupt magic (bytes 0-3)
        encoded[0] = 0x00;
        encoded[1] = 0x00;

        let result = Frame::decode(&encoded);
        assert!(
            matches!(result, Err(shardlm_protocol::ProtocolError::InvalidMagic)),
            "Expected InvalidMagic error"
        );
    }
}

// =============================================================================
// Section 2: OT Session Lifecycle Tests
// =============================================================================

mod ot_session_tests {
    use super::*;

    /// Test valid session init leads to session ready
    #[test]
    fn test_session_creation() {
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 64,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Session init
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        // Verify session parameters match
        assert_eq!(receiver.config().vocab_size, 100);
        assert_eq!(receiver.config().hidden_dim, 64);

        // Base OT
        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        // Mark ready
        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        assert!(sender.is_ready());
        assert!(receiver.is_ready());
    }

    /// Test first request must use ctr = 1
    #[test]
    fn test_counter_starts_at_one() {
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 64,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Quick setup
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Counter should start at 1
        assert_eq!(receiver.counter(), 1);
    }

    /// Test counter increments strictly by 1
    #[test]
    fn test_counter_increments() {
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create database
        let db: Vec<u8> = (0..100 * 8 * 4).map(|i| i as u8).collect();
        sender.set_embedding_db(db);

        // Setup session
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Make multiple requests
        for i in 1..=5 {
            assert_eq!(receiver.counter(), i as u64);

            let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&[0]).unwrap();
            let (_, fetch_response) = sender
                .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
                .unwrap();
            receiver
                .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
                .unwrap();

            assert_eq!(receiver.counter(), (i + 1) as u64);
        }
    }
}

// =============================================================================
// Section 3: Embedding OT Correctness
// =============================================================================

mod embedding_ot_tests {
    use super::*;

    /// Test single lookup retrieves correct row
    #[test]
    fn test_single_embedding_lookup() {
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create known database
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
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Fetch single row
        let token_id = 42u32;
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&[token_id]).unwrap();
        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();
        let result = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        // Verify correct row
        let expected_first_elem = (token_id * 8) as i32;
        let got_first_elem = i32::from_le_bytes([result[0], result[1], result[2], result[3]]);
        assert_eq!(expected_first_elem, got_first_elem, "First element should match");
    }

    /// Test batched lookup retrieves correct rows in order
    #[test]
    fn test_batched_embedding_lookup() {
        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create known database
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
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Fetch multiple rows
        let token_ids = vec![5u32, 10, 15, 20, 99];
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&token_ids).unwrap();
        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();
        let result = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        // Verify all rows
        let row_bytes = 8 * 4;
        for (i, &token_id) in token_ids.iter().enumerate() {
            let start = i * row_bytes;
            let expected_first = (token_id * 8) as i32;
            let got_first = i32::from_le_bytes([
                result[start],
                result[start + 1],
                result[start + 2],
                result[start + 3],
            ]);
            assert_eq!(
                expected_first, got_first,
                "Row {} (token {}) mismatch",
                i, token_id
            );
        }
    }

    /// Test max batch size (64 tokens)
    #[test]
    fn test_max_batch_size() {
        let config = OtSessionConfig {
            vocab_size: 1000,
            hidden_dim: 8,
            max_prompt_len: 64,
            ..Default::default()
        };

        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create database
        let db: Vec<u8> = (0..1000 * 8 * 4).map(|i| i as u8).collect();
        sender.set_embedding_db(db);

        // Setup session
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        if let Some((_, response_msg)) =
            sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap()
        {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        // Fetch exactly 64 tokens
        let token_ids: Vec<u32> = (0..64).collect();
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&token_ids).unwrap();
        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();
        let result = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        assert_eq!(result.len(), 64 * 8 * 4);
    }
}

// =============================================================================
// Section 4: Fixed-Point Encoding Tests
// =============================================================================

mod fixed_point_tests {
    use super::*;

    /// Test encode → decode round-trip
    #[test]
    fn test_fixed_point_roundtrip() {
        let test_values = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 0.001, -0.001, 100.0, -100.0, 0.123456,
        ];

        for val in test_values {
            let fixed = Fixed::from_f64_default(val).unwrap();
            let back = fixed.to_f64();
            let error = (val - back).abs();
            assert!(
                error < 0.0001,
                "Roundtrip error for {}: got {}, error {}",
                val,
                back,
                error
            );
        }
    }

    /// Test scaling factor is consistent
    #[test]
    fn test_scaling_factor_consistency() {
        let val = 1.0;

        // Scale 16: 1.0 should be 65536
        let fixed_16 = Fixed::from_f64(val, 16).unwrap();
        assert_eq!(fixed_16.raw, 65536);

        // Scale 12: 1.0 should be 4096
        let fixed_12 = Fixed::from_f64(val, 12).unwrap();
        assert_eq!(fixed_12.raw, 4096);
    }

    /// Test boundary values
    #[test]
    fn test_boundary_values() {
        // Maximum positive value that fits in i32 with scale 16
        // i32::MAX / 65536 ≈ 32767
        let max_val = 32000.0;
        let result = Fixed::from_f64_default(max_val);
        assert!(result.is_ok());

        // Value too large should fail
        let too_large = 100000.0;
        let result = Fixed::from_f64_default(too_large);
        assert!(result.is_err());
    }
}

// =============================================================================
// Section 5: Secret Sharing Tests
// =============================================================================

mod secret_sharing_tests {
    use super::*;

    /// Test share generation and recombination
    #[test]
    fn test_share_recombination() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, -1.0, 0.5]).unwrap();
        let shares = SharePair::from_plaintext(&plaintext);
        let reconstructed = shares.reconstruct().unwrap();

        assert_eq!(plaintext.data, reconstructed.data);
    }

    /// Test seeded sharing is deterministic
    #[test]
    fn test_deterministic_sharing() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();

        let shares1 = SharePair::from_plaintext_seeded(&plaintext, 12345);
        let shares2 = SharePair::from_plaintext_seeded(&plaintext, 12345);

        assert_eq!(shares1.server.data, shares2.server.data);
        assert_eq!(shares1.client.data, shares2.client.data);
    }

    /// Test server share alone reveals nothing
    #[test]
    fn test_server_share_privacy() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let shares = SharePair::from_plaintext(&plaintext);

        // Server share should not equal plaintext
        assert_ne!(shares.server.data, plaintext.data);

        // Server share should be uniformly random (statistically)
        // Basic check: not all zeros
        assert!(shares.server.data.iter().any(|&x| x != 0));
    }
}

// =============================================================================
// Section 6: Secure Linear Layer Tests
// =============================================================================

mod secure_linear_tests {
    use super::*;

    /// Test shared linear computation matches plaintext
    #[test]
    fn test_linear_correctness() {
        // Create simple weight matrix
        // W = [[1, 0, 0],
        //      [0, 2, 0],
        //      [0, 0, 3]]
        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let w_data: Vec<i32> = vec![
            (1.0 * scale_factor) as i32, 0, 0,
            0, (2.0 * scale_factor) as i32, 0,
            0, 0, (3.0 * scale_factor) as i32,
        ];
        let w = SharedMatrix::from_raw(w_data, 3, 3, DEFAULT_SCALE).unwrap();

        // Input x = [1, 2, 3]
        let x = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();

        // Expected output: Y = [1, 4, 9]
        let shared_x = SharedVector::from_plaintext(&x);
        let shared_y = w.multiply_shared(&shared_x).unwrap();
        let y = shared_y.reconstruct().unwrap();

        let y_f64 = y.to_f64_vec();
        assert!((y_f64[0] - 1.0).abs() < 0.1, "Got {}", y_f64[0]);
        assert!((y_f64[1] - 4.0).abs() < 0.1, "Got {}", y_f64[1]);
        assert!((y_f64[2] - 9.0).abs() < 0.1, "Got {}", y_f64[2]);
    }

    /// Test that neither share alone reveals the result
    #[test]
    fn test_linear_masking() {
        let w_data: Vec<i32> = (0..9).map(|i| i * 1000).collect();
        let w = SharedMatrix::from_raw(w_data, 3, 3, DEFAULT_SCALE).unwrap();

        let x = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0]).unwrap();
        let shared_x = SharedVector::from_plaintext(&x);
        let shared_y = w.multiply_shared(&shared_x).unwrap();

        // Neither share alone should reveal the output
        // The shares should be random-looking
        assert_ne!(shared_y.client_share.data, shared_y.server_share.data);
    }
}

// =============================================================================
// Section 7: End-to-End Slice Test
// =============================================================================

mod end_to_end_tests {
    use super::*;

    /// Test full pipeline with small model
    #[test]
    fn test_end_to_end_pipeline() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            ..Default::default()
        };

        let pipeline = ShardLmPipeline::new(config);

        // Run inference
        let token_ids = vec![5, 10, 15];
        let result = pipeline.run_secure(&[5, 10, 15]).unwrap();

        // Verify output shape
        assert_eq!(result.logits.len(), 100, "Should have vocab_size logits");

        // Verify we got top-k predictions
        assert_eq!(result.top_k.len(), 5);
    }

    /// Test secure embedding retrieval matches plaintext
    #[test]
    fn test_secure_embeddings_match_plaintext() {
        let config = TinyLlamaConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            ..Default::default()
        };

        let weights = ModelWeights::random(config, DEFAULT_SCALE);
        let pipeline = ShardLmPipeline::with_shared_weights(weights).unwrap();

        // Get embeddings via both methods
        let plaintext_embs = pipeline.plaintext.get_embeddings(&[5, 10, 15]);
        let secure_embs = pipeline.secure.secure_embedding_fetch(&[5, 10, 15]).unwrap();

        // Compare
        assert_eq!(plaintext_embs.len(), secure_embs.len());
        for (p, s) in plaintext_embs.iter().zip(&secure_embs) {
            assert_eq!(p.data, s.data, "Embeddings should match exactly");
        }
    }

    /// Test with random prompt ≤64 tokens
    #[test]
    fn test_random_prompt() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let config = TinyLlamaConfig {
            vocab_size: 1000,
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            ..Default::default()
        };

        let pipeline = ShardLmPipeline::new(config);

        // Generate random prompt
        let prompt_len = rng.gen_range(1..=64);
        let token_ids: Vec<u32> = (0..prompt_len)
            .map(|_| rng.gen_range(0..1000))
            .collect();

        // Run secure inference
        let result = pipeline.run_secure(&token_ids).unwrap();

        // Basic sanity checks
        assert_eq!(result.logits.len(), 1000);
        assert!(!result.top_k.is_empty());
    }
}

// =============================================================================
// Section 8: Determinism & Reproducibility
// =============================================================================

mod determinism_tests {
    use super::*;

    /// Test fixed-point operations are deterministic
    #[test]
    fn test_fixed_point_deterministic() {
        let a = Fixed::from_f64_default(1.5).unwrap();
        let b = Fixed::from_f64_default(2.5).unwrap();

        // Same operations should produce same results
        let sum1 = a.add(b).unwrap();
        let sum2 = a.add(b).unwrap();
        assert_eq!(sum1.raw, sum2.raw);

        let prod1 = a.mul(b).unwrap();
        let prod2 = a.mul(b).unwrap();
        assert_eq!(prod1.raw, prod2.raw);
    }

    /// Test seeded operations are reproducible
    #[test]
    fn test_seeded_reproducibility() {
        let plaintext = FixedVector::from_f64_slice_default(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Same seed should produce identical shares
        let shares1 = SharePair::from_plaintext_seeded(&plaintext, 999);
        let shares2 = SharePair::from_plaintext_seeded(&plaintext, 999);

        assert_eq!(shares1.client.data, shares2.client.data);
        assert_eq!(shares1.server.data, shares2.server.data);

        // Different seeds should produce different shares
        let shares3 = SharePair::from_plaintext_seeded(&plaintext, 1000);
        assert_ne!(shares1.server.data, shares3.server.data);
    }

    /// Test linear layer computation is deterministic
    #[test]
    fn test_linear_deterministic() {
        let w = SharedMatrix::from_raw(vec![1, 2, 3, 4], 2, 2, DEFAULT_SCALE).unwrap();
        let x = FixedVector::from_raw(vec![100, 200], DEFAULT_SCALE);

        let shared_x1 = SharedVector::from_plaintext_seeded(&x, 42);
        let shared_y1 = w.multiply_shared(&shared_x1).unwrap();
        let y1 = shared_y1.reconstruct().unwrap();

        let shared_x2 = SharedVector::from_plaintext_seeded(&x, 42);
        let shared_y2 = w.multiply_shared(&shared_x2).unwrap();
        let y2 = shared_y2.reconstruct().unwrap();

        assert_eq!(y1.data, y2.data);
    }
}

// =============================================================================
// Section 9: Transformer Components Integration Test
// =============================================================================

mod transformer_components_tests {
    use shardlm_fixed_point::{DEFAULT_SCALE, FixedVector};
    use shardlm_sharing::{
        RmsNormClient, RmsNormConfig, compute_rmsnorm,
        RopeFrequencies, apply_rope_to_q, apply_rope_to_k,
        AttentionClient, AttentionServer, compute_attention_with_rope,
        KvCacheClient, KvCacheServer,
        share_kv, reconstruct_kv,
    };

    /// Test RMSNorm integration
    #[test]
    fn test_rmsnorm_integration() {
        let config = RmsNormConfig {
            hidden_size: 64,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let norm = RmsNormClient::with_unit_weights(&config);

        // Create input
        let input: Vec<f64> = (0..64).map(|i| (i as f64 - 32.0) * 0.1).collect();
        let input_vec = FixedVector::from_f64_slice_default(&input).unwrap();

        // Share input
        let (client_share, server_share) = share_kv(&input_vec);

        // Apply RMSNorm
        let (out_client, out_server) = compute_rmsnorm(&norm, &client_share, &server_share).unwrap();

        // Reconstruct and verify
        let output = reconstruct_kv(&out_client, &out_server).unwrap();
        assert_eq!(output.len(), 64);

        // Verify RMS is ~1 (unit norm)
        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let out_f64: Vec<f64> = output.data.iter().map(|&v| v as f64 / scale_factor).collect();
        let rms: f64 = (out_f64.iter().map(|&x| x * x).sum::<f64>() / 64.0).sqrt();
        assert!((rms - 1.0).abs() < 0.1, "RMS should be approximately 1, got {}", rms);
    }

    /// Test RoPE integration with attention
    #[test]
    fn test_rope_attention_integration() {
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        let seq_len = 4;

        let client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let server = AttentionServer::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let freqs = RopeFrequencies::new(head_dim, 128, 10000.0);

        // Create Q for current position
        let q_data: Vec<f64> = (0..num_heads * head_dim).map(|i| (i as f64) * 0.1).collect();
        let q = FixedVector::from_f64_slice_default(&q_data).unwrap();
        let (q_client, q_server) = share_kv(&q);

        // Create K and V for previous positions
        let mut k_client_shares = Vec::new();
        let mut k_server_shares = Vec::new();
        let mut v_client_shares = Vec::new();
        let mut v_server_shares = Vec::new();

        for pos in 0..seq_len {
            let k_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((pos * 10 + i) as f64) * 0.05)
                .collect();
            let k = FixedVector::from_f64_slice_default(&k_data).unwrap();
            let (kc, ks) = share_kv(&k);
            k_client_shares.push(kc);
            k_server_shares.push(ks);

            let v_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((pos + i) as f64) * 0.1)
                .collect();
            let v = FixedVector::from_f64_slice_default(&v_data).unwrap();
            let (vc, vs) = share_kv(&v);
            v_client_shares.push(vc);
            v_server_shares.push(vs);
        }

        // Compute attention with RoPE at the last position
        let current_position = seq_len - 1;
        let output = compute_attention_with_rope(
            &client,
            &server,
            &q_client,
            &q_server,
            &k_client_shares,
            &k_server_shares,
            &v_client_shares,
            &v_server_shares,
            current_position,
            &freqs,
        ).unwrap();

        // Verify output dimensions
        assert_eq!(output.client_share.len(), num_heads * head_dim);
        assert_eq!(output.server_share.len(), num_heads * head_dim);

        // Verify reconstruction
        let reconstructed = reconstruct_kv(&output.client_share, &output.server_share).unwrap();
        assert_eq!(reconstructed.len(), num_heads * head_dim);
    }

    /// Test KV cache integration
    #[test]
    fn test_kv_cache_integration() {
        let num_layers = 2;
        let num_kv_heads = 2;
        let head_dim = 8;
        let max_seq_len = 32;

        let mut kv_client = KvCacheClient::new(num_layers, num_kv_heads, head_dim, max_seq_len, DEFAULT_SCALE);
        let mut kv_server = KvCacheServer::new(num_layers, num_kv_heads, head_dim, max_seq_len, DEFAULT_SCALE);

        // Simulate multi-token generation
        let kv_dim = num_kv_heads * head_dim;
        for token_idx in 0..5 {
            for layer in 0..num_layers {
                // Create K, V for this position
                let k_data: Vec<f64> = (0..kv_dim).map(|i| (token_idx * 10 + i) as f64 * 0.1).collect();
                let v_data: Vec<f64> = (0..kv_dim).map(|i| (token_idx + i) as f64 * 0.2).collect();

                let k = FixedVector::from_f64_slice_default(&k_data).unwrap();
                let v = FixedVector::from_f64_slice_default(&v_data).unwrap();

                // Share K and V
                let (k_c, k_s) = share_kv(&k);
                let (v_c, v_s) = share_kv(&v);

                // Append to both caches
                kv_client.append(layer, k_c, v_c).unwrap();
                kv_server.append(layer, k_s, v_s).unwrap();
            }
        }

        // Verify cache lengths match
        for layer in 0..num_layers {
            let client_keys = kv_client.get_k(layer).unwrap();
            let server_keys = kv_server.get_k(layer).unwrap();
            assert_eq!(client_keys.len(), 5, "Client should have 5 keys");
            assert_eq!(server_keys.len(), 5, "Server should have 5 keys");

            // Reconstruct first key and verify
            let k_reconstructed = reconstruct_kv(&client_keys[0], &server_keys[0]).unwrap();
            assert_eq!(k_reconstructed.len(), kv_dim);
        }
    }

    /// Test full transformer layer simulation
    #[test]
    fn test_transformer_layer_simulation() {
        // This simulates one forward pass through a transformer layer
        let hidden_size = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads;

        // Create input hidden state
        let input: Vec<f64> = (0..hidden_size).map(|i| (i as f64 - 16.0) * 0.1).collect();
        let input_vec = FixedVector::from_f64_slice_default(&input).unwrap();
        let (input_client, input_server) = share_kv(&input_vec);

        // Step 1: RMSNorm
        let norm_config = RmsNormConfig {
            hidden_size,
            eps: 1e-5,
            scale: DEFAULT_SCALE,
        };
        let norm = RmsNormClient::with_unit_weights(&norm_config);
        let (normed_client, normed_server) = compute_rmsnorm(&norm, &input_client, &input_server).unwrap();

        // Verify normed output
        let normed = reconstruct_kv(&normed_client, &normed_server).unwrap();
        assert_eq!(normed.len(), hidden_size);

        // Step 2: Simulate Q, K, V projection (just reshaping for this test)
        let q_data: Vec<f64> = (0..num_heads * head_dim).map(|i| (i as f64) * 0.1).collect();
        let k_data: Vec<f64> = (0..num_kv_heads * head_dim).map(|i| (i as f64) * 0.2).collect();
        let v_data: Vec<f64> = (0..num_kv_heads * head_dim).map(|i| (i as f64) * 0.3).collect();

        let q = FixedVector::from_f64_slice_default(&q_data).unwrap();
        let k = FixedVector::from_f64_slice_default(&k_data).unwrap();
        let v = FixedVector::from_f64_slice_default(&v_data).unwrap();

        let (q_client, q_server) = share_kv(&q);
        let (k_client, k_server) = share_kv(&k);
        let (v_client, v_server) = share_kv(&v);

        // Step 3: Apply RoPE to Q and K
        let freqs = RopeFrequencies::new(head_dim, 128, 10000.0);
        let position = 0;

        let q_reconstructed = reconstruct_kv(&q_client, &q_server).unwrap();
        let q_rotated = apply_rope_to_q(&q_reconstructed, position, &freqs, num_heads).unwrap();

        let k_reconstructed = reconstruct_kv(&k_client, &k_server).unwrap();
        let k_rotated = apply_rope_to_k(&k_reconstructed, position, &freqs, num_kv_heads).unwrap();

        // At position 0, RoPE should be identity
        assert_eq!(q_reconstructed.data, q_rotated.data);
        assert_eq!(k_reconstructed.data, k_rotated.data);

        // Step 4: Attention (with single K, V in cache)
        let attn_client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let attn_server = AttentionServer::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);

        let attn_output = compute_attention_with_rope(
            &attn_client,
            &attn_server,
            &q_client,
            &q_server,
            &[k_client],
            &[k_server],
            &[v_client],
            &[v_server],
            position,
            &freqs,
        ).unwrap();

        // Verify attention output
        let attn_out = reconstruct_kv(&attn_output.client_share, &attn_output.server_share).unwrap();
        assert_eq!(attn_out.len(), num_heads * head_dim);

        // Step 5: Residual connection (in plaintext for verification)
        let scale_factor = (1u64 << DEFAULT_SCALE) as f64;
        let residual: Vec<i32> = input_vec.data.iter()
            .zip(&attn_out.data)
            .map(|(&x, &y)| x.wrapping_add(y))
            .collect();
        let residual_vec = FixedVector::from_raw(residual, DEFAULT_SCALE);
        assert_eq!(residual_vec.len(), hidden_size);
    }

    /// Test multi-token generation flow
    #[test]
    fn test_multi_token_generation_flow() {
        let hidden_size = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads;
        let num_tokens = 4;

        let freqs = RopeFrequencies::new(head_dim, 128, 10000.0);
        let attn_client = AttentionClient::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);
        let attn_server = AttentionServer::new(num_heads, num_kv_heads, head_dim, DEFAULT_SCALE);

        // Simulate KV cache growing with each token
        let mut k_client_shares = Vec::new();
        let mut k_server_shares = Vec::new();
        let mut v_client_shares = Vec::new();
        let mut v_server_shares = Vec::new();

        for token_idx in 0..num_tokens {
            // Create Q for current token (smaller values to avoid overflow)
            let q_data: Vec<f64> = (0..num_heads * head_dim)
                .map(|i| ((i % 8) as f64) * 0.01)
                .collect();
            let q = FixedVector::from_f64_slice_default(&q_data).unwrap();
            let (q_client, q_server) = share_kv(&q);

            // Create K, V for current token (smaller values to avoid overflow)
            let k_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((i % 8) as f64) * 0.01)
                .collect();
            let v_data: Vec<f64> = (0..num_kv_heads * head_dim)
                .map(|i| ((i % 8) as f64) * 0.01)
                .collect();

            let k = FixedVector::from_f64_slice_default(&k_data).unwrap();
            let v = FixedVector::from_f64_slice_default(&v_data).unwrap();

            let (k_client, k_server) = share_kv(&k);
            let (v_client, v_server) = share_kv(&v);

            // Add to cache
            k_client_shares.push(k_client);
            k_server_shares.push(k_server);
            v_client_shares.push(v_client);
            v_server_shares.push(v_server);

            // Compute attention with growing cache
            let output = compute_attention_with_rope(
                &attn_client,
                &attn_server,
                &q_client,
                &q_server,
                &k_client_shares,
                &k_server_shares,
                &v_client_shares,
                &v_server_shares,
                token_idx,
                &freqs,
            ).unwrap();

            // Verify output
            let reconstructed = reconstruct_kv(&output.client_share, &output.server_share).unwrap();
            assert_eq!(reconstructed.len(), num_heads * head_dim);
        }

        // Final verification: cache should have all tokens
        assert_eq!(k_client_shares.len(), num_tokens);
        assert_eq!(v_client_shares.len(), num_tokens);
    }
}
