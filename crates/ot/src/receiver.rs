//! OT Receiver (Client side)

use shardlm_protocol::{
    EmbedFetchRequestPayload, EmbedFetchResponsePayload, Header, OtBaseOtMsgPayload,
    OtSessionInitPayload, OtSessionParamsPayload, OtSessionReadyPayload,
};

use crate::error::{OtError, Result};
use crate::extension::OtExtension;
use crate::session::{OtSession, OtSessionConfig};

/// OT Receiver (Client) manages session and generates OT queries
pub struct OtReceiver<E: OtExtension> {
    /// OT extension implementation
    extension: E,
    /// Current session
    session: OtSession,
    /// Last indices queried (for decoding response)
    last_indices: Vec<u32>,
}

impl<E: OtExtension> OtReceiver<E> {
    /// Create a new OT receiver with the given extension
    pub fn new(extension: E) -> Self {
        Self {
            extension,
            session: OtSession::new(OtSessionConfig::default()),
            last_indices: vec![],
        }
    }

    /// Get session ID
    pub fn session_id(&self) -> &[u8; 16] {
        &self.session.id
    }

    /// Get current counter
    pub fn counter(&self) -> u64 {
        self.session.counter
    }

    /// Generate OT_SESSION_INIT message
    pub fn generate_session_init(&self) -> (Header, OtSessionInitPayload) {
        let mut header = Header::pre_session();
        header.generate_client_nonce();

        let init = OtSessionInitPayload::default();
        (header, init)
    }

    /// Handle OT_SESSION_PARAMS response
    pub fn handle_session_params(&mut self, params: &OtSessionParamsPayload, session_id: [u8; 16]) -> Result<()> {
        // Update config from server params
        self.session.config.max_prompt_len = params.accepted_lmax;
        self.session.config.hidden_dim = params.accepted_d;
        self.session.config.vocab_size = params.vocab_size;
        self.session.config.scale = params.fixed_point_scale;
        self.session.config.ttl = std::time::Duration::from_secs(params.session_ttl_sec as u64);
        self.session.config.max_requests = params.max_requests;

        self.session.initialize_with_id(session_id)?;
        Ok(())
    }

    /// Generate initial base OT message
    pub fn generate_base_ot_init(&mut self) -> Result<(Header, OtBaseOtMsgPayload)> {
        let blob = self.extension.generate_base_ot_sender()?;

        let mut header = Header::new(self.session.id, 0);
        header.generate_client_nonce();

        let msg = OtBaseOtMsgPayload {
            phase: 1,
            is_final: false,
            blob,
        };

        Ok((header, msg))
    }

    /// Handle base OT response and optionally generate next message
    pub fn handle_base_ot_response(
        &mut self,
        response: &OtBaseOtMsgPayload,
    ) -> Result<Option<(Header, OtBaseOtMsgPayload)>> {
        // Client processes server's response (B points + commitment)
        let next_blob = self.extension.process_base_ot_sender(&response.blob)?;

        self.session.advance_base_ot_phase();

        if response.is_final || next_blob.is_none() {
            // Base OT is complete
            return Ok(None);
        }

        let mut header = Header::new(self.session.id, 0);
        header.generate_client_nonce();

        let msg = OtBaseOtMsgPayload {
            phase: self.session.base_ot_phase,
            is_final: self.extension.is_base_ot_complete(),
            blob: next_blob.unwrap_or_default(),
        };

        Ok(Some((header, msg)))
    }

    /// Handle OT_SESSION_READY message
    pub fn handle_session_ready(&mut self, ready: &OtSessionReadyPayload) -> Result<()> {
        if !ready.ok {
            return Err(OtError::SessionNotInitialized);
        }
        self.session.mark_ready()?;
        Ok(())
    }

    /// Generate EMBED_FETCH_REQUEST for the given token indices
    pub fn generate_embed_fetch(&mut self, token_ids: &[u32]) -> Result<(Header, EmbedFetchRequestPayload)> {
        if !self.session.is_ready() {
            return Err(OtError::SessionNotInitialized);
        }
        if token_ids.len() > self.session.config.max_prompt_len as usize {
            return Err(OtError::BatchSizeExceeded {
                size: token_ids.len(),
                max: self.session.config.max_prompt_len as usize,
            });
        }

        // Validate indices
        for &id in token_ids {
            if id >= self.session.config.vocab_size {
                return Err(OtError::InvalidIndex {
                    index: id,
                    max: self.session.config.vocab_size,
                });
            }
        }

        // Store indices for decoding
        self.last_indices = token_ids.to_vec();

        let ctr = self.session.counter;
        let query_blob = self
            .extension
            .generate_query(token_ids, &self.session.id, ctr)?;

        let mut header = Header::new(self.session.id, ctr);
        header.generate_client_nonce();

        let request = EmbedFetchRequestPayload::new(token_ids.len() as u16, query_blob)?;

        Ok((header, request))
    }

    /// Handle EMBED_FETCH_RESPONSE and decode embeddings
    pub fn handle_embed_fetch_response(
        &mut self,
        response: &EmbedFetchResponsePayload,
        ctr: u64,
    ) -> Result<Vec<u8>> {
        // Validate counter matches
        if ctr != self.session.counter {
            return Err(OtError::CounterMismatch {
                expected: self.session.counter,
                got: ctr,
            });
        }

        // Increment counter for next request
        self.session.counter += 1;
        self.session.request_count += 1;

        let row_bytes = response.row_bytes as usize;
        let num_items = response.len as usize;

        // Decode the OT response
        let embeddings =
            self.extension
                .decode_response(&response.response_blob, num_items, row_bytes)?;

        Ok(embeddings)
    }

    /// Check if session is ready
    pub fn is_ready(&self) -> bool {
        self.session.is_ready()
    }

    /// Get session config
    pub fn config(&self) -> &OtSessionConfig {
        &self.session.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::SimpleOtExtension;
    use crate::sender::OtSender;

    #[test]
    fn test_full_ot_flow() {
        // Setup sender (server) and receiver (client)
        let sender_ext = SimpleOtExtension::new();
        let receiver_ext = SimpleOtExtension::new();

        let config = OtSessionConfig {
            vocab_size: 100,
            hidden_dim: 8, // Small for testing
            ..Default::default()
        };

        let mut sender = OtSender::new(sender_ext, config.clone());
        let mut receiver = OtReceiver::new(receiver_ext);

        // Create a small embedding database (100 rows × 8 dims × 4 bytes = 3200 bytes)
        let mut db = Vec::with_capacity(100 * 8 * 4);
        for row in 0..100u32 {
            for col in 0..8u32 {
                let val = (row * 8 + col) as i32;
                db.extend_from_slice(&val.to_le_bytes());
            }
        }
        sender.set_embedding_db(db);

        // Step 1: Session init
        let (init_header, init_payload) = receiver.generate_session_init();
        let (params_header, params_payload) =
            sender.handle_session_init(&init_payload, init_header.client_nonce).unwrap();
        receiver.handle_session_params(&params_payload, params_header.session_id).unwrap();

        // Step 2: Base OT (simplified - in real protocol this would be multiple rounds)
        let (base_header, base_msg) = receiver.generate_base_ot_init().unwrap();
        let base_response = sender.handle_base_ot_msg(&base_msg, base_header.client_nonce).unwrap();

        if let Some((_, response_msg)) = base_response {
            receiver.handle_base_ot_response(&response_msg).unwrap();
        }

        // Step 3: Mark ready
        let (_, ready_payload) = sender.generate_session_ready([0u8; 8]).unwrap();
        receiver.handle_session_ready(&ready_payload).unwrap();

        assert!(sender.is_ready());
        assert!(receiver.is_ready());

        // Step 4: Fetch embeddings for tokens [5, 10, 15]
        let token_ids = vec![5, 10, 15];
        let (fetch_header, fetch_request) = receiver.generate_embed_fetch(&token_ids).unwrap();
        let (_, fetch_response) = sender
            .handle_embed_fetch(&fetch_request, fetch_header.ctr, fetch_header.client_nonce)
            .unwrap();

        let embeddings = receiver
            .handle_embed_fetch_response(&fetch_response, fetch_header.ctr)
            .unwrap();

        // Verify we got the right data
        // Row 5: values 40-47
        // Row 10: values 80-87
        // Row 15: values 120-127
        assert_eq!(embeddings.len(), 3 * 8 * 4); // 3 rows × 8 dims × 4 bytes

        // Check first element of each row
        let row0_first = i32::from_le_bytes([embeddings[0], embeddings[1], embeddings[2], embeddings[3]]);
        let row1_first = i32::from_le_bytes([embeddings[32], embeddings[33], embeddings[34], embeddings[35]]);
        let row2_first = i32::from_le_bytes([embeddings[64], embeddings[65], embeddings[66], embeddings[67]]);

        assert_eq!(row0_first, 5 * 8); // Token 5, first elem
        assert_eq!(row1_first, 10 * 8); // Token 10, first elem
        assert_eq!(row2_first, 15 * 8); // Token 15, first elem
    }
}
