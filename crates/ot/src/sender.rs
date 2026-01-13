//! OT Sender (Server side)

use shardlm_protocol::{
    EmbedFetchRequestPayload, EmbedFetchResponsePayload, Header, OtBaseOtMsgPayload,
    OtSessionInitPayload, OtSessionParamsPayload, OtSessionReadyPayload,
};

use crate::error::{OtError, Result};
use crate::extension::OtExtension;
use crate::session::{OtSession, OtSessionConfig};

/// OT Sender (Server) manages sessions and responds to OT queries
pub struct OtSender<E: OtExtension> {
    /// OT extension implementation
    extension: E,
    /// Current session
    session: OtSession,
    /// Embedding database (V × row_bytes)
    embedding_db: Vec<u8>,
}

impl<E: OtExtension> OtSender<E> {
    /// Create a new OT sender with the given extension and config
    pub fn new(extension: E, config: OtSessionConfig) -> Self {
        Self {
            extension,
            session: OtSession::new(config),
            embedding_db: vec![],
        }
    }

    /// Set the embedding database
    pub fn set_embedding_db(&mut self, db: Vec<u8>) {
        self.embedding_db = db;
    }

    /// Get session ID
    pub fn session_id(&self) -> &[u8; 16] {
        &self.session.id
    }

    /// Get session config
    pub fn config(&self) -> &OtSessionConfig {
        &self.session.config
    }

    /// Handle OT_SESSION_INIT and return OT_SESSION_PARAMS
    pub fn handle_session_init(
        &mut self,
        _init: &OtSessionInitPayload,
        client_nonce: [u8; 8],
    ) -> Result<(Header, OtSessionParamsPayload)> {
        let session_id = self.session.initialize()?;

        let mut header = Header::new(session_id, 0);
        header.client_nonce = client_nonce;
        header.generate_server_nonce();

        let params = OtSessionParamsPayload {
            accepted_lmax: self.session.config.max_prompt_len,
            accepted_d: self.session.config.hidden_dim,
            vocab_size: self.session.config.vocab_size,
            value_type: 1,
            fixed_point_scale: self.session.config.scale,
            row_layout: 1,
            selected_ot_suite: 0x0101, // IKNP
            suite_flags: 0,
            session_ttl_sec: self.session.config.ttl.as_secs() as u32,
            max_requests: self.session.config.max_requests,
        };

        Ok((header, params))
    }

    /// Handle OT_BASE_OT_MSG and return response (or None if done)
    pub fn handle_base_ot_msg(
        &mut self,
        msg: &OtBaseOtMsgPayload,
        client_nonce: [u8; 8],
    ) -> Result<Option<(Header, OtBaseOtMsgPayload)>> {
        // Server processes client's base OT message (A points) and responds (B points + commitment)
        let response_blob = Some(self.extension.process_base_ot_receiver(&msg.blob)?);

        self.session.advance_base_ot_phase();

        if let Some(blob) = response_blob {
            let is_final = self.extension.is_base_ot_complete();

            let mut header = Header::new(self.session.id, 0);
            header.client_nonce = client_nonce;
            header.generate_server_nonce();

            let response = OtBaseOtMsgPayload {
                phase: self.session.base_ot_phase,
                is_final,
                blob,
            };

            Ok(Some((header, response)))
        } else {
            Ok(None)
        }
    }

    /// Generate OT_SESSION_READY message
    pub fn generate_session_ready(
        &mut self,
        client_nonce: [u8; 8],
    ) -> Result<(Header, OtSessionReadyPayload)> {
        self.session.mark_ready()?;

        let mut header = Header::new(self.session.id, 0);
        header.client_nonce = client_nonce;
        header.generate_server_nonce();

        let ready = OtSessionReadyPayload {
            ok: true,
            session_ttl_sec: self.session.config.ttl.as_secs() as u32,
            max_requests: self.session.config.max_requests,
            starting_ctr: 1,
        };

        Ok((header, ready))
    }

    /// Handle EMBED_FETCH_REQUEST and return response
    pub fn handle_embed_fetch(
        &mut self,
        request: &EmbedFetchRequestPayload,
        ctr: u64,
        client_nonce: [u8; 8],
    ) -> Result<(Header, EmbedFetchResponsePayload)> {
        // Validate request
        self.session.validate_request(ctr)?;

        if request.len as usize > self.session.config.max_prompt_len as usize {
            return Err(OtError::BatchSizeExceeded {
                size: request.len as usize,
                max: self.session.config.max_prompt_len as usize,
            });
        }

        let row_bytes = self.session.row_bytes() as usize;

        // Process the OT query
        let response_blob = self.extension.process_query(
            &request.query_blob,
            &self.embedding_db,
            row_bytes,
            &self.session.id,
            ctr,
        )?;

        let mut header = Header::new(self.session.id, ctr);
        header.client_nonce = client_nonce;
        header.generate_server_nonce();

        let response = EmbedFetchResponsePayload::new(request.len, row_bytes as u32, response_blob);

        Ok((header, response))
    }

    /// Check if session is ready
    pub fn is_ready(&self) -> bool {
        self.session.is_ready()
    }

    /// Check if base OT is complete
    pub fn is_base_ot_complete(&self) -> bool {
        self.extension.is_base_ot_complete()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extension::SimpleOtExtension;

    #[test]
    fn test_sender_session_init() {
        let ext = SimpleOtExtension::new();
        let mut sender = OtSender::new(ext, OtSessionConfig::default());

        let init = OtSessionInitPayload::default();
        let (header, params) = sender.handle_session_init(&init, [0u8; 8]).unwrap();

        assert_ne!(header.session_id, [0u8; 16]);
        assert_eq!(params.accepted_lmax, 1024);
        assert_eq!(params.vocab_size, 32000);
    }
}
