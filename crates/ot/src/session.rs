//! OT Session management

use rand::RngCore;
use std::time::Duration;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use crate::error::{OtError, Result};

/// Session identifier (16 bytes)
pub type SessionId = [u8; 16];

/// Generate a new random session ID
pub fn generate_session_id() -> SessionId {
    let mut id = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut id);
    id
}

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtSessionState {
    /// Waiting for initialization
    Uninitialized,
    /// Base OT in progress
    BaseOtInProgress,
    /// Session ready for OT queries
    Ready,
    /// Session expired
    Expired,
}

/// OT Session configuration
#[derive(Debug, Clone)]
pub struct OtSessionConfig {
    /// Maximum prompt length (L_MAX = 64)
    pub max_prompt_len: u16,
    /// Hidden dimension (d from model)
    pub hidden_dim: u16,
    /// Vocabulary size (V)
    pub vocab_size: u32,
    /// Fixed-point scale
    pub scale: u8,
    /// Session time-to-live
    pub ttl: Duration,
    /// Maximum requests
    pub max_requests: u32,
}

impl Default for OtSessionConfig {
    fn default() -> Self {
        Self {
            max_prompt_len: 1024,
            hidden_dim: 2048,      // TinyLlama default
            vocab_size: 32000,     // TinyLlama default
            scale: 16,
            ttl: Duration::from_secs(3600),
            max_requests: 10000,
        }
    }
}

/// An OT session tracking state between client and server
#[derive(Debug)]
pub struct OtSession {
    /// Session ID
    pub id: SessionId,
    /// Current state
    pub state: OtSessionState,
    /// Session configuration
    pub config: OtSessionConfig,
    /// Request counter (starts at 1 after ready)
    pub counter: u64,
    /// Number of requests made
    pub request_count: u32,
    /// Session creation time (not available in WASM)
    #[cfg(not(target_arch = "wasm32"))]
    pub created_at: Instant,
    /// Session creation time in milliseconds (for WASM)
    #[cfg(target_arch = "wasm32")]
    pub created_at_ms: f64,
    /// Base OT phase counter
    pub base_ot_phase: u16,
}

impl OtSession {
    /// Create a new uninitialized session
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(config: OtSessionConfig) -> Self {
        Self {
            id: [0u8; 16],
            state: OtSessionState::Uninitialized,
            config,
            counter: 0,
            request_count: 0,
            created_at: Instant::now(),
            base_ot_phase: 0,
        }
    }

    /// Create a new uninitialized session (WASM version)
    #[cfg(target_arch = "wasm32")]
    pub fn new(config: OtSessionConfig) -> Self {
        Self {
            id: [0u8; 16],
            state: OtSessionState::Uninitialized,
            config,
            counter: 0,
            request_count: 0,
            created_at_ms: 0.0, // Will be set when session is created
            base_ot_phase: 0,
        }
    }

    /// Initialize the session with a new ID
    pub fn initialize(&mut self) -> Result<SessionId> {
        if self.state != OtSessionState::Uninitialized {
            return Err(OtError::SessionAlreadyInitialized);
        }
        self.id = generate_session_id();
        self.state = OtSessionState::BaseOtInProgress;
        self.base_ot_phase = 1;
        Ok(self.id)
    }

    /// Initialize with a specific ID (for client accepting server's ID)
    pub fn initialize_with_id(&mut self, id: SessionId) -> Result<()> {
        if self.state != OtSessionState::Uninitialized {
            return Err(OtError::SessionAlreadyInitialized);
        }
        self.id = id;
        self.state = OtSessionState::BaseOtInProgress;
        self.base_ot_phase = 1;
        Ok(())
    }

    /// Advance base OT phase
    pub fn advance_base_ot_phase(&mut self) -> u16 {
        self.base_ot_phase += 1;
        self.base_ot_phase
    }

    /// Mark session as ready
    pub fn mark_ready(&mut self) -> Result<()> {
        if self.state != OtSessionState::BaseOtInProgress {
            return Err(OtError::SessionNotInitialized);
        }
        self.state = OtSessionState::Ready;
        self.counter = 1; // First request uses counter 1
        Ok(())
    }

    /// Check if session is ready
    pub fn is_ready(&self) -> bool {
        self.state == OtSessionState::Ready
    }

    /// Check if session has expired
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_expired(&self) -> bool {
        self.state == OtSessionState::Expired
            || self.created_at.elapsed() > self.config.ttl
            || self.request_count >= self.config.max_requests
    }

    /// Check if session has expired (WASM version - only checks request count, not time)
    #[cfg(target_arch = "wasm32")]
    pub fn is_expired(&self) -> bool {
        self.state == OtSessionState::Expired
            || self.request_count >= self.config.max_requests
    }

    /// Validate and consume a request with the given counter
    pub fn validate_request(&mut self, ctr: u64) -> Result<()> {
        if !self.is_ready() {
            return Err(OtError::SessionNotInitialized);
        }
        if self.is_expired() {
            self.state = OtSessionState::Expired;
            return Err(OtError::SessionExpired);
        }
        if ctr != self.counter {
            return Err(OtError::CounterMismatch {
                expected: self.counter,
                got: ctr,
            });
        }
        self.counter += 1;
        self.request_count += 1;
        Ok(())
    }

    /// Get row size in bytes
    pub fn row_bytes(&self) -> u32 {
        self.config.hidden_dim as u32 * 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle() {
        let mut session = OtSession::new(OtSessionConfig::default());
        assert_eq!(session.state, OtSessionState::Uninitialized);

        let id = session.initialize().unwrap();
        assert_eq!(session.state, OtSessionState::BaseOtInProgress);
        assert_eq!(session.id, id);

        session.mark_ready().unwrap();
        assert_eq!(session.state, OtSessionState::Ready);
        assert_eq!(session.counter, 1);

        // First request
        session.validate_request(1).unwrap();
        assert_eq!(session.counter, 2);

        // Second request
        session.validate_request(2).unwrap();
        assert_eq!(session.counter, 3);
    }

    #[test]
    fn test_counter_mismatch() {
        let mut session = OtSession::new(OtSessionConfig::default());
        session.initialize().unwrap();
        session.mark_ready().unwrap();

        // Wrong counter
        let result = session.validate_request(5);
        assert!(matches!(
            result,
            Err(OtError::CounterMismatch {
                expected: 1,
                got: 5
            })
        ));
    }

    #[test]
    fn test_double_initialize() {
        let mut session = OtSession::new(OtSessionConfig::default());
        session.initialize().unwrap();

        let result = session.initialize();
        assert!(matches!(result, Err(OtError::SessionAlreadyInitialized)));
    }
}
