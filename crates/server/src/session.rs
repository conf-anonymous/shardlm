//! Session management
//!
//! Each session represents a client connection with:
//! - OT state (base OT keys, IKNP extension state)
//! - Counter for replay protection
//! - Expiration tracking

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use shardlm_model::KVCache;
use shardlm_ot::{IknpOtExtension, OtSender, OtSessionConfig};
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use crate::error::{Result, ServerError};

/// Session status (for status/refresh endpoints)
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionStatus {
    /// Session ID
    pub id: Uuid,
    /// Whether OT handshake is complete
    pub ready: bool,
    /// Remaining TTL in seconds
    pub ttl_secs: i64,
    /// Current request count
    pub request_count: u32,
    /// Maximum allowed requests
    pub max_requests: u32,
    /// Expected next counter value
    pub expected_counter: u64,
}

/// Session state
pub struct Session {
    /// Unique session ID
    pub id: Uuid,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,

    /// Whether base OT handshake is complete
    pub ready: bool,

    /// Expected next counter value
    pub expected_counter: u64,

    /// Total request count
    pub request_count: u32,

    /// Maximum allowed requests
    pub max_requests: u32,

    /// OT sender state (wrapped for interior mutability)
    ot_sender: RwLock<Option<OtSender<IknpOtExtension>>>,

    /// KV cache for transformer inference (persists across requests)
    kv_cache: RwLock<Option<KVCache>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("id", &self.id)
            .field("created_at", &self.created_at)
            .field("expires_at", &self.expires_at)
            .field("ready", &self.ready)
            .field("expected_counter", &self.expected_counter)
            .field("request_count", &self.request_count)
            .field("max_requests", &self.max_requests)
            .field("ot_sender", &"<OtSender>")
            .finish()
    }
}

impl Session {
    /// Create a new session
    pub fn new(ttl: Duration, max_requests: u32, ot_config: OtSessionConfig) -> Self {
        let now = Utc::now();
        let expires_at = now + chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::seconds(900));

        // Create OT sender with IKNP extension
        let sender_ext = IknpOtExtension::new_server();
        let sender = OtSender::new(sender_ext, ot_config);

        Self {
            id: Uuid::new_v4(),
            created_at: now,
            expires_at,
            ready: false,
            expected_counter: 0,
            request_count: 0,
            max_requests,
            ot_sender: RwLock::new(Some(sender)),
            kv_cache: RwLock::new(None), // Initialized lazily when first inference runs
        }
    }

    /// Check if session has expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Check if max requests exceeded
    pub fn is_exhausted(&self) -> bool {
        self.request_count >= self.max_requests
    }

    /// Get remaining TTL in seconds
    pub fn ttl_secs(&self) -> i64 {
        let remaining = self.expires_at - Utc::now();
        remaining.num_seconds().max(0)
    }

    /// Extend the session TTL by the given duration
    pub fn touch(&mut self, ttl: Duration) {
        let now = Utc::now();
        self.expires_at = now + chrono::Duration::from_std(ttl).unwrap_or(chrono::Duration::seconds(900));
    }

    /// Validate and increment counter
    pub fn validate_counter(&mut self, counter: u64) -> Result<()> {
        if counter != self.expected_counter {
            return Err(ServerError::InvalidCounter {
                expected: self.expected_counter,
                got: counter,
            });
        }
        self.expected_counter += 1;
        self.request_count += 1;
        Ok(())
    }

    /// Get mutable access to OT sender
    pub fn with_ot_sender<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut OtSender<IknpOtExtension>) -> Result<R>,
    {
        let mut guard = self.ot_sender.write();
        let sender = guard.as_mut().ok_or_else(|| {
            ServerError::Internal("OT sender not initialized".to_string())
        })?;
        f(sender)
    }

    /// Mark session as ready (base OT complete)
    pub fn mark_ready(&mut self) {
        self.ready = true;
        // First request uses counter 1 (matching the OT protocol)
        self.expected_counter = 1;
    }

    /// Get or create KV cache for this session
    pub fn with_kv_cache<F, R>(
        &self,
        num_layers: usize,
        max_seq_len: usize,
        num_kv_heads: usize,
        head_dim: usize,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut KVCache) -> R,
    {
        let mut guard = self.kv_cache.write();
        if guard.is_none() {
            *guard = Some(KVCache::new(num_layers, max_seq_len, num_kv_heads, head_dim));
        }
        f(guard.as_mut().unwrap())
    }
}

/// Session store
pub struct SessionStore {
    /// Active sessions
    sessions: DashMap<Uuid, Session>,

    /// Maximum sessions allowed
    max_sessions: usize,

    /// Default TTL for new sessions
    default_ttl: Duration,

    /// Default max requests per session
    default_max_requests: u32,

    /// OT config for new sessions (wrapped in RwLock for updates after model load)
    ot_config: RwLock<OtSessionConfig>,
}

impl SessionStore {
    /// Create a new session store
    pub fn new(
        max_sessions: usize,
        default_ttl: Duration,
        default_max_requests: u32,
        ot_config: OtSessionConfig,
    ) -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions,
            default_ttl,
            default_max_requests,
            ot_config: RwLock::new(ot_config),
        }
    }

    /// Update OT config (called after model load with actual vocab_size)
    pub fn update_ot_config(&self, vocab_size: u32, hidden_dim: u16) {
        let mut config = self.ot_config.write();
        config.vocab_size = vocab_size;
        config.hidden_dim = hidden_dim;
    }

    /// Create a new session
    pub fn create_session(&self) -> Result<Uuid> {
        // Check capacity
        if self.sessions.len() >= self.max_sessions {
            // Try to clean up expired sessions first
            self.cleanup_expired();

            if self.sessions.len() >= self.max_sessions {
                return Err(ServerError::Internal("Max sessions reached".to_string()));
            }
        }

        let session = Session::new(
            self.default_ttl,
            self.default_max_requests,
            self.ot_config.read().clone(),
        );
        let id = session.id;
        self.sessions.insert(id, session);
        Ok(id)
    }

    /// Get a session by ID
    pub fn get(&self, id: &Uuid) -> Result<dashmap::mapref::one::Ref<'_, Uuid, Session>> {
        self.sessions.get(id).ok_or_else(|| {
            ServerError::SessionNotFound(id.to_string())
        })
    }

    /// Get a mutable session by ID
    pub fn get_mut(&self, id: &Uuid) -> Result<dashmap::mapref::one::RefMut<'_, Uuid, Session>> {
        let mut session = self.sessions.get_mut(id).ok_or_else(|| {
            ServerError::SessionNotFound(id.to_string())
        })?;

        // Check expiration
        if session.is_expired() {
            drop(session);
            self.sessions.remove(id);
            return Err(ServerError::SessionExpired(id.to_string()));
        }

        // Check exhaustion
        if session.is_exhausted() {
            drop(session);
            self.sessions.remove(id);
            return Err(ServerError::MaxRequestsExceeded);
        }

        // Auto-extend TTL on activity (touch)
        session.touch(self.default_ttl);

        Ok(session)
    }

    /// Refresh a session's TTL without requiring mutable access
    /// Returns the new TTL in seconds, or an error if session not found/expired
    pub fn refresh(&self, id: &Uuid) -> Result<i64> {
        let mut session = self.sessions.get_mut(id).ok_or_else(|| {
            ServerError::SessionNotFound(id.to_string())
        })?;

        // Check expiration
        if session.is_expired() {
            drop(session);
            self.sessions.remove(id);
            return Err(ServerError::SessionExpired(id.to_string()));
        }

        // Extend TTL
        session.touch(self.default_ttl);
        Ok(session.ttl_secs())
    }

    /// Get session status without modifying it
    pub fn status(&self, id: &Uuid) -> Result<SessionStatus> {
        let session = self.sessions.get(id).ok_or_else(|| {
            ServerError::SessionNotFound(id.to_string())
        })?;

        // Check expiration (but don't remove - status is read-only)
        if session.is_expired() {
            return Err(ServerError::SessionExpired(id.to_string()));
        }

        Ok(SessionStatus {
            id: session.id,
            ready: session.ready,
            ttl_secs: session.ttl_secs(),
            request_count: session.request_count,
            max_requests: session.max_requests,
            expected_counter: session.expected_counter,
        })
    }

    /// Get default TTL
    pub fn default_ttl(&self) -> Duration {
        self.default_ttl
    }

    /// Remove a session
    pub fn remove(&self, id: &Uuid) -> Option<Session> {
        self.sessions.remove(id).map(|(_, s)| s)
    }

    /// Clean up expired sessions
    pub fn cleanup_expired(&self) {
        self.sessions.retain(|_, session| {
            !session.is_expired() && !session.is_exhausted()
        });
        tracing::debug!(
            "Cleaned up sessions, {} remaining",
            self.sessions.len()
        );
    }

    /// Get current session count
    pub fn len(&self) -> usize {
        self.sessions.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.sessions.is_empty()
    }
}

/// Shared session store
pub type SharedSessionStore = Arc<SessionStore>;
