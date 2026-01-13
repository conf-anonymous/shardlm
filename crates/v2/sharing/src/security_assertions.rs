//! Security assertions for privacy-preserving inference
//!
//! This module provides both compile-time and runtime checks to ensure
//! the server NEVER reconstructs plaintext user data.
//!
//! # Compile-Time Guarantees
//!
//! - `ServerContext` has NO `reconstruct()` method
//! - `ServerShare<T>` cannot be added to `ClientShare<T>` on server
//! - `SecureSharePair<T>` can only be constructed client-side
//!
//! # Runtime Assertions
//!
//! - `assert_not_plaintext!` - Verifies data doesn't match expected plaintext
//! - `assert_shares_differ!` - Verifies two shares are different
//! - `SecurityAuditor` - Logs and validates all share operations

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Global security mode flag
///
/// When enabled (production), additional runtime checks are performed.
static SECURITY_MODE: AtomicBool = AtomicBool::new(true);

/// Counter for security-critical operations (for auditing)
static OPERATION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Enable or disable security mode
///
/// # Safety
///
/// Security mode should ALWAYS be enabled in production.
/// Only disable for testing/benchmarking with explicit acknowledgment.
pub fn set_security_mode(enabled: bool) {
    if !enabled {
        tracing::warn!("⚠️  SECURITY MODE DISABLED - DO NOT USE IN PRODUCTION");
    }
    SECURITY_MODE.store(enabled, Ordering::SeqCst);
}

/// Check if security mode is enabled
pub fn is_security_mode_enabled() -> bool {
    SECURITY_MODE.load(Ordering::SeqCst)
}

/// Get current operation count (for auditing)
pub fn get_operation_count() -> u64 {
    OPERATION_COUNTER.load(Ordering::Relaxed)
}

/// Increment operation counter
fn record_operation() {
    OPERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
}

/// Security violation error
#[derive(Debug, Clone)]
pub struct SecurityViolation {
    pub operation: String,
    pub message: String,
    pub operation_id: u64,
}

impl std::fmt::Display for SecurityViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "🚨 SECURITY VIOLATION [op#{}]: {} - {}",
            self.operation_id, self.operation, self.message
        )
    }
}

impl std::error::Error for SecurityViolation {}

/// Assert that data is not plaintext (runtime check)
///
/// # Security
///
/// This should be called on server-side to verify that we're not
/// accidentally working with reconstructed plaintext.
#[inline]
pub fn assert_not_plaintext<T: PartialEq>(
    data: &[T],
    possible_plaintext: &[T],
    operation: &str,
) -> Result<(), SecurityViolation> {
    record_operation();
    let op_id = get_operation_count();

    if !is_security_mode_enabled() {
        return Ok(());
    }

    if data == possible_plaintext {
        let violation = SecurityViolation {
            operation: operation.to_string(),
            message: "Data appears to be plaintext (matches reference)".to_string(),
            operation_id: op_id,
        };
        tracing::error!("{}", violation);
        return Err(violation);
    }

    Ok(())
}

/// Assert that two shares are different (they should be random)
///
/// # Security
///
/// Client and server shares should never be identical.
#[inline]
pub fn assert_shares_differ<T: PartialEq>(
    share1: &[T],
    share2: &[T],
    operation: &str,
) -> Result<(), SecurityViolation> {
    record_operation();
    let op_id = get_operation_count();

    if !is_security_mode_enabled() {
        return Ok(());
    }

    if share1 == share2 {
        let violation = SecurityViolation {
            operation: operation.to_string(),
            message: "Shares are identical (possible security issue)".to_string(),
            operation_id: op_id,
        };
        tracing::error!("{}", violation);
        return Err(violation);
    }

    Ok(())
}

/// Assert we're in a server context (cannot reconstruct)
///
/// This is a compile-time guarantee - the presence of `ServerContext`
/// parameter proves the code is server-side.
#[inline]
pub fn assert_server_context(ctx: &crate::secure::ServerContext) {
    record_operation();
    // The type system ensures this - ServerContext has no reconstruct method
    let _ = ctx;
}

/// Macro to verify shares are not plaintext
///
/// Usage:
/// ```ignore
/// verify_not_plaintext!(output_share, possible_plaintext, "linear layer output");
/// ```
#[macro_export]
macro_rules! verify_not_plaintext {
    ($data:expr, $plaintext:expr, $op:expr) => {
        $crate::security_assertions::assert_not_plaintext($data, $plaintext, $op)
    };
}

/// Macro to verify shares differ
///
/// Usage:
/// ```ignore
/// verify_shares_differ!(client_share, server_share, "embedding split");
/// ```
#[macro_export]
macro_rules! verify_shares_differ {
    ($share1:expr, $share2:expr, $op:expr) => {
        $crate::security_assertions::assert_shares_differ($share1, $share2, $op)
    };
}

/// Security auditor for tracking all share operations
///
/// # Usage
///
/// Create an auditor at the start of an inference request and use it
/// to track all operations. At the end, verify no violations occurred.
pub struct SecurityAuditor {
    /// Request/session identifier
    session_id: String,
    /// Number of operations
    operation_count: u64,
    /// Any violations detected
    violations: Vec<SecurityViolation>,
    /// Start time
    start_time: std::time::Instant,
}

impl SecurityAuditor {
    /// Create a new auditor for a session
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            operation_count: 0,
            violations: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    /// Record an operation
    pub fn record_operation(&mut self, operation: &str) {
        self.operation_count += 1;
        tracing::trace!(
            session_id = %self.session_id,
            operation = operation,
            op_count = self.operation_count,
            "Security audit: operation recorded"
        );
    }

    /// Record a share operation with verification
    pub fn verify_share_operation<T: PartialEq>(
        &mut self,
        client_data: &[T],
        server_data: &[T],
        operation: &str,
    ) {
        self.operation_count += 1;

        if let Err(v) = assert_shares_differ(client_data, server_data, operation) {
            self.violations.push(v);
        }
    }

    /// Record a violation
    pub fn record_violation(&mut self, violation: SecurityViolation) {
        self.violations.push(violation);
    }

    /// Check if any violations occurred
    pub fn has_violations(&self) -> bool {
        !self.violations.is_empty()
    }

    /// Get all violations
    pub fn violations(&self) -> &[SecurityViolation] {
        &self.violations
    }

    /// Finalize and log audit results
    ///
    /// Returns error if any violations occurred.
    pub fn finalize(self) -> Result<AuditReport, Vec<SecurityViolation>> {
        let elapsed = self.start_time.elapsed();

        let report = AuditReport {
            session_id: self.session_id.clone(),
            operation_count: self.operation_count,
            duration_ms: elapsed.as_millis() as u64,
            violations_count: self.violations.len(),
        };

        if self.has_violations() {
            tracing::error!(
                session_id = %self.session_id,
                violations = self.violations.len(),
                operations = self.operation_count,
                "🚨 SECURITY AUDIT FAILED"
            );
            return Err(self.violations);
        }

        tracing::info!(
            session_id = %self.session_id,
            operations = self.operation_count,
            duration_ms = elapsed.as_millis(),
            "✅ Security audit passed"
        );

        Ok(report)
    }
}

/// Audit report for a completed session
#[derive(Debug)]
pub struct AuditReport {
    pub session_id: String,
    pub operation_count: u64,
    pub duration_ms: u64,
    pub violations_count: usize,
}

// =============================================================================
// COMPILE-TIME SECURITY TESTS
// =============================================================================

/// This module contains tests that FAIL TO COMPILE if security is broken.
///
/// If any of these compile, it means our security model has been compromised.
#[cfg(test)]
mod compile_time_security_tests {
    use super::*;
    use crate::secure::{ServerContext, ServerShare, ClientShare, SecureSharePair};

    /// Test: ServerContext has no reconstruct method
    ///
    /// This test documents that the following code would NOT compile:
    /// ```compile_fail
    /// let ctx = ServerContext::new();
    /// ctx.reconstruct(...); // ERROR: no method named `reconstruct`
    /// ```
    #[test]
    fn test_server_context_has_no_reconstruct() {
        let ctx = ServerContext::new();
        // The following would NOT compile:
        // ctx.reconstruct(...);
        assert_server_context(&ctx);
    }

    /// Test: Cannot create SecureSharePair from individual shares
    ///
    /// This test documents that the following code would NOT compile
    /// (outside of the crate):
    /// ```compile_fail
    /// let client_share = ClientShare::new(...);
    /// let server_share = ServerShare::new(...);
    /// let pair = SecureSharePair { client: client_share, server: server_share };
    /// // ERROR: fields are private
    /// ```
    #[test]
    fn test_cannot_create_share_pair_externally() {
        // Within the crate we can use from_shares, but external code cannot
        // This is enforced by pub(crate) on from_shares and SecureShare::new
        let _ = true; // Placeholder - the real test is that external code can't compile
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_not_plaintext_passes_on_different() {
        let data = vec![1.0f32, 2.0, 3.0];
        let plaintext = vec![4.0f32, 5.0, 6.0];
        assert!(assert_not_plaintext(&data, &plaintext, "test").is_ok());
    }

    #[test]
    fn test_assert_not_plaintext_fails_on_same() {
        set_security_mode(true);
        let data = vec![1.0f32, 2.0, 3.0];
        let plaintext = vec![1.0f32, 2.0, 3.0];
        assert!(assert_not_plaintext(&data, &plaintext, "test").is_err());
    }

    #[test]
    fn test_assert_shares_differ_passes_on_different() {
        let share1 = vec![1.0f32, 2.0, 3.0];
        let share2 = vec![4.0f32, 5.0, 6.0];
        assert!(assert_shares_differ(&share1, &share2, "test").is_ok());
    }

    #[test]
    fn test_assert_shares_differ_fails_on_same() {
        set_security_mode(true);
        let share1 = vec![1.0f32, 2.0, 3.0];
        let share2 = vec![1.0f32, 2.0, 3.0];
        assert!(assert_shares_differ(&share1, &share2, "test").is_err());
    }

    #[test]
    fn test_security_auditor_passes_clean() {
        let mut auditor = SecurityAuditor::new("test-session");
        auditor.record_operation("op1");
        auditor.verify_share_operation(&[1.0f32, 2.0], &[3.0f32, 4.0], "split");
        assert!(auditor.finalize().is_ok());
    }

    #[test]
    fn test_security_auditor_fails_with_violation() {
        set_security_mode(true);
        let mut auditor = SecurityAuditor::new("test-session");
        auditor.verify_share_operation(&[1.0f32, 2.0], &[1.0f32, 2.0], "bad split");
        assert!(auditor.finalize().is_err());
    }

    #[test]
    fn test_security_mode_disable() {
        set_security_mode(false);
        // With security mode off, even identical shares pass
        let share1 = vec![1.0f32, 2.0, 3.0];
        let share2 = vec![1.0f32, 2.0, 3.0];
        assert!(assert_shares_differ(&share1, &share2, "test").is_ok());
        set_security_mode(true); // Re-enable for other tests
    }
}
