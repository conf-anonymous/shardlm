//! API routes

pub mod health;
pub mod session;
pub mod ot;
pub mod inference;

use axum::{
    Router,
    routing::{get, post},
};

use crate::state::AppState;

/// Create the main router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Health and readiness
        .route("/health", get(health::health))
        .route("/ready", get(health::ready))
        .route("/v1/info", get(health::info))
        // Session management
        .route("/v1/session/new", post(session::create_session))
        .route("/v1/session/status", post(session::session_status))
        .route("/v1/session/refresh", post(session::session_refresh))
        // OT protocol
        .route("/v1/ot/base/init", post(ot::base_ot_init))
        .route("/v1/ot/ready", post(ot::ot_ready))
        .route("/v1/ot/embed/fetch", post(ot::embed_fetch))
        // Inference
        .route("/v1/inference/forward", post(inference::forward))
        // Add state
        .with_state(state)
}
