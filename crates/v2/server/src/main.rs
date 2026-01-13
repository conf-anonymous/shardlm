//! ShardLM v2 Server Binary
//!
//! GPU-accelerated Llama 70B inference server.
//!
//! Run with:
//!   SHARDLM_V2_MODEL_DIR=/data/llama-70b-instruct-weights cargo run -p shardlm-v2-server --features cuda --release

use std::net::SocketAddr;

use axum::extract::DefaultBodyLimit;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use shardlm_v2_server::{routes, AppState, ServerConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "shardlm_v2_server=info,shardlm_v2_model=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = ServerConfig::from_env();
    tracing::info!("Starting ShardLM v2 Server v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Model architecture: {:?}", config.model_architecture);
    tracing::info!("Model directory: {:?}", config.model_dir);
    tracing::info!("Bind address: {}", config.bind_address());
    tracing::info!("Number of GPUs: {}", config.num_gpus);

    // Create application state
    let mut state = AppState::new(config.clone());

    // Load model
    tracing::info!("Loading model (this may take a few minutes on first run)...");
    if let Err(e) = state.load_model().await {
        tracing::error!("Failed to load model: {}. Server will start but not be ready.", e);
    } else {
        tracing::info!("Model loaded successfully!");
    }

    // Build CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Create router (without compression for debugging)
    // Increase body limit to 50MB to handle batched forward requests
    // (28 layers × 3584 floats × 2 shares × 4 bytes ≈ 800KB, but JSON adds overhead)
    let app = routes::create_router(state)
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // 50MB
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    // Parse bind address
    let addr: SocketAddr = config.bind_address().parse()?;

    tracing::info!("Server listening on {}", addr);

    // Start server
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
