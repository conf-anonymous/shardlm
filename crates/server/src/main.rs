//! ShardLM Server Binary
//!
//! Run with: cargo run -p shardlm-server --release

use std::net::SocketAddr;

use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    decompression::RequestDecompressionLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use shardlm_server::{AppState, ServerConfig, routes};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "shardlm_server=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = ServerConfig::from_env();
    tracing::info!("Starting ShardLM Server v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Model directory: {:?}", config.model_dir);
    tracing::info!("Bind address: {}", config.bind_address());

    // Create application state
    let mut state = AppState::new(config.clone());

    // Load model (embeddings)
    if let Err(e) = state.load_model().await {
        tracing::warn!("Failed to load model: {}. Server will start but not be ready.", e);
    }

    // Build CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any) // TODO: Restrict in production
        .allow_methods(Any)
        .allow_headers(Any);

    // Create router with compression for responses and decompression for requests
    // This significantly reduces network bandwidth for the large embedding/inference payloads
    let app = routes::create_router(state)
        .layer(CompressionLayer::new().zstd(true))
        .layer(RequestDecompressionLayer::new().zstd(true))
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
