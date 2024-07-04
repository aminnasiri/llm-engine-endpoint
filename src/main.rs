use anyhow::{Error, Result};
use tracing::{info};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use llm_engine_endpoint::core::load_model::initialise_model;
use llm_engine_endpoint::openai::http_entities::{CompletionResponse, CompletionsRequest};
use llm_engine_endpoint::openai::http_service::{ run_completions};

#[tokio::main]
async fn main() -> Result<CompletionResponse, Error> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // axum logs rejections from built-in extractors with the `axum::rejection`
                // target, at `TRACE` level. `axum::rejection=trace` enables showing those events
                "llm_engine_endpoint=debug,tower_http=debug".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let Ok(api_token) = std::env::var("HF_TOKEN") else {
        return Err(anyhow::anyhow!("Error getting HF_TOKEN env var"));
    };

    info!("Model is loading in memory");

    let state = initialise_model(api_token)?;

    info!("Model loaded and is ready now");

    let prompt = "Create a mind map with mermaidJS to How might we redesign our guest's digital nighttime experience so they have more restful sleep?";
    let max_token: i32 = 100;
    let temp: f64 = 0.8;
    let request = CompletionsRequest::new("Phi-3-128".parse()?, prompt.parse()?, max_token, temp);
    let response = run_completions(state, request).await;

    // info!(response);

    Ok(response)
}
