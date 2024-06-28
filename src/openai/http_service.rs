use axum::extract::State;
use axum::Json;
use axum::response::IntoResponse;
use candle_core::Device;
use candle_transformers::models::phi3::Model as Phi3;
use tokenizers::Tokenizer;
use tracing::{debug, info};
use crate::openai::http_entities::{AppState, CompletionsRequest, TextGeneration};

impl From<(Phi3, Device, Tokenizer, Option<f64>)> for AppState {
    fn from(e: (Phi3, Device, Tokenizer, Option<f64>)) -> Self {
        Self {
            model: e.0,
            device: e.1,
            tokenizer: e.2,
            temperature: e.3,
        }
    }
}

impl From<AppState> for TextGeneration {
    fn from(e: AppState) -> Self {
        Self::new(
            e.model,
            e.tokenizer,
            299792458, // seed RNG
            e.temperature,  // temperature
            None,      // top_p - Nucleus sampling probability stuff
            1.1,       // repeat penalty
            64,        // context size to consider for the repeat penalty
            &e.device,
        )
    }
}

pub async fn health(State(state): State<AppState>) -> &'static str {
    info!("Health endpoint called");

    debug!("Model state is {}", state.device.is_metal());

    "Service is up!"
}

pub async fn run_completions(
    State(state): State<AppState>,
    Json(request): Json<CompletionsRequest>) -> impl IntoResponse {

    let ai_gen = TextGeneration::from(state);
    ai_gen.run(String::from(request.prompt.clone()), 50)
}