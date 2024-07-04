use std::process::{ExitCode, Termination};
use std::time::Instant;
use candle_core::{Device, DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::{error, info};
use uuid::Uuid;

use crate::core::output_stream::TokenOutputStream;
use candle_transformers::models::phi3::Model as Phi3;

#[derive(Serialize, Deserialize)]
pub struct CompletionsRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: i32,
    pub temperature: f64,
}

impl CompletionsRequest {
    pub fn new(model: String, prompt: String, max_tokens: i32, temperature: f64) -> Self {
        Self { model, prompt, max_tokens, temperature }
    }
}


#[derive(Serialize, Deserialize)]
pub struct CompletionResponse {
    id: String,
    object: String,
    created: i64,
    model: String,
}

impl CompletionResponse {
    pub fn new(id: String, object: String, created: i64, model: String) -> Self {
        Self { id, object, created, model}
    }
}

impl Termination for CompletionResponse {
    fn report(self) -> ExitCode {
        ExitCode::SUCCESS
    }
}


// #[derive(Deserialize)]
// pub struct Prompt {
//     pub prompt: String,
// }

#[derive(Clone)]
pub struct AppState {
    pub(crate) model: Phi3,
    pub(crate) device: Device,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) temperature: Option<f64>,
}

pub struct TextGeneration {
    model: Phi3,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Phi3,
        tokenizer: Tokenizer,
        seed: u64,
        _temp: Option<f64>,
        _top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, Some(0.0), None);

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub(crate) fn run(mut self, prompt: String, sample_len: usize) -> CompletionResponse {
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .unwrap()
            .get_ids()
            .to_vec();

        info!("Got tokens!");

        let eos_token = match self.tokenizer.get_token("</s>") {
            Some(token) => token,
            None => {
                error!("cannot find the </s> token");
                panic!("cannot find the </s> token")
            }
        };

        let mut string = String::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let logits = self.model.forward(&input, start_pos).unwrap();
            let logits = logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )
                    .unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);

            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                info!("Found a token! {}", t);
                string.push_str(&t);
            }
        }
        let id = Uuid::new_v4().to_string();
        let create_time = Instant::now().elapsed().as_secs_f32() as i64;

        CompletionResponse::new(id, string, create_time, "Phi-3".to_string())
    }
}
