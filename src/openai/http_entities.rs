use candle_core::{Device, DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::{error, info};
use crate::core::output_stream::TokenOutputStream;
use candle_transformers::models::gemma::Model as Gemma;

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
    choices: Vec<Choice>,
    usage: Usage,
}

impl CompletionResponse {
    pub fn new(id: String, object: String, created: i64, model: String, choices: Vec<Choice>, usage: Usage) -> Self {
        Self { id, object, created, model, choices, usage }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Choice {
    text: String,
    index: i64,
    logprobs: Option<f64>,
    finish_reason: String,
}

impl Choice {
    pub fn new(text: String, index: i64, logprobs: Option<f64>, finish_reason: String) -> Self {
        Self { text, index, logprobs, finish_reason }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Usage {
    prompt_tokens: i64,
    completion_tokens: i64,
    total_tokens: i64,
}

impl Usage {
    pub fn new(prompt_tokens: i64, completion_tokens: i64, total_tokens: i64) -> Self {
        Self { prompt_tokens, completion_tokens, total_tokens }
    }


    pub fn prompt_tokens(&self) -> i64 {
        self.prompt_tokens
    }
    pub fn completion_tokens(&self) -> i64 {
        self.completion_tokens
    }
    pub fn total_tokens(&self) -> i64 {
        self.total_tokens
    }
}


// #[derive(Deserialize)]
// pub struct Prompt {
//     pub prompt: String,
// }

#[derive(Clone)]
pub struct AppState {
    pub(crate) model: Gemma,
    pub(crate) device: Device,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) temperature: Option<f64>,
}

pub struct TextGeneration {
    model: Gemma,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        model: Gemma,
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

    pub(crate) fn run(mut self, prompt: String, sample_len: usize) -> String {
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

        string
    }
}
