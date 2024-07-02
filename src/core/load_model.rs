use std::collections::HashSet;

use anyhow::Error as E;
use candle_core::{Device, DType};
use candle_nn::{Activation, VarBuilder};
use candle_transformers::models::phi3::{Config, Model as Phi3};
use hf_hub::{Repo, RepoType};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use serde::{Deserialize, Deserializer};
use tokenizers::Tokenizer;
use crate::openai::http_entities::AppState;

pub fn hub_load_safe_tensors(repo: &ApiRepo,
                             json_file: &str, ) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: Weightmaps = serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;

    let pathbufs: Vec<std::path::PathBuf> = json
        .weight_map
        .iter()
        .map(|f| repo.get(f).unwrap())
        .collect();

    Ok(pathbufs)
}

// Custom deserializer for the weight_map to directly extract values into a HashSet
fn deserialize_weight_map<'de, D>(deserializer: D) -> anyhow::Result<HashSet<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let map = serde_json::Value::deserialize(deserializer)?;
    match map {
        serde_json::Value::Object(obj) => Ok(obj
            .values()
            .filter_map(|v| v.as_str().map(ToString::to_string))
            .collect::<HashSet<String>>()),
        _ => Err(serde::de::Error::custom(
            "Expected an object for weight_map",
        )),
    }
}

fn get_tokenizer(repo: &ApiRepo) -> anyhow::Result<Tokenizer> {
    let tokenizer_filename = repo.get("tokenizer.json")?;

    Tokenizer::from_file(tokenizer_filename).map_err(E::msg)
}

fn get_device() -> Device {
    let device_cuda = Device::new_cuda(0);
    let device_metal = Device::new_metal(0);

    let device = device_metal.or(device_cuda).unwrap_or(Device::Cpu);

    device
}

#[derive(Debug, Deserialize)]
struct Weightmaps {
    #[serde(deserialize_with = "deserialize_weight_map")]
    weight_map: HashSet<String>,
}

fn get_repo(token: String) -> anyhow::Result<ApiRepo> {
    let api = ApiBuilder::new().with_token(Some(token)).build()?;

    let model_id = "microsoft/Phi-3-mini-4k-instruct".to_string();

    Ok(api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        "ff07dc01615f8113924aed013115ab2abd32115b".to_string(),
    )))
}

pub fn initialise_model(token: String) -> anyhow::Result<AppState> {
    let repo = get_repo(token)?;
    let tokenizer = get_tokenizer(&repo)?;

    let device = get_device();

    let filenames = hub_load_safe_tensors(&repo, "model.safetensors.index.json")?;

    let config = Config {
        vocab_size: 32064,
        hidden_act: Activation::Silu,
        hidden_size: 3072,
        intermediate_size: 8192,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 32,
        rms_norm_eps: 1e-05,
        rope_theta: 10000.0,
        bos_token_id: Some(1),
        eos_token_id: Some(32000),
        rope_scaling: None,
        max_position_embeddings: 4096,
    };

    let model = {
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        Phi3::new(&config, vb)?
    };

    Ok((model, device, tokenizer, Some(0.7)).into())
}