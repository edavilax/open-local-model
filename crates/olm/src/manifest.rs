use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

use tensor::Dtype;

use crate::OlmError;

pub fn get_manifest(json: &str) -> Result<Manifest, OlmError> {
    let manifest: Manifest =
        serde_json::from_str(json).map_err(|e| OlmError::InvalidJson(e.to_string()))?;
    if manifest.format_version != 1 {
        return Err(OlmError::FieldValidation(String::from(
            "Only support format version 1",
        )));
    }
    if manifest.model_family != "transformer-decoder" {
        return Err(OlmError::FieldValidation(String::from(
            "Only support model type of transformer-decoder",
        )));
    }
    if manifest.tokenizer.eos_token_ids.len() == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have EOS token IDs",
        )));
    }
    if manifest.hyperparameters.hidden_size == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero hidden size",
        )));
    }
    if manifest.hyperparameters.intermediate_size == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero intermediate size",
        )));
    }
    if manifest.hyperparameters.num_hidden_layers == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero hidden layers",
        )));
    }
    if manifest.hyperparameters.num_attention_heads == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero attention heads",
        )));
    }
    if manifest.hyperparameters.num_key_value_heads == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero KV heads",
        )));
    }
    if manifest.hyperparameters.num_attention_heads % manifest.hyperparameters.num_key_value_heads
        != 0
    {
        return Err(OlmError::FieldValidation(String::from(
            "KV heads must divide the attention heads",
        )));
    }
    if manifest.hyperparameters.head_dim == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero head dimension",
        )));
    }
    if manifest.hyperparameters.head_dim % 2 != 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have even head dimension",
        )));
    }
    if manifest.hyperparameters.vocab_size == 0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero vocab size",
        )));
    }
    if manifest.hyperparameters.rms_norm_eps <= 0.0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero, positive RMS norm EPS",
        )));
    }
    if manifest.tokenizer.bos_token_id.is_some()
        && manifest.tokenizer.bos_token_id.unwrap() as usize >= manifest.hyperparameters.vocab_size
    {
        return Err(OlmError::FieldValidation(String::from(
            "BOS token ID must fit in the vocab size",
        )));
    }
    if manifest
        .tokenizer
        .eos_token_ids
        .iter()
        .any(|x| *x as usize >= manifest.hyperparameters.vocab_size)
    {
        return Err(OlmError::FieldValidation(String::from(
            "EOS token IDs must fit in the vocab size",
        )));
    }
    let tensor_requirements = get_tensor_requirements(&manifest);
    for (tensor_name, req) in tensor_requirements.iter() {
        if req.is_required && !manifest.tensors.contains_key(tensor_name) {
            return Err(OlmError::FieldValidation(format!(
                "{tensor_name} is a required tensor"
            )));
        }
    }
    for (tensor_name, tensor) in manifest.tensors.iter() {
        let tensor_requirement = tensor_requirements.get(tensor_name);
        if tensor_requirement.is_none() {
            return Err(OlmError::FieldValidation(format!(
                "Tensor {tensor_name} is an unrecognized tensor"
            )));
        }
        let tensor_requirement = tensor_requirement.unwrap();
        if tensor.shape != tensor_requirement.shape {
            return Err(OlmError::FieldValidation(format!(
                "Tensor {tensor_name} has an invalid shape"
            )));
        }
        for required_with in &tensor_requirement.required_with {
            if !manifest.tensors.contains_key(required_with) {
                return Err(OlmError::FieldValidation(format!(
                    "Tensor {tensor_name} needs to be accompanied with {required_with}"
                )));
            }
        }
    }
    let theta = match manifest.hyperparameters.rope {
        RopeParameters::Default { theta }
        | RopeParameters::Llama3 { theta, .. }
        | RopeParameters::Yarn { theta, .. } => theta,
    };
    if theta <= 0.0 {
        return Err(OlmError::FieldValidation(String::from(
            "Must have non-zero, positive RoPE theta",
        )));
    }
    Ok(manifest)
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub format_version: u32,
    pub model_family: String,
    pub metadata: Metadata,
    pub tokenizer: Tokenizer,
    pub hyperparameters: Hyperparameters,
    pub tensors: BTreeMap<String, Tensor>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    pub name: String,
    #[serde(default)]
    pub source: String,
    pub context_window: usize,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Tokenizer {
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub add_bos_token: bool,
    #[serde(default)]
    pub chat_template: Option<String>,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Hyperparameters {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope: RopeParameters,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum RopeParameters {
    Default {
        theta: f32,
    },
    Llama3 {
        theta: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: u32,
    },
    Yarn {
        theta: f32,
        factor: f32,
        original_max_position_embeddings: u32,
    },
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

struct TensorRequirements {
    is_required: bool,
    required_with: Vec<String>,
    shape: Vec<usize>,
}

fn get_tensor_requirements(manifest: &Manifest) -> HashMap<String, TensorRequirements> {
    let vocab_size = manifest.hyperparameters.vocab_size;
    let hidden_size = manifest.hyperparameters.hidden_size;
    let num_attention_heads = manifest.hyperparameters.num_attention_heads;
    let head_dim = manifest.hyperparameters.head_dim;
    let num_key_value_heads = manifest.hyperparameters.num_key_value_heads;
    let intermediate_size = manifest.hyperparameters.intermediate_size;

    let q_dim = num_attention_heads * head_dim;
    let kv_dim = num_key_value_heads * head_dim;

    // 1. Global (non-layer) tensors
    let mut tensor_requirements: HashMap<String, TensorRequirements> = HashMap::from([
        (
            "embed_tokens".into(),
            TensorRequirements {
                is_required: true,
                required_with: Vec::new(),
                shape: vec![vocab_size, hidden_size],
            },
        ),
        (
            "final_norm".into(),
            TensorRequirements {
                is_required: true,
                required_with: Vec::new(),
                shape: vec![hidden_size],
            },
        ),
        (
            "lm_head".into(),
            TensorRequirements {
                is_required: false,
                required_with: Vec::new(),
                shape: vec![vocab_size, hidden_size],
            },
        ),
    ]);

    // 2. Per-layer tensors
    for n in 0..manifest.hyperparameters.num_hidden_layers {
        // Attention and norm co-occurrence names for layer n
        let q_bias = format!("layers.{n}.attn.q_bias");
        let k_bias = format!("layers.{n}.attn.k_bias");
        let v_bias = format!("layers.{n}.attn.v_bias");

        let q_norm = format!("layers.{n}.attn.q_norm");
        let k_norm = format!("layers.{n}.attn.k_norm");

        let layer_tensors = [
            // Input norm
            (
                format!("layers.{n}.input_norm"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![hidden_size],
                },
            ),
            // Attention projections
            (
                format!("layers.{n}.attn.q_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![q_dim, hidden_size],
                },
            ),
            (
                format!("layers.{n}.attn.k_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![kv_dim, hidden_size],
                },
            ),
            (
                format!("layers.{n}.attn.v_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![kv_dim, hidden_size],
                },
            ),
            (
                format!("layers.{n}.attn.o_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![hidden_size, q_dim],
                },
            ),
            // Attention biases (all present or all absent)
            (
                q_bias.clone(),
                TensorRequirements {
                    is_required: false,
                    required_with: vec![k_bias.clone(), v_bias.clone()],
                    shape: vec![q_dim],
                },
            ),
            (
                k_bias.clone(),
                TensorRequirements {
                    is_required: false,
                    required_with: vec![q_bias.clone(), v_bias.clone()],
                    shape: vec![kv_dim],
                },
            ),
            (
                v_bias.clone(),
                TensorRequirements {
                    is_required: false,
                    required_with: vec![q_bias, k_bias],
                    shape: vec![kv_dim],
                },
            ),
            // Attention Q/K norms (both present or both absent)
            (
                q_norm.clone(),
                TensorRequirements {
                    is_required: false,
                    required_with: vec![k_norm.clone()],
                    shape: vec![head_dim],
                },
            ),
            (
                k_norm.clone(),
                TensorRequirements {
                    is_required: false,
                    required_with: vec![q_norm],
                    shape: vec![head_dim],
                },
            ),
            // Post-attention norm
            (
                format!("layers.{n}.post_attn_norm"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![hidden_size],
                },
            ),
            // MLP projections
            (
                format!("layers.{n}.mlp.gate_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![intermediate_size, hidden_size],
                },
            ),
            (
                format!("layers.{n}.mlp.up_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![intermediate_size, hidden_size],
                },
            ),
            (
                format!("layers.{n}.mlp.down_proj"),
                TensorRequirements {
                    is_required: true,
                    required_with: Vec::new(),
                    shape: vec![hidden_size, intermediate_size],
                },
            ),
        ];

        tensor_requirements.extend(layer_tensors);
    }
    tensor_requirements
}
