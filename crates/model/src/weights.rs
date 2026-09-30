use std::collections::BTreeMap;

use anyhow::{Ok, Result, bail};

use tensor::Tensor;

use crate::ModelError;

pub struct Weights {
    pub embed: Tensor,
    pub layers: Vec<Layer>,
    pub final_norm: Tensor,
    pub lm_head: Option<Tensor>,
}

impl Weights {
    pub fn from_tensor_data(
        mut tensor_data: BTreeMap<String, Tensor>,
        num_layers: usize,
    ) -> Result<Weights> {
        let data = &mut tensor_data;
        // Collect the layers.
        let mut layers: Vec<Layer> = Vec::new();
        for i in 0..num_layers {
            layers.push(Layer {
                attn_norm: get_tensor(data, &format!("layers.{i}.input_norm"))?,
                attn: Attention {
                    q: get_tensor(data, &format!("layers.{i}.attn.q_proj"))?,
                    k: get_tensor(data, &format!("layers.{i}.attn.k_proj"))?,
                    v: get_tensor(data, &format!("layers.{i}.attn.v_proj"))?,
                    o: get_tensor(data, &format!("layers.{i}.attn.o_proj"))?,
                    q_bias: get_optional_tensor(data, &format!("layers.{i}.attn.q_bias")),
                    k_bias: get_optional_tensor(data, &format!("layers.{i}.attn.k_bias")),
                    v_bias: get_optional_tensor(data, &format!("layers.{i}.attn.v_bias")),
                    q_norm: get_optional_tensor(data, &format!("layers.{i}.attn.q_norm")),
                    k_norm: get_optional_tensor(data, &format!("layers.{i}.attn.k_norm")),
                },
                mlp_norm: get_tensor(data, &format!("layers.{i}.post_attn_norm"))?,
                mlp: Mlp {
                    gate: get_tensor(data, &format!("layers.{i}.mlp.gate_proj"))?,
                    up: get_tensor(data, &format!("layers.{i}.mlp.up_proj"))?,
                    down: get_tensor(data, &format!("layers.{i}.mlp.down_proj"))?,
                },
            });
        }
        // Collect all the weights.
        let weights = Weights {
            embed: get_tensor(data, "embed_tokens")?,
            layers: layers,
            final_norm: get_tensor(data, "final_norm")?,
            lm_head: get_optional_tensor(data, "lm_head"),
        };
        // Check that the map has no extra weights before returning.
        let extra_weight_names: Vec<String> = data.keys().cloned().collect();
        if !extra_weight_names.is_empty() {
            bail!(ModelError::ExtraWeightsFound {
                additional_names: extra_weight_names
            })
        }
        Ok(weights)
    }
}

pub struct Layer {
    pub attn_norm: Tensor,
    pub attn: Attention,
    pub mlp_norm: Tensor,
    pub mlp: Mlp,
}

pub struct Attention {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub o: Tensor,
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
}

pub struct Mlp {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

// Private helpers

fn get_tensor(data: &mut BTreeMap<String, Tensor>, tensor_name: &str) -> Result<Tensor> {
    let tensor = data
        .remove(tensor_name)
        .ok_or_else(|| ModelError::WeightNotFound {
            name: String::from(tensor_name),
        })?;
    Ok(tensor)
}

fn get_optional_tensor(data: &mut BTreeMap<String, Tensor>, tensor_name: &str) -> Option<Tensor> {
    data.remove(tensor_name)
}
