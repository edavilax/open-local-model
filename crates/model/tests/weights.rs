//! `Weights::from_tensor_data` against the toy-f32 model.

use std::collections::BTreeMap;

use model::weights::Weights;
use olm::Model;
use tensor::{Dtype, Storage, Tensor};
use testutil::{fixture, mmap_f32};

const LAYERS: usize = 2;

fn toy_data() -> BTreeMap<String, Tensor> {
    Model::new(fixture("toy-f32/model")).unwrap().tensor_data
}

fn f32_tensor(shape: Vec<usize>, values: &[f32]) -> Tensor {
    Tensor::new(shape, Dtype::F32, Storage::Mmap(mmap_f32(values))).unwrap()
}

#[track_caller]
fn err(data: BTreeMap<String, Tensor>, num_layers: usize) -> String {
    match Weights::from_tensor_data(data, num_layers) {
        Ok(_) => panic!("built weights, but it should have failed"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn each_field_holds_the_tensor_it_is_named_for() {
    let want = toy_data();
    let w = Weights::from_tensor_data(toy_data(), LAYERS).unwrap();

    let mut got: Vec<(String, &Tensor)> = vec![
        ("embed_tokens".into(), &w.embed),
        ("final_norm".into(), &w.final_norm),
    ];
    for (i, l) in w.layers.iter().enumerate() {
        got.push((format!("layers.{i}.input_norm"), &l.attn_norm));
        got.push((format!("layers.{i}.attn.q_proj"), &l.attn.q));
        got.push((format!("layers.{i}.attn.k_proj"), &l.attn.k));
        got.push((format!("layers.{i}.attn.v_proj"), &l.attn.v));
        got.push((format!("layers.{i}.attn.o_proj"), &l.attn.o));
        got.push((
            format!("layers.{i}.attn.q_bias"),
            l.attn.q_bias.as_ref().unwrap(),
        ));
        got.push((
            format!("layers.{i}.attn.k_bias"),
            l.attn.k_bias.as_ref().unwrap(),
        ));
        got.push((
            format!("layers.{i}.attn.v_bias"),
            l.attn.v_bias.as_ref().unwrap(),
        ));
        got.push((format!("layers.{i}.post_attn_norm"), &l.mlp_norm));
        got.push((format!("layers.{i}.mlp.gate_proj"), &l.mlp.gate));
        got.push((format!("layers.{i}.mlp.up_proj"), &l.mlp.up));
        got.push((format!("layers.{i}.mlp.down_proj"), &l.mlp.down));
    }

    assert_eq!(got.len(), want.len());
    for (name, t) in got {
        let expected = &want[name.as_str()];
        assert_eq!(t.shape(), expected.shape(), "{name}");
        assert!(
            t.as_f32().unwrap() == expected.as_f32().unwrap(),
            "{name} holds the wrong data"
        );
    }
}

#[test]
fn toy_is_tied_with_qkv_bias_and_no_qk_norm() {
    let w = Weights::from_tensor_data(toy_data(), LAYERS).unwrap();

    assert_eq!(w.layers.len(), LAYERS);
    assert!(w.lm_head.is_none());
    for l in &w.layers {
        assert!(l.attn.q_bias.is_some());
        assert!(l.attn.k_bias.is_some());
        assert!(l.attn.v_bias.is_some());
        assert!(l.attn.q_norm.is_none());
        assert!(l.attn.k_norm.is_none());
    }
}

#[test]
fn untied_model_keeps_its_lm_head() {
    let mut data = toy_data();
    data.insert(
        "lm_head".into(),
        f32_tensor(vec![1024, 64], &vec![0.5; 1024 * 64]),
    );

    let w = Weights::from_tensor_data(data, LAYERS).unwrap();

    let head = w.lm_head.expect("lm_head should be loaded");
    assert_eq!(head.shape(), &[1024, 64]);
    assert_eq!(head.as_f32().unwrap()[0], 0.5);
}

#[test]
fn qk_norm_is_loaded_per_layer() {
    let mut data = toy_data();
    for i in 0..LAYERS {
        let q = i as f32 + 0.25;
        let k = i as f32 + 0.75;
        data.insert(
            format!("layers.{i}.attn.q_norm"),
            f32_tensor(vec![16], &[q; 16]),
        );
        data.insert(
            format!("layers.{i}.attn.k_norm"),
            f32_tensor(vec![16], &[k; 16]),
        );
    }

    let w = Weights::from_tensor_data(data, LAYERS).unwrap();

    for (i, l) in w.layers.iter().enumerate() {
        assert_eq!(
            l.attn.q_norm.as_ref().unwrap().as_f32().unwrap()[0],
            i as f32 + 0.25
        );
        assert_eq!(
            l.attn.k_norm.as_ref().unwrap().as_f32().unwrap()[0],
            i as f32 + 0.75
        );
    }
}

#[test]
fn missing_tensor_is_named_in_the_error() {
    let mut data = toy_data();
    data.remove("layers.1.mlp.up_proj");

    let e = err(data, LAYERS);
    assert!(e.contains("layers.1.mlp.up_proj"), "{e}");
}

#[test]
fn unused_tensor_is_named_in_the_error() {
    let mut data = toy_data();
    data.insert(
        "layers.0.attn.o_bias".into(),
        f32_tensor(vec![64], &[0.0; 64]),
    );

    let e = err(data, LAYERS);
    assert!(e.contains("layers.0.attn.o_bias"), "{e}");
}

#[test]
fn fewer_layers_than_the_data_holds_is_an_error() {
    let e = err(toy_data(), 1);
    assert!(e.contains("layers.1."), "{e}");
}
