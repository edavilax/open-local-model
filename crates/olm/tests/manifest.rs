//! `get_manifest` against OLM spec sections 3 and 5.
//! Each test starts from a valid manifest and breaks one thing.

use olm::manifest::{Manifest, RopeParameters, get_manifest};
use serde_json::{Value, json};
use tensor::Dtype;

// With 4 heads, 2 KV heads and head_dim 6, every size is distinct
// (q 24, kv 12), so a transposed shape never matches by luck.
const HIDDEN: usize = 10;
const INTER: usize = 14;
const VOCAB: usize = 32;

fn desc(shape: &[usize]) -> Value {
    json!({ "shape": shape, "dtype": "f32" })
}

fn manifest(layers: usize, heads: usize, kv_heads: usize, head_dim: usize) -> Value {
    let (q, kv) = (heads * head_dim, kv_heads * head_dim);
    let mut v = json!({
        "format_version": 1,
        "model_family": "transformer-decoder",
        "metadata": { "name": "unit", "source": "org/unit", "context_window": 128 },
        "tokenizer": {
            "bos_token_id": null,
            "eos_token_ids": [VOCAB - 1],
            "add_bos_token": false,
            "chat_template": "{{ messages }}"
        },
        "hyperparameters": {
            "hidden_size": HIDDEN,
            "intermediate_size": INTER,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope": { "type": "default", "theta": 10000.0 }
        },
        "tensors": {
            "embed_tokens": desc(&[VOCAB, HIDDEN]),
            "final_norm": desc(&[HIDDEN])
        }
    });
    for n in 0..layers {
        let t = &mut v["tensors"];
        t[format!("layers.{n}.input_norm")] = desc(&[HIDDEN]);
        t[format!("layers.{n}.attn.q_proj")] = desc(&[q, HIDDEN]);
        t[format!("layers.{n}.attn.k_proj")] = desc(&[kv, HIDDEN]);
        t[format!("layers.{n}.attn.v_proj")] = desc(&[kv, HIDDEN]);
        t[format!("layers.{n}.attn.o_proj")] = desc(&[HIDDEN, q]);
        t[format!("layers.{n}.attn.q_bias")] = desc(&[q]);
        t[format!("layers.{n}.attn.k_bias")] = desc(&[kv]);
        t[format!("layers.{n}.attn.v_bias")] = desc(&[kv]);
        t[format!("layers.{n}.post_attn_norm")] = desc(&[HIDDEN]);
        t[format!("layers.{n}.mlp.gate_proj")] = desc(&[INTER, HIDDEN]);
        t[format!("layers.{n}.mlp.up_proj")] = desc(&[INTER, HIDDEN]);
        t[format!("layers.{n}.mlp.down_proj")] = desc(&[HIDDEN, INTER]);
    }
    v
}

fn valid() -> Value {
    manifest(2, 4, 2, 6)
}

fn insert(v: &mut Value, path: &str, key: &str, new: Value) {
    v.pointer_mut(path)
        .unwrap()
        .as_object_mut()
        .unwrap()
        .insert(key.into(), new);
}

fn remove(v: &mut Value, path: &str, key: &str) {
    v.pointer_mut(path)
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove(key)
        .unwrap();
}

#[track_caller]
fn ok(v: &Value) -> Manifest {
    get_manifest(&v.to_string()).unwrap()
}

/// The error message, lowercased. Panics if the manifest was accepted.
#[track_caller]
fn err(v: &Value) -> String {
    get_manifest(&v.to_string())
        .unwrap_err()
        .to_string()
        .to_lowercase()
}

fn fixture(model: &str) -> String {
    std::fs::read_to_string(testutil::fixture(&format!("{model}/model/manifest.json"))).unwrap()
}

// Fixtures

#[test]
fn parses_toy_f32_fixture() {
    let m = get_manifest(&fixture("toy-f32")).unwrap();
    assert_eq!(m.metadata.name, "toy-f32");
    assert_eq!(m.hyperparameters.hidden_size, 64);
    assert_eq!(m.hyperparameters.num_hidden_layers, 2);
    assert!(
        matches!(m.hyperparameters.rope, RopeParameters::Default { theta } if theta == 10000.0)
    );
    assert_eq!(m.tokenizer.eos_token_ids, [1021, 1023]);
    assert_eq!(m.tensors.len(), 26);
    assert_eq!(m.tensors["embed_tokens"].shape, [1024, 64]);
    assert_eq!(m.tensors["embed_tokens"].dtype, Dtype::F32);
}

// Move a fixture to the accepting side when its dtype lands in the tensor crate.
#[test]
fn rejects_toy_fixtures_with_unsupported_dtypes() {
    for model in ["toy-bf16", "toy-q8_0", "toy-q4_0"] {
        assert!(get_manifest(&fixture(model)).is_err(), "{model}");
    }
}

// Accepted

#[test]
fn accepts_valid_manifest() {
    assert_eq!(ok(&valid()).tensors.len(), 26);
}

#[test]
fn accepts_other_valid_hyperparameters() {
    ok(&manifest(1, 4, 2, 6)); // one layer
    ok(&manifest(2, 4, 4, 6)); // KV heads == heads
    ok(&manifest(2, 4, 2, 8)); // head_dim unrelated to hidden_size / heads
}

#[test]
fn accepts_untied_lm_head() {
    let mut v = valid();
    v["tensors"]["lm_head"] = desc(&[VOCAB, HIDDEN]);
    ok(&v);
}

#[test]
fn accepts_layer_without_biases() {
    let mut v = valid();
    for b in ["q_bias", "k_bias", "v_bias"] {
        remove(&mut v, "/tensors", &format!("layers.1.attn.{b}"));
    }
    ok(&v);
}

#[test]
fn accepts_qk_norm() {
    let mut v = valid();
    v["tensors"]["layers.0.attn.q_norm"] = desc(&[6]);
    v["tensors"]["layers.0.attn.k_norm"] = desc(&[6]);
    ok(&v);
}

#[test]
fn accepts_missing_source() {
    let mut v = valid();
    remove(&mut v, "/metadata", "source");
    ok(&v);
}

#[test]
fn accepts_bos_token_id_int_or_absent() {
    let mut v = valid();
    v["tokenizer"]["bos_token_id"] = json!(7);
    assert_eq!(ok(&v).tokenizer.bos_token_id, Some(7));
    remove(&mut v, "/tokenizer", "bos_token_id");
    assert_eq!(ok(&v).tokenizer.bos_token_id, None);
}

#[test]
fn accepts_null_chat_template() {
    let mut v = valid();
    v["tokenizer"]["chat_template"] = Value::Null;
    assert_eq!(ok(&v).tokenizer.chat_template, None);
}

#[test]
fn parses_llama3_and_yarn_rope() {
    let mut v = valid();
    v["hyperparameters"]["rope"] = json!({
        "type": "llama3", "theta": 500000.0, "factor": 32.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
    });
    assert!(
        matches!(ok(&v).hyperparameters.rope, RopeParameters::Llama3 { factor, .. } if factor == 32.0)
    );

    v["hyperparameters"]["rope"] = json!({
        "type": "yarn", "theta": 1000000.0, "factor": 4.0,
        "original_max_position_embeddings": 32768
    });
    assert!(
        matches!(ok(&v).hyperparameters.rope, RopeParameters::Yarn { factor, .. } if factor == 4.0)
    );
}

// Rejected: schema

#[test]
fn rejects_invalid_json() {
    for text in ["", "{", "[]", "null"] {
        assert!(get_manifest(text).is_err(), "{text:?}");
    }
}

#[test]
fn rejects_unknown_fields() {
    let paths = [
        "",
        "/metadata",
        "/tokenizer",
        "/hyperparameters",
        "/hyperparameters/rope",
        "/tensors/final_norm",
    ];
    for path in paths {
        let mut v = valid();
        insert(&mut v, path, "bogus", json!(1));
        assert!(err(&v).contains("bogus"), "{path}");
    }
}

#[test]
fn rejects_missing_fields() {
    let fields = [
        ("", "format_version"),
        ("", "model_family"),
        ("", "tensors"),
        ("/metadata", "name"),
        ("/metadata", "context_window"),
        ("/tokenizer", "eos_token_ids"),
        ("/hyperparameters", "hidden_size"),
        ("/hyperparameters", "head_dim"),
        ("/hyperparameters", "rope"),
        ("/hyperparameters/rope", "type"),
        ("/hyperparameters/rope", "theta"),
        ("/tensors/final_norm", "shape"),
        ("/tensors/final_norm", "dtype"),
    ];
    for (path, key) in fields {
        let mut v = valid();
        remove(&mut v, path, key);
        assert!(err(&v).contains(key), "{path}/{key}");
    }
}

#[test]
fn rejects_wrong_types() {
    let cases = [
        ("", "format_version", json!("1")),
        ("/tokenizer", "eos_token_ids", json!([-1])),
        ("/hyperparameters", "hidden_size", json!(10.5)),
        ("/hyperparameters", "hidden_size", json!(-10)),
        ("/tensors/final_norm", "shape", json!([-10])),
        ("/tensors", "final_norm", json!("f32")),
    ];
    for (path, key, value) in cases {
        let mut v = valid();
        insert(&mut v, path, key, value.clone());
        err(&v);
    }
}

// Delete a dtype from this list when the tensor crate gains it.
#[test]
fn rejects_unsupported_dtypes_by_name() {
    for dtype in ["f16", "bf16", "q8_0", "q4_0", "q4_1", "f64"] {
        let mut v = valid();
        v["tensors"]["final_norm"]["dtype"] = json!(dtype);
        assert!(err(&v).contains(dtype), "{dtype}");
    }
}

#[test]
fn rejects_bad_rope() {
    let mut v = valid();
    v["hyperparameters"]["rope"]["type"] = json!("linear");
    assert!(err(&v).contains("linear"));

    let mut v = valid();
    v["hyperparameters"]["rope"]["factor"] = json!(8.0); // llama3 field on default rope
    assert!(err(&v).contains("factor"));

    let mut v = valid();
    v["hyperparameters"]["rope"] = json!({ "type": "llama3", "theta": 500000.0 });
    assert!(err(&v).contains("missing field"));
}

// Rejected: rules

#[test]
fn rejects_wrong_format_version() {
    let mut v = valid();
    v["format_version"] = json!(2);
    assert!(err(&v).contains("version"));
}

#[test]
fn rejects_wrong_model_family() {
    let mut v = valid();
    v["model_family"] = json!("moe");
    assert!(err(&v).contains("model"));
}

#[test]
fn rejects_empty_eos_token_ids() {
    let mut v = valid();
    v["tokenizer"]["eos_token_ids"] = json!([]);
    assert!(err(&v).contains("eos"));
}

#[test]
fn rejects_zero_hyperparameters() {
    let fields = [
        ("hidden_size", "hidden size"),
        ("intermediate_size", "intermediate size"),
        ("num_hidden_layers", "hidden layers"),
        ("num_attention_heads", "attention heads"),
        ("num_key_value_heads", "kv heads"),
        ("head_dim", "head dim"),
        ("vocab_size", "vocab size"),
    ];
    for (field, needle) in fields {
        let mut v = valid();
        v["hyperparameters"][field] = json!(0);
        assert!(err(&v).contains(needle), "{field}");
    }
}

#[test]
fn rejects_non_positive_rms_norm_eps() {
    for eps in [0.0, -1e-6] {
        let mut v = valid();
        v["hyperparameters"]["rms_norm_eps"] = json!(eps);
        assert!(err(&v).contains("rms norm eps"), "{eps}");
    }
}

#[test]
fn rejects_non_positive_rope_theta() {
    for theta in [0.0, -10000.0] {
        let mut v = valid();
        v["hyperparameters"]["rope"]["theta"] = json!(theta);
        assert!(err(&v).contains("theta"), "{theta}");
    }
}

// Tensors are shaped for 3 KV heads, so divisibility is the only broken rule.
#[test]
fn rejects_kv_heads_not_dividing_heads() {
    assert!(err(&manifest(2, 4, 3, 6)).contains("kv heads"));
}

// Tensors are shaped for head_dim 5, so evenness is the only broken rule.
#[test]
fn rejects_odd_head_dim() {
    assert!(err(&manifest(2, 4, 2, 5)).contains("head dim"));
}

#[test]
fn rejects_token_ids_past_vocab() {
    let mut v = valid();
    v["tokenizer"]["eos_token_ids"] = json!([0, VOCAB]);
    assert!(err(&v).contains("eos"));

    let mut v = valid();
    v["tokenizer"]["bos_token_id"] = json!(VOCAB);
    assert!(err(&v).contains("bos"));
}

#[test]
fn rejects_missing_required_tensor() {
    for name in [
        "embed_tokens",
        "final_norm",
        "layers.0.attn.q_proj",
        "layers.1.mlp.down_proj",
    ] {
        let mut v = valid();
        remove(&mut v, "/tensors", name);
        assert!(err(&v).contains(name), "{name}");
    }
}

#[test]
fn rejects_missing_layer() {
    let mut v = manifest(1, 4, 2, 6);
    v["hyperparameters"]["num_hidden_layers"] = json!(2);
    assert!(err(&v).contains("layers.1."));
}

#[test]
fn rejects_unknown_tensor_names() {
    let names = [
        "layers.2.input_norm",
        "norm.weight",
        "layers.0.attn.q_proj.weight",
    ];
    for name in names {
        let mut v = valid();
        v["tensors"][name] = desc(&[HIDDEN]);
        assert!(err(&v).contains(name), "{name}");
    }
}

#[test]
fn rejects_wrong_shapes() {
    let cases: [(&str, &[usize]); 7] = [
        ("final_norm", &[HIDDEN + 1]),
        ("final_norm", &[HIDDEN, 1]),
        ("final_norm", &[]),
        ("embed_tokens", &[VOCAB + 1, HIDDEN]),
        ("layers.0.attn.q_proj", &[HIDDEN, 24]), // transposed
        ("layers.0.attn.k_proj", &[24, HIDDEN]), // sized for query heads
        ("layers.1.mlp.down_proj", &[INTER, HIDDEN]),
    ];
    for (name, shape) in cases {
        let mut v = valid();
        v["tensors"][name]["shape"] = json!(shape);
        assert!(err(&v).contains(name), "{name} {shape:?}");
    }
}

#[test]
fn rejects_partial_bias_set() {
    for missing in ["q_bias", "k_bias", "v_bias"] {
        let mut v = valid();
        remove(&mut v, "/tensors", &format!("layers.1.attn.{missing}"));
        assert!(err(&v).contains(missing), "{missing}");
    }
}

#[test]
fn rejects_half_a_qk_norm_pair() {
    let mut v = valid();
    v["tensors"]["layers.0.attn.q_norm"] = desc(&[6]);
    assert!(err(&v).contains("k_norm"));
}
