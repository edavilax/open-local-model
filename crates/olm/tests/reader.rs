//! `Model::new` against OLM spec sections 2 and 5.
//! Error tests break one thing in a private copy of the toy-f32 model.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use olm::Model;
use serde_json::{Value, json};
use testutil::{DEFAULT_TOL, assert_close, fixture, load_fixture};

fn toy() -> PathBuf {
    fixture("toy-f32/model")
}

/// A private copy of the toy model, deleted on drop.
struct Copy(PathBuf);

impl Copy {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("olm-reader-{}-{id}", std::process::id()));
        fs::create_dir_all(dir.join("weights")).unwrap();
        for name in ["manifest.json", "tokenizer.json"] {
            fs::copy(toy().join(name), dir.join(name)).unwrap();
        }
        for entry in fs::read_dir(toy().join("weights")).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), dir.join("weights").join(entry.file_name())).unwrap();
        }
        Copy(dir)
    }

    fn weight(&self, name: &str) -> PathBuf {
        self.0.join("weights").join(format!("{name}.bin"))
    }

    fn edit_manifest(&self, edit: impl FnOnce(&mut Value)) {
        let path = self.0.join("manifest.json");
        let mut v: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        edit(&mut v);
        fs::write(&path, v.to_string()).unwrap();
    }
}

impl Drop for Copy {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The error message, lowercased. Panics if the model loaded.
#[track_caller]
fn err(dir: &Path) -> String {
    match Model::new(dir) {
        Ok(_) => panic!("loaded, but it should have been rejected"),
        Err(e) => e.to_string().to_lowercase(),
    }
}

fn decode(path: &Path) -> Vec<f32> {
    let bytes = fs::read(path).unwrap();
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

// Loading

#[test]
fn loads_toy_f32_fixture() {
    let m = Model::new(toy()).unwrap();
    assert_eq!(m.manifest.metadata.name, "toy-f32");
    assert_eq!(m.tensor_data.len(), 26);
    for (name, desc) in &m.manifest.tensors {
        assert_eq!(m.tensor_data[name].shape(), desc.shape, "{name}");
    }
}

#[test]
fn loaded_tensors_hold_the_file_contents() {
    let m = Model::new(toy()).unwrap();
    for (name, tensor) in &m.tensor_data {
        let expected = decode(&toy().join("weights").join(format!("{name}.bin")));
        assert_eq!(tensor.as_f32().unwrap(), expected, "{name}");
    }
}

// File to mmap to tensor to oracle: rows of the loaded embedding, gathered
// by the oracle's token ids, are the oracle's embedding output.
#[test]
fn loaded_embedding_matches_the_oracle() {
    let m = Model::new(toy()).unwrap();
    let embed = m.tensor_data["embed_tokens"].as_f32().unwrap();
    let ids = load_fixture::<u32>("toy-f32/oracle/token_ids.npy");
    let expected = load_fixture::<f32>("toy-f32/oracle/embed.npy");
    for (i, id) in ids.data.iter().enumerate() {
        let start = *id as usize * 64;
        assert_close(
            &embed[start..start + 64],
            testutil::row(&expected, i),
            DEFAULT_TOL,
        );
    }
}

// The tokenizer is built from tokenizer.json plus the manifest's settings.
#[test]
fn loaded_tokenizer_matches_the_oracle() {
    let m = Model::new(toy()).unwrap();
    let prompt = "The quick brown fox jumps over the lazy dog. 你好 🦀 fn main() {";
    let ids = load_fixture::<u32>("toy-f32/oracle/token_ids.npy").data;
    assert_eq!(m.tokenizer.encode(prompt).unwrap(), ids);
    assert_eq!(m.tokenizer.settings().eos_token_ids, [1021, 1023]);
    assert!(m.tokenizer.is_eos(1023));
}

// Mapped storage refuses mutable views; a heap copy would not.
#[test]
fn weights_are_mapped_read_only() {
    let mut m = Model::new(toy()).unwrap();
    assert!(
        m.tensor_data
            .get_mut("final_norm")
            .unwrap()
            .as_mut_f32()
            .is_err()
    );
}

#[test]
fn loads_a_copy() {
    Model::new(&Copy::new().0).unwrap();
}

// Spec 2: files under weights/ that the manifest does not name are ignored.
#[test]
fn ignores_stray_files_in_weights() {
    let c = Copy::new();
    fs::write(c.0.join("weights").join(".DS_Store"), b"junk").unwrap();
    fs::write(c.weight("layers.9.input_norm"), [0u8; 256]).unwrap();
    assert_eq!(Model::new(&c.0).unwrap().tensor_data.len(), 26);
}

// Rejected. The spec wants every error to name the culprit.

#[test]
fn rejects_missing_directory() {
    assert!(Model::new(fixture("nope/model")).is_err());
}

#[test]
fn rejects_missing_manifest_by_name() {
    let c = Copy::new();
    fs::remove_file(c.0.join("manifest.json")).unwrap();
    assert!(err(&c.0).contains("manifest.json"));
}

#[test]
fn rejects_invalid_manifest() {
    let c = Copy::new();
    c.edit_manifest(|v| v["bogus"] = json!(1));
    assert!(err(&c.0).contains("bogus"));
}

// Spec 2: tokenizer.json MUST exist at the root.
#[test]
fn rejects_missing_tokenizer_by_name() {
    let c = Copy::new();
    fs::remove_file(c.0.join("tokenizer.json")).unwrap();
    assert!(err(&c.0).contains("tokenizer.json"));
}

#[test]
fn rejects_corrupt_tokenizer_by_name() {
    let c = Copy::new();
    fs::write(c.0.join("tokenizer.json"), b"{}").unwrap();
    assert!(err(&c.0).contains("tokenizer.json"));
}

#[test]
fn rejects_missing_tensor_file_by_name() {
    let c = Copy::new();
    fs::remove_file(c.weight("layers.1.attn.k_proj")).unwrap();
    assert!(err(&c.0).contains("layers.1.attn.k_proj"));
}

#[test]
fn rejects_truncated_tensor_file_by_name() {
    let c = Copy::new();
    let path = c.weight("layers.0.mlp.up_proj");
    let bytes = fs::read(&path).unwrap();
    fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
    assert!(err(&c.0).contains("layers.0.mlp.up_proj"));
}

#[test]
fn rejects_tensor_file_one_byte_too_long_by_name() {
    let c = Copy::new();
    let path = c.weight("final_norm");
    let mut bytes = fs::read(&path).unwrap();
    bytes.push(0);
    fs::write(&path, bytes).unwrap();
    assert!(err(&c.0).contains("final_norm"));
}

#[test]
fn rejects_empty_tensor_file_by_name() {
    let c = Copy::new();
    fs::write(c.weight("embed_tokens"), []).unwrap();
    assert!(err(&c.0).contains("embed_tokens"));
}

// The manifest is self-consistent for a 1023-token vocab, but the file on
// disk still holds 1024 rows.
#[test]
fn rejects_manifest_shape_that_disagrees_with_the_file() {
    let c = Copy::new();
    c.edit_manifest(|v| {
        v["hyperparameters"]["vocab_size"] = json!(1023);
        v["tokenizer"]["eos_token_ids"] = json!([1021]);
        v["tensors"]["embed_tokens"]["shape"] = json!([1023, 64]);
    });
    assert!(err(&c.0).contains("embed_tokens"));
}
