use anyhow::Result;
use memmap2::Mmap;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

pub mod manifest;

pub struct Model {
    pub manifest: manifest::Manifest,
    pub tensor_data: BTreeMap<String, tensor::Tensor>,
}

impl Model {
    pub fn new(root_dir: impl AsRef<Path>) -> Result<Self> {
        let root = root_dir.as_ref();
        // Get the manifest from root.
        let manifest_file = root.join("manifest.json");
        let manifest_contents =
            std::fs::read_to_string(manifest_file).map_err(|err| OlmError::FileReadError {
                fname: String::from("manifest.json"),
                details: err.to_string(),
            })?;
        let manifest = manifest::get_manifest(&manifest_contents)?;
        // Get the tensor data from manifest.
        let mut tensor_data: BTreeMap<String, tensor::Tensor> = BTreeMap::new();
        let tensor_root = root.join("weights");
        for (tensor_name, tensor_metadata) in manifest.tensors.iter() {
            let tensor_fname = tensor_root.join(tensor_name.clone() + ".bin");
            let tensor_file =
                std::fs::File::open(tensor_fname).map_err(|err| OlmError::FileReadError {
                    fname: tensor_name.clone() + ".bin",
                    details: err.to_string(),
                })?;
            let mmap = unsafe { Mmap::map(&tensor_file)? };
            let tensor = tensor::Tensor::new(
                tensor_metadata.shape.clone(),
                tensor_metadata.dtype,
                tensor::Storage::Mmap(mmap),
            )
            .map_err(|err| OlmError::TensorReadError {
                name: tensor_name.clone(),
                details: err.to_string(),
            })?;
            tensor_data.insert(tensor_name.clone(), tensor);
        }
        Ok(Model {
            manifest: manifest,
            tensor_data: tensor_data,
        })
    }
}

#[derive(Debug, Error)]
enum OlmError {
    #[error("{name} has unsupported rank of {rank}")]
    RankMismatch { name: String, rank: usize },
    #[error("{0}")]
    FieldValidation(String),
    #[error("Failed to read {fname}: {details}")]
    FileReadError { fname: String, details: String },
    #[error("{name} cannot be converted to a tensor: {details}")]
    TensorReadError { name: String, details: String },
}
