use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;
use thiserror::Error;

pub mod manifest;

pub struct Model {
    pub manifest: manifest::Manifest,
    pub tensor_data: BTreeMap<String, tensor::Tensor>,
}

impl Model {
    pub fn new(path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        let _ = std::fs::read_to_string(path)?;
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum OlmError {
    #[error("{name} has unsupported rank of {rank}")]
    RankMismatch { name: String, rank: usize },
    #[error("{0}")]
    FieldValidation(String),
    #[error("{name} cannot be converted to a tensor: {source}")]
    TensorError {
        name: String,
        source: tensor::TensorError,
    },
}
