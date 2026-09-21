use std::fmt;
use std::write;

pub mod manifest;

#[derive(Debug)]
pub enum OlmError {
    InvalidJson(String),
    RankMismatch(String, usize),
    FieldValidation(String),
}

impl fmt::Display for OlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OlmError::InvalidJson(message) => write!(f, "{message}"),
            OlmError::RankMismatch(name, rank) => {
                write!(f, "{name} has unsupported rank of {rank}")
            }
            OlmError::FieldValidation(message) => write!(f, "{message}"),
        }
    }
}
