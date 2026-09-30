use thiserror::Error;

pub mod attention;
pub mod config;
pub mod weights;

#[derive(Debug, Error)]
enum ModelError {
    #[error("Weight not found for {name}")]
    WeightNotFound { name: String },
    #[error("Extra weights found: {}", format_names(.additional_names))]
    ExtraWeightsFound { additional_names: Vec<String> },
}

fn format_names(names: &[String]) -> String {
    names.join(", ")
}
