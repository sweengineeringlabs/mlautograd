use swe_ml_tensor::TensorError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MlError {
    #[error("Tensor error: {0}")]
    TensorError(#[from] TensorError),

    #[error("Tape error: {0}")]
    TapeError(String),

    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Layer error: {0}")]
    Layer(String),

    #[error("Training error: {0}")]
    TrainingError(String),
}
