use crate::api::error::ml::ml_result::MlResult;
use crate::api::types::tensor::Tensor;

/// Validates a tensor before it enters a computation.
pub trait Validator {
    fn validate(&self, input: &Tensor) -> MlResult<()>;
}
