use crate::api::error::ml::ml_error::MlError;
use crate::api::error::ml::ml_result::MlResult;
use crate::api::traits::validator::Validator;
use crate::api::types::tensor::Tensor;

pub(crate) struct DefaultValidator;

impl Validator for DefaultValidator {
    fn validate(&self, input: &Tensor) -> MlResult<()> {
        if input.numel() == 0 {
            return Err(MlError::TapeError("empty tensor".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: validate
    #[test]
    fn test_default_validator_accepts_non_empty_tensor() {
        let v = DefaultValidator;
        let t = Tensor::ones(vec![2, 3]);
        assert!(v.validate(&t).is_ok());
    }

    /// @covers: validate
    #[test]
    fn test_default_validator_rejects_empty_tensor() {
        let v = DefaultValidator;
        let t = Tensor::zeros(vec![0]);
        assert!(v.validate(&t).is_err());
    }
}
