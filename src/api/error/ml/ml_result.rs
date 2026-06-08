use crate::api::error::ml::ml_error::MlError;

pub type MlResult<T> = Result<T, MlError>;
