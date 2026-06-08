//! Tests for MlResult (src/api/error/ml/ml_result.rs).
//!
//! @covers: MlResult

use mlautograd::{MlError, MlResult};

fn ok_result() -> MlResult<i32> {
    Ok(42)
}

fn err_result() -> MlResult<i32> {
    Err(MlError::TapeError("forced error".into()))
}

#[test]
fn test_type_ml_result_ok_variant_holds_value() {
    let result = ok_result();
    assert!(result.is_ok());
    assert_eq!(result.expect("ok"), 42);
}

#[test]
fn test_type_ml_result_err_variant_holds_error() {
    let result = err_result();
    assert!(result.is_err());
    let err = result.expect_err("err");
    assert!(err.to_string().contains("forced error"));
}
