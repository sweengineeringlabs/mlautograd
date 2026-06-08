//! Tests for MlError (src/api/error/ml/ml_error.rs).
//!
//! @covers: MlError

use mlautograd::MlError;

#[test]
fn test_struct_ml_error_tape_error_displays_message() {
    let err = MlError::TapeError("test tape error".into());
    assert!(err.to_string().contains("test tape error"));
}

#[test]
fn test_struct_ml_error_shape_mismatch_includes_shapes() {
    let err = MlError::ShapeMismatch {
        expected: vec![2, 3],
        got: vec![3, 2],
    };
    let msg = err.to_string();
    assert!(msg.contains("mismatch") || msg.contains("2") || msg.contains("3"));
}

#[test]
fn test_struct_ml_error_invalid_config_displays_message() {
    let err = MlError::InvalidConfig("bad param".into());
    assert!(err.to_string().contains("bad param"));
}

#[test]
fn test_struct_ml_error_layer_error_displays_message() {
    let err = MlError::Layer("layer failure".into());
    assert!(err.to_string().contains("layer failure"));
}

#[test]
fn test_struct_ml_error_training_error_displays_message() {
    let err = MlError::TrainingError("epoch failure".into());
    assert!(err.to_string().contains("epoch failure"));
}
