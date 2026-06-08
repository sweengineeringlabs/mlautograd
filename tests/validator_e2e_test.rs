//! Tests for the Validator trait (src/api/traits/validator.rs).
//!
//! @covers: Validator

use mlautograd::{TapeContext, Tensor};

/// Verify that set_grad silently rejects an empty tensor (DefaultValidator in action).
#[test]
fn test_trait_validator_set_grad_rejects_empty_tensor() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("tensor");
    let empty_grad = Tensor::zeros(vec![0]);
    TapeContext::set_grad(&t, empty_grad);
    // DefaultValidator rejects empty tensors; grad must not be stored.
    assert!(
        TapeContext::grad(&t).is_none(),
        "grad must not be stored for an empty gradient tensor"
    );
}

/// Verify that set_grad accepts a valid non-empty tensor.
#[test]
fn test_trait_validator_set_grad_accepts_non_empty_tensor() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("tensor");
    let grad = Tensor::ones(vec![2]);
    TapeContext::set_grad(&t, grad);
    assert!(
        TapeContext::grad(&t).is_some(),
        "grad must be stored for a valid non-empty gradient tensor"
    );
}
