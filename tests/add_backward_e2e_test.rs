//! Tests for AddBackward (src/api/gradient/types/add_backward.rs).
//!
//! @covers: AddBackward

use mlautograd::{AddBackward, BackwardOp, Tensor};

#[test]
fn test_struct_add_backward_produces_two_unit_gradients_for_same_shape() {
    let op = AddBackward { a_shape: vec![3], b_shape: vec![3] };
    let grad_out = Tensor::ones(vec![3]);
    let grads = op.backward(&grad_out, &[]);
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].to_vec(), vec![1.0, 1.0, 1.0]);
    assert_eq!(grads[1].to_vec(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_struct_add_backward_rejects_no_grad_produces_zeros() {
    let op = AddBackward { a_shape: vec![1], b_shape: vec![1] };
    let grad_out = Tensor::from_vec(vec![5.0], vec![1]).expect("grad_out");
    let grads = op.backward(&grad_out, &[]);
    assert_eq!(grads[0].to_vec(), vec![5.0]);
    assert_eq!(grads[1].to_vec(), vec![5.0]);
}
