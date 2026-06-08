//! Tests for MatMulBackward (src/api/gradient/types/mat_mul_backward.rs).
//!
//! @covers: MatMulBackward

use mlautograd::{BackwardOp, MatMulBackward, Tensor};

#[test]
fn test_struct_mat_mul_backward_produces_correct_grad_shapes() {
    let op = MatMulBackward;
    let a = Tensor::ones(vec![2, 3]);
    let b = Tensor::ones(vec![3, 4]);
    let grad_out = Tensor::ones(vec![2, 4]);
    let grads = op.backward(&grad_out, &[a, b]);
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 3], "grad_a shape must match a");
    assert_eq!(grads[1].shape(), &[3, 4], "grad_b shape must match b");
}

#[test]
fn test_struct_mat_mul_backward_identity_matrix_preserves_grad() {
    let op = MatMulBackward;
    // a = [1, 0; 0, 1], b = [1, 0; 0, 1], grad_out = [[1, 2],[3, 4]]
    let a = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("a");
    let b = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("b");
    let grad_out = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).expect("grad");
    let grads = op.backward(&grad_out, &[a, b]);
    // grad_a = grad_out @ b^T = grad_out @ I = grad_out
    assert_eq!(grads[0].to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}
