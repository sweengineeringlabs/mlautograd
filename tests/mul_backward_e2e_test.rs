//! Tests for MulBackward (src/api/gradient/types/mul_backward.rs).
//!
//! @covers: MulBackward

use mlautograd::{BackwardOp, MulBackward, Tensor};

#[test]
fn test_struct_mul_backward_grad_a_equals_grad_out_times_b() {
    let op = MulBackward;
    let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]).expect("a");
    let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]).expect("b");
    let grad_out = Tensor::ones(vec![2]);
    let grads = op.backward(&grad_out, &[a, b]);
    assert_eq!(grads[0].to_vec(), vec![4.0, 5.0], "grad_a = b");
    assert_eq!(grads[1].to_vec(), vec![2.0, 3.0], "grad_b = a");
}

#[test]
fn test_struct_mul_backward_scaled_grad_out() {
    let op = MulBackward;
    let a = Tensor::from_vec(vec![1.0], vec![1]).expect("a");
    let b = Tensor::from_vec(vec![3.0], vec![1]).expect("b");
    let grad_out = Tensor::from_vec(vec![2.0], vec![1]).expect("grad");
    let grads = op.backward(&grad_out, &[a, b]);
    assert_eq!(grads[0].to_vec(), vec![6.0], "grad_a = grad_out * b = 2 * 3");
    assert_eq!(grads[1].to_vec(), vec![2.0], "grad_b = grad_out * a = 2 * 1");
}
