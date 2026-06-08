//! Tests for ReLUBackward (src/api/gradient/types/re_l_u_backward.rs).
//!
//! @covers: ReLUBackward

use mlautograd::{BackwardOp, ReLUBackward, Tensor};

#[test]
fn test_struct_re_l_u_backward_zeros_grad_for_non_positive_inputs() {
    let op = ReLUBackward;
    let input = Tensor::from_vec(vec![-2.0, 0.0, 1.0, 3.0], vec![4]).expect("input");
    let grad_out = Tensor::ones(vec![4]);
    let grads = op.backward(&grad_out, &[input]);
    assert_eq!(grads[0].to_vec(), vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_struct_re_l_u_backward_scales_positive_inputs_by_grad_out() {
    let op = ReLUBackward;
    let input = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("input");
    let grad_out = Tensor::from_vec(vec![3.0, 4.0], vec![2]).expect("grad");
    let grads = op.backward(&grad_out, &[input]);
    assert_eq!(grads[0].to_vec(), vec![3.0, 4.0]);
}
