//! Tests for SoftmaxBackward (src/api/gradient/types/softmax_backward.rs).
//!
//! @covers: SoftmaxBackward

use mlautograd::{BackwardOp, SoftmaxBackward, Tensor};

#[test]
fn test_struct_softmax_backward_output_shape_matches_input() {
    let op = SoftmaxBackward { dim: -1 };
    let softmax_out = Tensor::from_vec(vec![0.25, 0.25, 0.25, 0.25], vec![1, 4]).expect("s");
    let grad_out = Tensor::ones(vec![1, 4]);
    let grads = op.backward(&grad_out, &[softmax_out]);
    assert_eq!(grads[0].shape(), &[1, 4]);
}

#[test]
fn test_struct_softmax_backward_uniform_distribution_has_zero_gradient() {
    let op = SoftmaxBackward { dim: -1 };
    // Uniform softmax with uniform grad → gradient is zero (sums cancel)
    let softmax_out = Tensor::from_vec(vec![0.5, 0.5], vec![1, 2]).expect("s");
    let grad_out = Tensor::ones(vec![1, 2]);
    let grads = op.backward(&grad_out, &[softmax_out]);
    let vals = grads[0].to_vec();
    assert!(vals.iter().all(|&v| v.abs() < 1e-5), "uniform grad must be zero");
}
