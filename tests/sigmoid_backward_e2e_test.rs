//! Tests for SigmoidBackward (src/api/gradient/types/sigmoid_backward.rs).
//!
//! @covers: SigmoidBackward

use mlautograd::{BackwardOp, SigmoidBackward, Tensor};

#[test]
fn test_struct_sigmoid_backward_at_half_gives_quarter_grad() {
    let op = SigmoidBackward;
    let sig_out = Tensor::from_vec(vec![0.5], vec![1]).expect("sig");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[sig_out]);
    let val = grads[0].to_vec()[0];
    assert!((val - 0.25).abs() < 1e-5, "sigmoid grad at 0.5 is s*(1-s) = 0.25");
}

#[test]
fn test_struct_sigmoid_backward_near_zero_output_gives_near_zero_grad() {
    let op = SigmoidBackward;
    // sigmoid near 0 → s*(1-s) ≈ 0*1 = 0
    let sig_out = Tensor::from_vec(vec![0.01], vec![1]).expect("sig");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[sig_out]);
    let val = grads[0].to_vec()[0];
    assert!(val < 0.1, "grad near zero output should be small");
}
