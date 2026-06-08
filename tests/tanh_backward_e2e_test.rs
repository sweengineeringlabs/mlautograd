//! Tests for TanhBackward (src/api/gradient/types/tanh_backward.rs).
//!
//! @covers: TanhBackward

use mlautograd::{BackwardOp, TanhBackward, Tensor};

#[test]
fn test_struct_tanh_backward_at_zero_output_gives_unit_grad() {
    let op = TanhBackward;
    let tanh_out = Tensor::from_vec(vec![0.0], vec![1]).expect("tanh");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[tanh_out]);
    let val = grads[0].to_vec()[0];
    assert!((val - 1.0).abs() < 1e-5, "1 - tanh(0)^2 = 1");
}

#[test]
fn test_struct_tanh_backward_at_one_output_gives_near_zero_grad() {
    let op = TanhBackward;
    // tanh^2 ≈ 1 when output is near 1 → grad ≈ 0
    let tanh_out = Tensor::from_vec(vec![0.999], vec![1]).expect("tanh");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[tanh_out]);
    let val = grads[0].to_vec()[0];
    assert!(val < 0.01, "grad near saturated tanh should be near zero");
}
