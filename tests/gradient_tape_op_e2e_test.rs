//! Tests for the GradientTapeOp trait (src/api/traits/gradient_tape_op.rs).
//!
//! @covers: GradientTapeOp

use mlautograd::{AddBackward, TapeContext, TapeEntry, Tensor};

#[test]
fn test_trait_gradient_tape_op_is_recording_true_by_default() {
    TapeContext::clear_tape();
    assert!(TapeContext::is_recording(), "tape must be enabled by default");
}

#[test]
fn test_trait_gradient_tape_op_no_grad_disables_during_closure() {
    TapeContext::clear_tape();
    let recording_inside = TapeContext::no_grad(|| TapeContext::is_recording());
    assert!(!recording_inside, "tape must be disabled inside no_grad");
    assert!(TapeContext::is_recording(), "tape must be re-enabled after no_grad");
}

#[test]
fn test_trait_gradient_tape_op_backward_seeds_loss_gradient() {
    TapeContext::clear_tape();
    let loss = Tensor::from_vec(vec![1.0], vec![1]).expect("loss tensor");
    TapeContext::backward(&loss);
    // backward does not panic when no ops are recorded
}

#[test]
fn test_trait_gradient_tape_op_record_then_backward_propagates_grad() {
    TapeContext::clear_tape();
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("a");
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).expect("b");
    let c = Tensor::from_vec(vec![4.0, 6.0], vec![2]).expect("c");
    let entry = TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id(), b.id()],
        backward_op: Box::new(AddBackward { a_shape: vec![2], b_shape: vec![2] }),
        saved_tensors: vec![],
    };
    TapeContext::record_op(entry);
    TapeContext::backward(&c);
    assert!(TapeContext::grad(&a).is_some(), "grad must exist for a after backward");
    assert!(TapeContext::grad(&b).is_some(), "grad must exist for b after backward");
}
