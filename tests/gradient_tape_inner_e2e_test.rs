//! E2E tests for the gradient tape inner state contract.
//!
//! Exercises the internal tape state through the public TapeContext API,
//! covering the `GradientTapeInner` type and its `core/` implementation.
//!
//! @covers: GradientTapeInner

use mlautograd::{AddBackward, TapeContext, TapeEntry, Tensor};

#[test]
fn test_type_gradient_tape_inner_entry_stored_and_backward_runs() {
    TapeContext::clear_tape();
    let a = Tensor::from_vec(vec![1.0], vec![1]).expect("a");
    let c = Tensor::from_vec(vec![2.0], vec![1]).expect("c");
    TapeContext::record_op(TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id()],
        backward_op: Box::new(AddBackward { a_shape: vec![1], b_shape: vec![1] }),
        saved_tensors: vec![],
    });
    TapeContext::backward(&c);
    assert!(TapeContext::grad(&a).is_some());
}

#[test]
fn test_type_gradient_tape_inner_grad_map_filled_after_backward() {
    TapeContext::clear_tape();
    let a = Tensor::from_vec(vec![2.0], vec![1]).expect("a");
    let b = Tensor::from_vec(vec![3.0], vec![1]).expect("b");
    let c_val = a.to_vec()[0] + b.to_vec()[0];
    let c = Tensor::from_vec(vec![c_val], vec![1]).expect("c");
    TapeContext::record_op(TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id(), b.id()],
        backward_op: Box::new(AddBackward { a_shape: vec![1], b_shape: vec![1] }),
        saved_tensors: vec![],
    });
    TapeContext::backward(&c);
    assert!(TapeContext::grad(&a).is_some());
    assert!(TapeContext::grad(&b).is_some());
}

#[test]
fn test_type_gradient_tape_inner_enabled_restored_after_no_grad() {
    TapeContext::clear_tape();
    assert!(TapeContext::is_recording());
    let was_recording = TapeContext::no_grad(|| TapeContext::is_recording());
    assert!(!was_recording, "inner.enabled must be false inside no_grad");
    assert!(TapeContext::is_recording(), "inner.enabled must be restored");
}
