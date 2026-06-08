//! Tests for TapeContext (src/api/gradient/types/tape_context.rs).
//!
//! @covers: TapeContext

use mlautograd::{TapeContext, Tensor};

#[test]
fn test_struct_tape_context_is_recording_true_on_start() {
    TapeContext::clear_tape();
    assert!(TapeContext::is_recording());
}

#[test]
fn test_struct_tape_context_clear_tape_removes_all_state() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    TapeContext::set_grad(&t, Tensor::ones(vec![1]));
    TapeContext::clear_tape();
    assert!(TapeContext::grad(&t).is_none());
}
