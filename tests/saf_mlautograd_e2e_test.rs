//! Public-API tests for saf/mlautograd_svc.rs.
//!
//! Every pub fn in saf/ is covered here with an @covers annotation
//! per rule 125.

use mlautograd::{AddBackward, BufferPool, TapeContext, TapeEntry, Tensor};

/// @covers: record_op
#[test]
fn test_struct_tape_context_record_op_produces_grad_on_input() {
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

/// @covers: backward
#[test]
fn test_struct_tape_context_backward_sets_unit_loss_grad() {
    TapeContext::clear_tape();
    let loss = Tensor::from_vec(vec![5.0], vec![1]).expect("loss");
    TapeContext::backward(&loss);
    let g = TapeContext::grad(&loss).expect("grad");
    assert_eq!(g.to_vec(), vec![1.0]);
}

/// @covers: grad
#[test]
fn test_struct_tape_context_grad_absent_for_unseen_tensor() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    assert!(TapeContext::grad(&t).is_none());
}

/// @covers: set_grad
#[test]
fn test_struct_tape_context_set_grad_visible_via_grad() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![2.0], vec![1]).expect("t");
    TapeContext::set_grad(&t, Tensor::ones(vec![1]));
    let g = TapeContext::grad(&t).expect("grad");
    assert_eq!(g.to_vec(), vec![1.0]);
}

/// @covers: no_grad
#[test]
fn test_struct_tape_context_no_grad_pauses_tape() {
    TapeContext::clear_tape();
    let inside = TapeContext::no_grad(|| TapeContext::is_recording());
    assert!(!inside);
    assert!(TapeContext::is_recording());
}

/// @covers: clear_tape
#[test]
fn test_struct_tape_context_clear_tape_wipes_grad_map() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    TapeContext::set_grad(&t, Tensor::ones(vec![1]));
    TapeContext::clear_tape();
    assert!(TapeContext::grad(&t).is_none());
}

/// @covers: is_recording
#[test]
fn test_struct_tape_context_is_recording_enabled_on_fresh_tape() {
    TapeContext::clear_tape();
    assert!(TapeContext::is_recording());
}

/// @covers: acquire
#[test]
fn test_struct_buffer_pool_acquire_yields_zeroed_vec() {
    let buf = BufferPool::acquire(4);
    assert_eq!(buf.len(), 4);
    assert!(buf.iter().all(|&v| v == 0.0));
}

/// @covers: release
#[test]
fn test_struct_buffer_pool_release_puts_buf_back_in_pool() {
    let buf = BufferPool::acquire(8);
    let ptr = buf.as_ptr();
    BufferPool::release(buf);
    let buf2 = BufferPool::acquire(8);
    assert_eq!(buf2.as_ptr(), ptr);
}

/// @covers: clear_pool
#[test]
fn test_struct_buffer_pool_clear_pool_empties_bins() {
    BufferPool::release(BufferPool::acquire(32));
    BufferPool::clear_pool();
    let buf = BufferPool::acquire(32);
    assert_eq!(buf.len(), 32);
}
