//! E2E tests for the gradient tape operations contract.
//!
//! Exercises the `TapeOps` interface through the public TapeContext API.
//!
//! @covers: TapeOps

use mlautograd::{AddBackward, TapeContext, TapeEntry, Tensor};

#[test]
fn test_type_tape_ops_record_and_backward_propagate_grad() {
    TapeContext::clear_tape();
    let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("a");
    let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).expect("b");
    let c_data: Vec<f32> = a.to_vec().iter().zip(b.to_vec().iter()).map(|(x, y)| x + y).collect();
    let c = Tensor::from_vec(c_data, vec![2]).expect("c");
    TapeContext::record_op(TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id(), b.id()],
        backward_op: Box::new(AddBackward { a_shape: vec![2], b_shape: vec![2] }),
        saved_tensors: vec![],
    });
    TapeContext::backward(&c);
    let grad_a = TapeContext::grad(&a).expect("grad_a");
    assert_eq!(grad_a.to_vec(), vec![1.0, 1.0]);
}

#[test]
fn test_type_tape_ops_set_grad_roundtrips_via_grad() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![5.0], vec![1]).expect("t");
    let g = Tensor::ones(vec![1]);
    TapeContext::set_grad(&t, g);
    let result = TapeContext::grad(&t).expect("grad");
    assert_eq!(result.to_vec(), vec![1.0]);
}

#[test]
fn test_type_tape_ops_no_grad_suppresses_is_recording() {
    TapeContext::clear_tape();
    let inner = TapeContext::no_grad(|| TapeContext::is_recording());
    assert!(!inner);
    assert!(TapeContext::is_recording());
}

#[test]
fn test_type_tape_ops_clear_tape_wipes_all_state() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    TapeContext::set_grad(&t, Tensor::ones(vec![1]));
    TapeContext::clear_tape();
    assert!(TapeContext::grad(&t).is_none());
}
