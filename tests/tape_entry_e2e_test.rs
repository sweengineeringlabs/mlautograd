//! Tests for TapeEntry (src/api/types/tape_entry.rs).
//!
//! @covers: TapeEntry

use mlautograd::{AddBackward, TapeEntry, Tensor};

#[test]
fn test_struct_tape_entry_stores_backward_op_and_ids() {
    let a = Tensor::from_vec(vec![1.0], vec![1]).expect("a");
    let c = Tensor::from_vec(vec![2.0], vec![1]).expect("c");
    let entry = TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id()],
        backward_op: Box::new(AddBackward { a_shape: vec![1], b_shape: vec![1] }),
        saved_tensors: vec![],
    };
    assert_eq!(entry.output_id, c.id());
    assert_eq!(entry.input_ids.len(), 1);
    assert_eq!(entry.saved_tensors.len(), 0);
}

#[test]
fn test_struct_tape_entry_saved_tensors_are_accessible() {
    let saved = Tensor::ones(vec![2]);
    let shape = saved.shape().to_vec();
    let entry = TapeEntry {
        output_id: Tensor::zeros(vec![1]).id(),
        input_ids: vec![],
        backward_op: Box::new(AddBackward { a_shape: vec![2], b_shape: vec![2] }),
        saved_tensors: vec![saved],
    };
    assert_eq!(entry.saved_tensors.len(), 1);
    assert_eq!(entry.saved_tensors[0].shape(), shape.as_slice());
}
