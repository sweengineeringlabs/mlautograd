//! Integration tests for the `swe-ml-tensor` dependency.
//!
//! Verifies that tensor allocation, arithmetic, and gradient operations
//! exercised by mlautograd work correctly end-to-end with swe-ml-tensor.

use mlautograd::{AddBackward, TapeContext, TapeEntry, Tensor};
use swe_ml_tensor::Shape;

#[test]
fn test_swe_ml_tensor_shape_from_vec_matches_zeros_shape() {
    let shape = Shape::new(vec![2, 3]);
    let t = Tensor::zeros(shape);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.numel(), 6);
}

#[test]
fn test_swe_ml_tensor_zeros_allocates_correct_shape() {
    let t = Tensor::zeros(vec![2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert!(t.to_vec().iter().all(|&v| v == 0.0));
}

#[test]
fn test_swe_ml_tensor_ones_fills_with_one() {
    let t = Tensor::ones(vec![3]);
    assert_eq!(t.to_vec(), vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_swe_ml_tensor_from_vec_roundtrip() {
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let t = Tensor::from_vec(data.clone(), vec![4]).expect("from_vec");
    assert_eq!(t.to_vec(), data);
}

#[test]
fn test_swe_ml_tensor_backward_pass_produces_correct_gradients() {
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
    let grad = TapeContext::grad(&a).expect("grad_a");
    assert_eq!(grad.to_vec(), vec![1.0, 1.0]);
}
