//! Tests for TensorId (src/api/types/tensor_id.rs).
//!
//! @covers: TensorId

use mlautograd::Tensor;

#[test]
fn test_struct_tensor_id_is_unique_across_tensors() {
    let a = Tensor::zeros(vec![1]);
    let b = Tensor::zeros(vec![1]);
    assert_ne!(a.id(), b.id(), "each tensor must have a distinct id");
}

#[test]
fn test_struct_tensor_id_equality_is_reflexive() {
    let t = Tensor::zeros(vec![1]);
    assert_eq!(t.id(), t.id(), "the same tensor's id must equal itself");
}

#[test]
fn test_struct_tensor_id_display_contains_tensor_id_prefix() {
    let t = Tensor::zeros(vec![1]);
    let s = format!("{}", t.id());
    assert!(s.contains("TensorId"), "display must include TensorId prefix");
}

#[test]
fn test_struct_tensor_id_copy_semantic_preserved() {
    let t = Tensor::zeros(vec![1]);
    let id = t.id();
    let id2 = id; // Copy
    assert_eq!(id, id2);
}
