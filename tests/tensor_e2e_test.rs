//! Tests for Tensor (src/api/types/tensor.rs).
//!
//! @covers: Tensor

use mlautograd::Tensor;

#[test]
fn test_struct_tensor_zeros_produces_correct_shape_and_values() {
    let t = Tensor::zeros(vec![2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert!(t.to_vec().iter().all(|&v| v == 0.0));
}

#[test]
fn test_struct_tensor_ones_produces_correct_shape_and_values() {
    let t = Tensor::ones(vec![3]);
    assert_eq!(t.shape(), &[3]);
    assert!(t.to_vec().iter().all(|&v| v == 1.0));
}

#[test]
fn test_struct_tensor_from_vec_roundtrips_data() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let t = Tensor::from_vec(data.clone(), vec![2, 2]).expect("from_vec");
    assert_eq!(t.to_vec(), data);
    assert_eq!(t.shape(), &[2, 2]);
}

#[test]
fn test_struct_tensor_numel_matches_product_of_shape() {
    let t = Tensor::zeros(vec![3, 4]);
    assert_eq!(t.numel(), 12);
}

#[test]
fn test_struct_tensor_id_is_unique_per_instance() {
    let a = Tensor::zeros(vec![1]);
    let b = Tensor::zeros(vec![1]);
    assert_ne!(a.id(), b.id(), "each tensor must receive a unique id");
}

#[test]
fn test_struct_tensor_unbroadcast_to_same_shape_is_identity() {
    let t = Tensor::ones(vec![2, 3]);
    let result = t.unbroadcast_to(&[2, 3]);
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result.to_vec(), t.to_vec());
}

#[test]
fn test_struct_tensor_unbroadcast_to_reduces_broadcast_dim() {
    // Tensor shape [1, 3] broadcast to [2, 3], then unbroadcast back to [1, 3]
    let t = Tensor::ones(vec![2, 3]);
    let result = t.unbroadcast_to(&[1, 3]);
    assert_eq!(result.shape(), &[1, 3]);
    // Each element should be sum over the broadcast dimension (2 ones = 2.0)
    assert_eq!(result.to_vec(), vec![2.0, 2.0, 2.0]);
}
