//! Tests for the BackwardOp trait (src/api/traits/backward_op.rs).
//!
//! @covers: BackwardOp

use mlautograd::{AddBackward, BackwardOp, Tensor};

#[test]
fn test_struct_backward_op_add_backward_satisfies_trait() {
    let op: Box<dyn BackwardOp> = Box::new(AddBackward {
        a_shape: vec![2],
        b_shape: vec![2],
    });
    let grad = Tensor::ones(vec![2]);
    let grads = op.backward(&grad, &[]);
    assert_eq!(grads.len(), 2, "AddBackward must produce two gradients");
}

#[test]
fn test_struct_backward_op_send_sync_constraint_satisfied() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AddBackward>();
}
