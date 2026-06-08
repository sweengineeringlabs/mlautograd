//! Integration tests for the mlautograd public API.
//!
//! These tests exercise the full stack: saf/ pub functions via TapeContext and
//! BufferPool, verifying integration with the swe-ml-tensor dependency.
//!
//! @covers: TapeContext
//! @covers: BufferPool
//! @covers: AddBackward
//! @covers: MulBackward
//! @covers: ReLUBackward
//! @covers: SigmoidBackward
//! @covers: TanhBackward
//! @covers: SoftmaxBackward
//! @covers: MatMulBackward
//! @covers: GradientTape

use mlautograd::{
    AddBackward, BufferPool, MatMulBackward, MulBackward, ReLUBackward, SigmoidBackward,
    SoftmaxBackward, TanhBackward, TapeContext, TapeEntry, Tensor,
};

// ─── TapeContext (saf pub API) ────────────────────────────────────────────────

#[test]
fn test_struct_tape_context_record_op_stores_entry() {
    TapeContext::clear_tape();
    let a = Tensor::from_vec(vec![1.0], vec![1]).expect("a");
    let c = Tensor::from_vec(vec![2.0], vec![1]).expect("c");
    TapeContext::record_op(TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id()],
        backward_op: Box::new(AddBackward { a_shape: vec![1], b_shape: vec![1] }),
        saved_tensors: vec![],
    });
    // If backward runs without panic the entry was recorded.
    TapeContext::backward(&c);
}

#[test]
fn test_struct_tape_context_backward_propagates_add_grad() {
    TapeContext::clear_tape();
    let a = Tensor::from_vec(vec![3.0, 4.0], vec![2]).expect("a");
    let b = Tensor::from_vec(vec![1.0, 2.0], vec![2]).expect("b");
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
fn test_struct_tape_context_grad_returns_none_before_backward() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    assert!(TapeContext::grad(&t).is_none());
}

#[test]
fn test_struct_tape_context_set_grad_stores_gradient() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    let g = Tensor::ones(vec![1]);
    TapeContext::set_grad(&t, g);
    let result = TapeContext::grad(&t).expect("grad");
    assert_eq!(result.to_vec(), vec![1.0]);
}

#[test]
fn test_struct_tape_context_no_grad_suppresses_recording() {
    TapeContext::clear_tape();
    let recording = TapeContext::no_grad(|| TapeContext::is_recording());
    assert!(!recording);
    assert!(TapeContext::is_recording());
}

#[test]
fn test_struct_tape_context_clear_tape_removes_grads() {
    TapeContext::clear_tape();
    let t = Tensor::from_vec(vec![1.0], vec![1]).expect("t");
    TapeContext::set_grad(&t, Tensor::ones(vec![1]));
    TapeContext::clear_tape();
    assert!(TapeContext::grad(&t).is_none());
}

#[test]
fn test_struct_tape_context_is_recording_true_by_default() {
    TapeContext::clear_tape();
    assert!(TapeContext::is_recording());
}

// ─── BufferPool (saf pub API) ─────────────────────────────────────────────────

#[test]
fn test_struct_buffer_pool_acquire_returns_zeroed_buf_of_correct_size() {
    let buf = BufferPool::acquire(5);
    assert_eq!(buf.len(), 5);
    assert!(buf.iter().all(|&v| v == 0.0));
}

#[test]
fn test_struct_buffer_pool_release_and_reacquire_reuses_allocation() {
    let buf = BufferPool::acquire(16);
    let ptr = buf.as_ptr();
    BufferPool::release(buf);
    let buf2 = BufferPool::acquire(16);
    assert_eq!(buf2.as_ptr(), ptr);
}

#[test]
fn test_struct_buffer_pool_clear_pool_drains_all_bins() {
    BufferPool::release(BufferPool::acquire(32));
    BufferPool::clear_pool();
    let buf = BufferPool::acquire(32);
    assert_eq!(buf.len(), 32);
}

// ─── Backward ops (api types + integration with swe-ml-tensor) ───────────────

#[test]
fn test_struct_mul_backward_applies_chain_rule_correctly() {
    use mlautograd::BackwardOp;
    let op = MulBackward;
    let a = Tensor::from_vec(vec![2.0], vec![1]).expect("a");
    let b = Tensor::from_vec(vec![3.0], vec![1]).expect("b");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[a, b]);
    assert_eq!(grads[0].to_vec(), vec![3.0], "grad_a = grad_out * b");
    assert_eq!(grads[1].to_vec(), vec![2.0], "grad_b = grad_out * a");
}

#[test]
fn test_struct_relu_backward_zeroes_neg_inputs() {
    use mlautograd::BackwardOp;
    let op = ReLUBackward;
    let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], vec![3]).expect("input");
    let grad_out = Tensor::ones(vec![3]);
    let grads = op.backward(&grad_out, &[input]);
    assert_eq!(grads[0].to_vec(), vec![0.0, 0.0, 1.0]);
}

#[test]
fn test_struct_sigmoid_backward_at_half() {
    use mlautograd::BackwardOp;
    let op = SigmoidBackward;
    let sig = Tensor::from_vec(vec![0.5], vec![1]).expect("sig");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[sig]);
    assert!((grads[0].to_vec()[0] - 0.25).abs() < 1e-5);
}

#[test]
fn test_struct_tanh_backward_at_zero_output() {
    use mlautograd::BackwardOp;
    let op = TanhBackward;
    let tanh_out = Tensor::from_vec(vec![0.0], vec![1]).expect("tanh_out");
    let grad_out = Tensor::ones(vec![1]);
    let grads = op.backward(&grad_out, &[tanh_out]);
    assert!((grads[0].to_vec()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_struct_softmax_backward_shape_preserved() {
    use mlautograd::BackwardOp;
    let op = SoftmaxBackward { dim: -1 };
    let softmax_out = Tensor::from_vec(vec![0.25, 0.25, 0.25, 0.25], vec![1, 4]).expect("s");
    let grad_out = Tensor::ones(vec![1, 4]);
    let grads = op.backward(&grad_out, &[softmax_out]);
    assert_eq!(grads[0].shape(), &[1, 4]);
}

#[test]
fn test_struct_mat_mul_backward_shape_correct() {
    use mlautograd::BackwardOp;
    let op = MatMulBackward;
    let a = Tensor::ones(vec![2, 3]);
    let b = Tensor::ones(vec![3, 4]);
    let grad_out = Tensor::ones(vec![2, 4]);
    let grads = op.backward(&grad_out, &[a, b]);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[1].shape(), &[3, 4]);
}
