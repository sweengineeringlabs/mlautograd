//! Tests for the TensorPoolOp trait (src/api/traits/tensor_pool_op.rs).
//!
//! @covers: TensorPoolOp

use mlautograd::BufferPool;

#[test]
fn test_trait_tensor_pool_op_acquire_returns_zeroed_buffer() {
    let buf = BufferPool::acquire(4);
    assert_eq!(buf.len(), 4);
    assert!(buf.iter().all(|&v| v == 0.0), "acquired buffer must be zeroed");
}

#[test]
fn test_trait_tensor_pool_op_release_and_reacquire_reuses_memory() {
    let buf = BufferPool::acquire(8);
    let ptr = buf.as_ptr();
    BufferPool::release(buf);
    let buf2 = BufferPool::acquire(8);
    assert_eq!(buf2.len(), 8);
    assert_eq!(buf2.as_ptr(), ptr, "reacquired buffer must reuse the released allocation");
}

#[test]
fn test_trait_tensor_pool_op_clear_drains_pool() {
    BufferPool::release(BufferPool::acquire(16));
    BufferPool::clear_pool();
    // After clearing, a new acquire should succeed (no panic)
    let buf = BufferPool::acquire(16);
    assert_eq!(buf.len(), 16);
}
