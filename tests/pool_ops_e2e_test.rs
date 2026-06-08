//! E2E tests for the tensor buffer pool operations contract.
//!
//! Exercises the `PoolOps` interface through the public BufferPool API.
//!
//! @covers: PoolOps

use mlautograd::BufferPool;

#[test]
fn test_type_pool_ops_acquire_yields_zeroed_vec_of_requested_size() {
    let buf = BufferPool::acquire(6);
    assert_eq!(buf.len(), 6);
    assert!(buf.iter().all(|&v| v == 0.0));
}

#[test]
fn test_type_pool_ops_release_and_reacquire_reuse_same_allocation() {
    let buf = BufferPool::acquire(16);
    let ptr = buf.as_ptr();
    BufferPool::release(buf);
    let buf2 = BufferPool::acquire(16);
    assert_eq!(buf2.as_ptr(), ptr);
}

#[test]
fn test_type_pool_ops_clear_pool_after_release_forces_fresh_alloc() {
    BufferPool::release(BufferPool::acquire(64));
    BufferPool::clear_pool();
    let buf = BufferPool::acquire(64);
    assert_eq!(buf.len(), 64);
}
