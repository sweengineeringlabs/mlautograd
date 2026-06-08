//! Tests for BufferPool (src/api/gradient/types/buffer_pool.rs).
//!
//! @covers: BufferPool

use mlautograd::BufferPool;

#[test]
fn test_struct_buffer_pool_acquire_produces_correct_length() {
    let buf = BufferPool::acquire(7);
    assert_eq!(buf.len(), 7);
}

#[test]
fn test_struct_buffer_pool_acquire_zero_size_returns_empty() {
    let buf = BufferPool::acquire(0);
    assert!(buf.is_empty());
}

#[test]
fn test_struct_buffer_pool_release_then_acquire_reuses_allocation() {
    let buf = BufferPool::acquire(10);
    let ptr = buf.as_ptr();
    BufferPool::release(buf);
    let buf2 = BufferPool::acquire(10);
    assert_eq!(buf2.as_ptr(), ptr);
}

#[test]
fn test_struct_buffer_pool_clear_pool_succeeds_without_panic() {
    BufferPool::release(BufferPool::acquire(8));
    BufferPool::clear_pool();
}
