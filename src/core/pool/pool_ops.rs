use crate::api::traits::tensor_pool_op::TensorPoolOp;
use crate::core::pool::tensor_pool::TensorPool;
use std::cell::RefCell;

pub(crate) struct PoolOps;

thread_local! {
    static POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

impl PoolOps {
    pub(crate) fn acquire_inner(size: usize) -> Vec<f32> {
        POOL.with(|pool| pool.borrow_mut().acquire(size))
    }

    pub(crate) fn release_inner(buf: Vec<f32>) {
        POOL.with(|pool| pool.borrow_mut().release(buf));
    }

    pub(crate) fn clear_pool_inner() {
        POOL.with(|pool| pool.borrow_mut().clear());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: acquire_inner
    #[test]
    fn test_pool_ops_acquire_inner_returns_zeroed_buffer() {
        let buf = PoolOps::acquire_inner(10);
        assert_eq!(buf.len(), 10);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    /// @covers: release_inner
    #[test]
    fn test_pool_ops_release_inner_and_reacquire_reuses_allocation() {
        let buf = PoolOps::acquire_inner(8);
        let ptr = buf.as_ptr();
        PoolOps::release_inner(buf);
        let buf2 = PoolOps::acquire_inner(8);
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.len(), 8);
        assert!(buf2.iter().all(|&v| v == 0.0));
    }

    /// @covers: clear_pool_inner
    #[test]
    fn test_pool_ops_clear_pool_inner_drains_all_bins() {
        PoolOps::release_inner(PoolOps::acquire_inner(16));
        PoolOps::release_inner(PoolOps::acquire_inner(64));
        PoolOps::clear_pool_inner();
        let buf = PoolOps::acquire_inner(16);
        assert_eq!(buf.len(), 16);
    }

    /// @covers: acquire_inner
    #[test]
    fn test_pool_ops_acquire_inner_zero_size_returns_empty() {
        let buf = PoolOps::acquire_inner(0);
        assert!(buf.is_empty());
    }
}
