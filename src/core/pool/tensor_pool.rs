use crate::api::traits::tensor_pool_op::TensorPoolOp;
use std::collections::HashMap;

pub(crate) struct TensorPool {
    bins: HashMap<usize, Vec<Vec<f32>>>,
}

impl TensorPool {
    pub(crate) fn new() -> Self {
        Self {
            bins: HashMap::new(),
        }
    }
}

impl TensorPoolOp for TensorPool {
    fn acquire(&mut self, size: usize) -> Vec<f32> {
        let bucket = size.next_power_of_two().max(1);
        if let Some(stack) = self.bins.get_mut(&bucket) {
            if let Some(mut buf) = stack.pop() {
                buf.clear();
                buf.resize(size, 0.0);
                return buf;
            }
        }
        vec![0.0; size]
    }

    fn release(&mut self, buf: Vec<f32>) {
        let bucket = buf.capacity().next_power_of_two().max(1);
        self.bins.entry(bucket).or_default().push(buf);
    }

    fn clear(&mut self) {
        self.bins.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: new
    #[test]
    fn test_tensor_pool_new_creates_empty_pool() {
        let pool = TensorPool::new();
        // A freshly created pool has no bins — first acquire allocates fresh.
        assert!(pool.bins.is_empty());
    }

    /// @covers: acquire
    #[test]
    fn test_tensor_pool_acquire_returns_zeroed_buffer() {
        let mut pool = TensorPool::new();
        let buf = pool.acquire(5);
        assert_eq!(buf.len(), 5);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    /// @covers: release
    #[test]
    fn test_tensor_pool_release_and_reacquire_reuses_allocation() {
        let mut pool = TensorPool::new();
        let buf = pool.acquire(8);
        let ptr = buf.as_ptr();
        pool.release(buf);
        let buf2 = pool.acquire(8);
        assert_eq!(buf2.as_ptr(), ptr);
    }

    /// @covers: clear
    #[test]
    fn test_tensor_pool_clear_drains_all_bins() {
        let mut pool = TensorPool::new();
        let buf = pool.acquire(16);
        pool.release(buf);
        pool.clear();
        let buf = pool.acquire(16);
        assert_eq!(buf.len(), 16);
    }
}
