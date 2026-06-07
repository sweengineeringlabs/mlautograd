use std::cell::RefCell;
use std::collections::HashMap;

struct TensorPool {
    bins: HashMap<usize, Vec<Vec<f32>>>,
}

impl TensorPool {
    fn new() -> Self {
        Self {
            bins: HashMap::new(),
        }
    }

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

thread_local! {
    static POOL: RefCell<TensorPool> = RefCell::new(TensorPool::new());
}

pub fn acquire(size: usize) -> Vec<f32> {
    POOL.with(|pool| pool.borrow_mut().acquire(size))
}

pub fn release(buf: Vec<f32>) {
    POOL.with(|pool| pool.borrow_mut().release(buf));
}

pub fn clear_pool() {
    POOL.with(|pool| pool.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_returns_zeroed_buffer() {
        let buf = acquire(10);
        assert_eq!(buf.len(), 10);
        assert!(buf.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn release_and_reacquire_reuses_allocation() {
        let buf = acquire(8);
        let ptr = buf.as_ptr();
        release(buf);
        let buf2 = acquire(8);
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.len(), 8);
        assert!(buf2.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn clear_pool_drains_all_bins() {
        release(acquire(16));
        release(acquire(64));
        clear_pool();
        let buf = acquire(16);
        assert_eq!(buf.len(), 16);
    }

    #[test]
    fn acquire_zero_size() {
        let buf = acquire(0);
        assert!(buf.is_empty());
    }
}
