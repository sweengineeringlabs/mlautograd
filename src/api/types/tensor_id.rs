use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) u64);

impl TensorId {
    pub(crate) fn next() -> Self {
        TensorId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorId({})", self.0)
    }
}
