use crate::api::types::tape_entry::TapeEntry;
use crate::api::types::tensor::Tensor;
use crate::api::types::tensor_id::TensorId;
use std::collections::HashMap;

pub(crate) struct GradientTapeInner {
    pub(crate) entries: Vec<TapeEntry>,
    pub(crate) grads: HashMap<TensorId, Tensor>,
    pub(crate) enabled: bool,
}

impl GradientTapeInner {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            grads: HashMap::new(),
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: new
    #[test]
    fn test_gradient_tape_inner_new_is_enabled_with_empty_state() {
        let inner = GradientTapeInner::new();
        assert!(inner.enabled);
        assert!(inner.entries.is_empty());
        assert!(inner.grads.is_empty());
    }
}
