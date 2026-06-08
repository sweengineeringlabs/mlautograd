use crate::api::traits::gradient_tape_op::GradientTapeOp;
use crate::api::types::tape_entry::TapeEntry;
use crate::api::types::tensor::Tensor;
use crate::api::types::tensor_id::TensorId;
use crate::core::gradient::tape::default_gradient_tape::DefaultGradientTape;
use std::cell::RefCell;

pub(crate) struct TapeOps;

thread_local! {
    static TAPE: RefCell<DefaultGradientTape> = RefCell::new(DefaultGradientTape::new());
}

impl TapeOps {
    pub(crate) fn record_op_inner(entry: TapeEntry) {
        TAPE.with(|tape| tape.borrow_mut().record(entry));
    }

    pub(crate) fn backward_inner(loss_id: TensorId, loss_shape: &[usize]) {
        TAPE.with(|tape| {
            tape.borrow_mut().backward(loss_id, loss_shape);
        });
    }

    pub(crate) fn grad_inner(id: TensorId) -> Option<Tensor> {
        TAPE.with(|tape| tape.borrow().grad(id).cloned())
    }

    pub(crate) fn set_grad_inner(id: TensorId, grad: Tensor) {
        TAPE.with(|tape| {
            tape.borrow_mut().inner.grads.insert(id, grad);
        });
    }

    pub(crate) fn no_grad_inner<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let was_enabled = TAPE.with(|tape| {
            let mut t = tape.borrow_mut();
            let prev = t.is_enabled();
            t.disable();
            prev
        });
        let result = f();
        if was_enabled {
            TAPE.with(|tape| tape.borrow_mut().enable());
        }
        result
    }

    pub(crate) fn clear_tape_inner() {
        TAPE.with(|tape| tape.borrow_mut().clear());
    }

    pub(crate) fn is_recording_inner() -> bool {
        TAPE.with(|tape| tape.borrow().is_enabled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: record_op_inner
    #[test]
    fn test_tape_ops_record_op_inner_stores_entry() {
        use crate::api::gradient::types::add_backward::AddBackward;
        use crate::api::types::tape_entry::TapeEntry;
        use crate::api::types::tensor::Tensor;
        TapeOps::clear_tape_inner();
        let t = Tensor::zeros(vec![2]);
        let entry = TapeEntry {
            backward_op: Box::new(AddBackward { a_shape: vec![2], b_shape: vec![2] }),
            output_id: t.id(),
            input_ids: vec![],
            saved_tensors: vec![],
        };
        TapeOps::record_op_inner(entry);
        TapeOps::backward_inner(t.id(), t.shape());
    }

    /// @covers: is_recording_inner
    #[test]
    fn test_tape_ops_is_recording_inner_true_by_default() {
        TapeOps::clear_tape_inner();
        assert!(TapeOps::is_recording_inner());
    }

    /// @covers: no_grad_inner
    #[test]
    fn test_tape_ops_no_grad_inner_disables_during_closure() {
        TapeOps::clear_tape_inner();
        let recording_inside = TapeOps::no_grad_inner(|| TapeOps::is_recording_inner());
        assert!(!recording_inside);
        assert!(TapeOps::is_recording_inner());
    }

    /// @covers: set_grad_inner
    #[test]
    fn test_tape_ops_set_grad_inner_stores_gradient() {
        TapeOps::clear_tape_inner();
        let t = Tensor::zeros(vec![2]);
        let g = Tensor::ones(vec![2]);
        TapeOps::set_grad_inner(t.id(), g);
        let result = TapeOps::grad_inner(t.id());
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_vec(), vec![1.0, 1.0]);
    }

    /// @covers: grad_inner
    #[test]
    fn test_tape_ops_grad_inner_returns_none_when_not_set() {
        TapeOps::clear_tape_inner();
        let t = Tensor::zeros(vec![2]);
        assert!(TapeOps::grad_inner(t.id()).is_none());
    }

    /// @covers: backward_inner
    #[test]
    fn test_tape_ops_backward_inner_runs_without_entries() {
        TapeOps::clear_tape_inner();
        let loss = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        TapeOps::backward_inner(loss.id(), loss.shape());
    }

    /// @covers: clear_tape_inner
    #[test]
    fn test_tape_ops_clear_tape_inner_removes_grads() {
        TapeOps::clear_tape_inner();
        let t = Tensor::zeros(vec![2]);
        TapeOps::set_grad_inner(t.id(), Tensor::ones(vec![2]));
        TapeOps::clear_tape_inner();
        assert!(TapeOps::grad_inner(t.id()).is_none());
    }
}
