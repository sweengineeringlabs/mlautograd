use crate::api::traits::gradient_tape_op::GradientTapeOp;
use crate::api::types::tape_entry::TapeEntry;
use crate::api::types::tensor::Tensor;
use crate::api::types::tensor_id::TensorId;
use crate::core::gradient::tape::gradient_tape_inner::GradientTapeInner;

pub(crate) struct DefaultGradientTape {
    pub(crate) inner: GradientTapeInner,
}

impl DefaultGradientTape {
    pub(crate) fn new() -> Self {
        Self {
            inner: GradientTapeInner::new(),
        }
    }
}

impl GradientTapeOp for DefaultGradientTape {
    fn record(&mut self, entry: TapeEntry) {
        if self.inner.enabled {
            self.inner.entries.push(entry);
        }
    }

    fn backward(&mut self, loss_id: TensorId, loss_shape: &[usize]) {
        let seed = Tensor::ones(loss_shape.to_vec());
        self.inner.grads.insert(loss_id, seed);

        for i in (0..self.inner.entries.len()).rev() {
            let output_id = self.inner.entries[i].output_id;
            let grad_output = match self.inner.grads.get(&output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let input_grads = self.inner.entries[i]
                .backward_op
                .backward(&grad_output, &self.inner.entries[i].saved_tensors);

            for (j, input_id) in self.inner.entries[i].input_ids.iter().enumerate() {
                if j < input_grads.len() {
                    let new_grad = &input_grads[j];
                    if let Some(existing) = self.inner.grads.get(input_id) {
                        let accumulated = existing.add_raw(new_grad).expect("gradient accumulation");
                        self.inner.grads.insert(*input_id, accumulated);
                    } else {
                        self.inner.grads.insert(*input_id, new_grad.clone());
                    }
                }
            }
        }
    }

    fn grad(&self, id: TensorId) -> Option<&Tensor> {
        self.inner.grads.get(&id)
    }

    fn clear(&mut self) {
        self.inner.entries.clear();
        self.inner.grads.clear();
    }

    fn enable(&mut self) {
        self.inner.enabled = true;
    }

    fn disable(&mut self) {
        self.inner.enabled = false;
    }

    fn is_enabled(&self) -> bool {
        self.inner.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// @covers: new
    #[test]
    fn test_default_gradient_tape_new_starts_enabled() {
        let tape = DefaultGradientTape::new();
        assert!(tape.is_enabled());
    }

    /// @covers: disable
    #[test]
    fn test_default_gradient_tape_disable_stops_recording() {
        let mut tape = DefaultGradientTape::new();
        tape.disable();
        assert!(!tape.is_enabled());
    }

    /// @covers: enable
    #[test]
    fn test_default_gradient_tape_enable_resumes_recording() {
        let mut tape = DefaultGradientTape::new();
        tape.disable();
        tape.enable();
        assert!(tape.is_enabled());
    }

    /// @covers: is_enabled
    #[test]
    fn test_default_gradient_tape_is_enabled_reflects_toggle() {
        let mut tape = DefaultGradientTape::new();
        assert!(tape.is_enabled());
        tape.disable();
        assert!(!tape.is_enabled());
    }

    /// @covers: backward
    #[test]
    fn test_default_gradient_tape_backward_seeds_loss_gradient() {
        let mut tape = DefaultGradientTape::new();
        let loss = Tensor::from_vec(vec![5.0], vec![1]).unwrap();
        tape.backward(loss.id(), loss.shape());
        let g = tape.grad(loss.id());
        assert!(g.is_some());
        assert_eq!(g.unwrap().to_vec(), vec![1.0]);
    }

    /// @covers: clear
    #[test]
    fn test_default_gradient_tape_clear_removes_grads() {
        let mut tape = DefaultGradientTape::new();
        let loss = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        tape.backward(loss.id(), loss.shape());
        tape.clear();
        assert!(tape.grad(loss.id()).is_none());
    }

    /// @covers: record
    #[test]
    fn test_default_gradient_tape_record_ignores_when_disabled() {
        use crate::api::gradient::types::add_backward::AddBackward;
        use crate::api::types::tape_entry::TapeEntry;
        let mut tape = DefaultGradientTape::new();
        tape.disable();
        let t = Tensor::zeros(vec![2]);
        let entry = TapeEntry {
            backward_op: Box::new(AddBackward { a_shape: vec![2], b_shape: vec![2] }),
            output_id: t.id(),
            input_ids: vec![],
            saved_tensors: vec![],
        };
        tape.record(entry);
        assert!(tape.inner.entries.is_empty());
    }

    /// @covers: grad
    #[test]
    fn test_default_gradient_tape_grad_returns_none_when_not_set() {
        let tape = DefaultGradientTape::new();
        let t = Tensor::zeros(vec![2]);
        assert!(tape.grad(t.id()).is_none());
    }
}
