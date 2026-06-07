use crate::tape::tape_entry::TapeEntry;
use crate::tensor::{Tensor, TensorId};
use std::cell::RefCell;
use std::collections::HashMap;

pub struct GradientTape {
    entries: Vec<TapeEntry>,
    pub(crate) grads: HashMap<TensorId, Tensor>,
    enabled: bool,
}

impl GradientTape {
    pub(crate) fn new() -> Self {
        Self {
            entries: Vec::new(),
            grads: HashMap::new(),
            enabled: true,
        }
    }

    pub fn record(&mut self, entry: TapeEntry) {
        if self.enabled {
            self.entries.push(entry);
        }
    }

    pub fn backward(&mut self, loss_id: TensorId, loss_shape: &[usize]) {
        let seed = Tensor::ones(loss_shape.to_vec());
        self.grads.insert(loss_id, seed);

        for i in (0..self.entries.len()).rev() {
            let output_id = self.entries[i].output_id;
            let grad_output = match self.grads.get(&output_id) {
                Some(g) => g.clone(),
                None => continue,
            };

            let input_grads = self.entries[i]
                .backward_op
                .backward(&grad_output, &self.entries[i].saved_tensors);

            for (j, input_id) in self.entries[i].input_ids.iter().enumerate() {
                if j < input_grads.len() {
                    let new_grad = &input_grads[j];
                    if let Some(existing) = self.grads.get(input_id) {
                        let accumulated = existing.add_raw(new_grad).expect("gradient accumulation");
                        self.grads.insert(*input_id, accumulated);
                    } else {
                        self.grads.insert(*input_id, new_grad.clone());
                    }
                }
            }
        }
    }

    pub fn grad(&self, id: TensorId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.grads.clear();
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

thread_local! {
    static TAPE: RefCell<GradientTape> = RefCell::new(GradientTape::new());
}

pub fn record_op(entry: TapeEntry) {
    TAPE.with(|tape| tape.borrow_mut().record(entry));
}

pub fn backward(loss: &Tensor) {
    TAPE.with(|tape| {
        tape.borrow_mut().backward(loss.id(), loss.shape());
    });
}

pub fn grad(tensor: &Tensor) -> Option<Tensor> {
    TAPE.with(|tape| tape.borrow().grad(tensor.id()).cloned())
}

pub fn set_grad(tensor: &Tensor, grad: Tensor) {
    TAPE.with(|tape| {
        tape.borrow_mut().grads.insert(tensor.id(), grad);
    });
}

pub fn no_grad<F, R>(f: F) -> R
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

pub fn clear_tape() {
    TAPE.with(|tape| tape.borrow_mut().clear());
}

pub fn is_recording() -> bool {
    TAPE.with(|tape| tape.borrow().is_enabled())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_tape_starts_enabled() {
        let tape = GradientTape::new();
        assert!(tape.is_enabled());
    }

    #[test]
    fn test_gradient_tape_disable_stops_recording() {
        let mut tape = GradientTape::new();
        tape.disable();
        assert!(!tape.is_enabled());
    }

    #[test]
    fn test_no_grad_disables_recording_during_closure() {
        clear_tape();
        let was_recording_inside = no_grad(|| is_recording());
        assert!(!was_recording_inside);
        assert!(is_recording());
    }

    #[test]
    fn test_is_recording_returns_true_by_default() {
        clear_tape();
        assert!(is_recording());
    }

    #[test]
    fn test_backward_runs_without_entries() {
        clear_tape();
        let loss = Tensor::from_vec(vec![1.0], vec![1]).unwrap();
        backward(&loss);
    }

    #[test]
    fn test_grad_returns_none_when_not_set() {
        clear_tape();
        let t = Tensor::zeros(vec![2]);
        let g = grad(&t);
        assert!(g.is_none());
    }

    #[test]
    fn test_set_grad_stores_gradient() {
        clear_tape();
        let t = Tensor::zeros(vec![2]);
        let g = Tensor::ones(vec![2]);
        set_grad(&t, g);
        let result = grad(&t);
        assert!(result.is_some());
        assert_eq!(result.unwrap().to_vec(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_backward_seeds_loss_gradient() {
        let mut tape = GradientTape::new();
        let loss = Tensor::from_vec(vec![5.0], vec![1]).unwrap();
        tape.backward(loss.id(), loss.shape());
        let g = tape.grad(loss.id());
        assert!(g.is_some());
        assert_eq!(g.unwrap().to_vec(), vec![1.0]);
    }
}
