use crate::api::types::tensor::Tensor;

pub trait BackwardOp: Send + Sync {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor>;
}
