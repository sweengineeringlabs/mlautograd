use crate::api::types::tape_entry::TapeEntry;
use crate::api::types::tensor::Tensor;
use crate::api::types::tensor_id::TensorId;

pub trait GradientTapeOp {
    fn record(&mut self, entry: TapeEntry);
    fn backward(&mut self, loss_id: TensorId, loss_shape: &[usize]);
    fn grad(&self, id: TensorId) -> Option<&Tensor>;
    fn clear(&mut self);
    fn enable(&mut self);
    fn disable(&mut self);
    fn is_enabled(&self) -> bool;
}
