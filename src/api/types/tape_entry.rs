use crate::api::traits::backward_op::BackwardOp;
use crate::api::types::tensor_id::TensorId;
use crate::api::types::tensor::Tensor;

pub struct TapeEntry {
    pub backward_op: Box<dyn BackwardOp>,
    pub output_id: TensorId,
    pub input_ids: Vec<TensorId>,
    pub saved_tensors: Vec<Tensor>,
}
