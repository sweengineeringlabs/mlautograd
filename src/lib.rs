pub mod error;
pub mod tensor_id;
pub mod tensor;
pub mod pool;
pub mod tape;
pub mod gradient;

pub use error::{MlError, MlResult};
pub use tensor::Tensor;
pub use tensor_id::TensorId;
pub use tape::{BackwardOp, GradientTape, TapeEntry};
pub use tape::{record_op, backward, grad, set_grad, no_grad, clear_tape, is_recording};
pub use gradient::add::{unbroadcast, AddBackward};
pub use gradient::matmul::MatMulBackward;
pub use gradient::mul::MulBackward;
pub use gradient::relu::ReLUBackward;
pub use gradient::sigmoid::SigmoidBackward;
pub use gradient::tanh::TanhBackward;
