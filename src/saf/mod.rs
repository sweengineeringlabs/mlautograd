pub use crate::core::gradient_tape::{record_op, backward, grad, set_grad, no_grad, clear_tape, is_recording};
pub use crate::core::gradient::add::{unbroadcast, AddBackward};
pub use crate::core::gradient::matmul::MatMulBackward;
pub use crate::core::gradient::mul::MulBackward;
pub use crate::core::gradient::relu::ReLUBackward;
pub use crate::core::gradient::sigmoid::SigmoidBackward;
pub use crate::core::gradient::tanh::TanhBackward;
// GradientTape stays internal — do NOT re-export it
