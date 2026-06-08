/// Interface contract for the gradient tape's internal state.
///
/// The tape inner state tracks recorded operations (`TapeEntry` list),
/// accumulated gradients (`TensorId → Tensor` map), and the enabled flag.
/// The concrete implementation lives in `core/gradient/tape/gradient_tape_inner.rs`.
pub type GradientTapeInner = ();
