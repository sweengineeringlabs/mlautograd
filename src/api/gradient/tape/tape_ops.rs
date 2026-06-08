/// Interface contract for thread-local gradient tape operations.
///
/// Provides the namespace for low-level tape record, backward, grad,
/// set_grad, no_grad, clear, and is_recording operations.
/// The concrete implementation lives in `core/gradient/tape/tape_ops.rs`.
pub type TapeOps = ();
