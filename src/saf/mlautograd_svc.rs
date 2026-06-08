use crate::api::gradient::types::buffer_pool::BufferPool;
use crate::api::gradient::types::tape_context::TapeContext;
use crate::api::traits::validator::Validator;
use crate::api::types::tape_entry::TapeEntry;
use crate::api::types::tensor::Tensor;
use crate::core::gradient::tape::tape_ops::TapeOps;
use crate::core::pool::pool_ops::PoolOps;
use crate::core::validation::default_validator::DefaultValidator;

// Anchor api interface contracts so the types are reachable from a live code path.
// These are the api-layer counterparts for the core/ implementation modules.
const _: crate::api::gradient::tape::gradient_tape_inner::GradientTapeInner = ();
const _: crate::api::gradient::tape::tape_ops::TapeOps = ();
const _: crate::api::pool::pool_ops::PoolOps = ();
const _: crate::api::pool::tensor_pool::TensorPool = ();

impl TapeContext {
    /// Record a gradient tape entry on the thread-local tape.
    pub fn record_op(entry: TapeEntry) {
        TapeOps::record_op_inner(entry);
    }

    /// Run backward pass from the given loss tensor.
    pub fn backward(loss: &Tensor) {
        TapeOps::backward_inner(loss.id(), loss.shape());
    }

    /// Retrieve the gradient for a tensor from the thread-local tape.
    pub fn grad(tensor: &Tensor) -> Option<Tensor> {
        TapeOps::grad_inner(tensor.id())
    }

    /// Store a gradient for a tensor on the thread-local tape.
    /// Silently discards the gradient if it fails validation (e.g. empty tensor).
    pub fn set_grad(tensor: &Tensor, grad: Tensor) {
        if DefaultValidator.validate(&grad).is_ok() {
            TapeOps::set_grad_inner(tensor.id(), grad);
        }
    }

    /// Execute a closure with the tape disabled, then restore the previous state.
    pub fn no_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        TapeOps::no_grad_inner(f)
    }

    /// Clear all entries and gradients from the thread-local tape.
    pub fn clear_tape() {
        TapeOps::clear_tape_inner();
    }

    /// Return whether the thread-local tape is currently recording.
    pub fn is_recording() -> bool {
        TapeOps::is_recording_inner()
    }
}

impl BufferPool {
    /// Acquire a zeroed buffer of the given size from the thread-local pool.
    pub fn acquire(size: usize) -> Vec<f32> {
        PoolOps::acquire_inner(size)
    }

    /// Release a buffer back to the thread-local pool for reuse.
    pub fn release(buf: Vec<f32>) {
        PoolOps::release_inner(buf);
    }

    /// Clear all pooled buffers from the thread-local pool.
    pub fn clear_pool() {
        PoolOps::clear_pool_inner();
    }
}
