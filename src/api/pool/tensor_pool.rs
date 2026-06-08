/// Interface contract for the tensor buffer pool.
///
/// A pool that manages reusable `Vec<f32>` buffers bucketed by power-of-two
/// capacity to amortize allocation costs during gradient computation.
/// The concrete implementation lives in `core/pool/tensor_pool.rs`.
pub type TensorPool = ();
