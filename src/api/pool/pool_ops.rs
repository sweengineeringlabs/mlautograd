/// Interface contract for thread-local tensor buffer pool operations.
///
/// Provides the namespace for acquire, release, and clear_pool operations
/// on the thread-local buffer pool.
/// The concrete implementation lives in `core/pool/pool_ops.rs`.
pub type PoolOps = ();
