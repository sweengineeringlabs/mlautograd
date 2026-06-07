# mlautograd

Automatic differentiation engine for Rust. Provides a gradient tape, backward ops, buffer pooling, and the core `Tensor` wrapper used by the `sweengineeringlabs` ML stack.

## Use cases

### ML: custom loss functions with automatic differentiation
Define any differentiable computation over `Tensor` values, record ops on the tape, and call `backward` to get exact gradients — no finite-difference approximations.

### Physics simulation: differentiable simulators for robotics
Wrap a rigid-body or articulated-body step function with autograd ops. Backpropagate through the simulator to learn control policies or tune physical parameters from trajectory data.

### Financial modeling: gradient-based calibration of pricing models
Express an options pricing model (Black-Scholes, Heston, etc.) as a graph of `Tensor` operations. Minimize calibration loss against market quotes with any gradient-based optimizer.

### Control systems: differentiable control loops
Represent a PID or MPC controller as a differentiable computation. Use gradients of closed-loop cost with respect to controller gains to tune parameters automatically.

## Crate layout

| Module | Contents |
|---|---|
| `tensor` | `Tensor` wrapper around `llmtensor::Tensor` with `TensorId` and `requires_grad` |
| `tensor_id` | Monotonic `TensorId` for tape bookkeeping |
| `tape` | Thread-local `GradientTape`, `BackwardOp` trait, `TapeEntry`, and the free functions `backward`, `grad`, `no_grad`, etc. |
| `pool` | Thread-local `Vec<f32>` buffer pool for zero-copy backward pass allocations |
| `gradient` | Built-in backward ops: `AddBackward`, `MatMulBackward`, `MulBackward`, `ReLUBackward`, `SigmoidBackward`, `SoftmaxBackward`, `TanhBackward` |
| `error` | `MlError` / `MlResult` shared across the stack |

## Quick start

```rust
use mlautograd::{Tensor, tape, MlResult};

fn main() -> MlResult<()> {
    tape::clear_tape();

    let mut x = Tensor::from_vec(vec![2.0, 3.0], vec![2])?;
    x.set_requires_grad(true);

    // Forward pass — ops are recorded automatically
    let y = x.mul_scalar_raw(3.0);
    let loss = Tensor::from_vec(vec![y.sum_all_raw()], vec![1])?;

    tape::backward(&loss);

    if let Some(grad) = tape::grad(&x) {
        println!("grad x = {:?}", grad.to_vec()); // [3.0, 3.0]
    }
    Ok(())
}
```
