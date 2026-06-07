# mlautograd

> **TLDR:** Automatic differentiation engine — gradient tape, autograd Tensor, and backward ops for Rust ML. See [Overview](docs/README.md) for details.

## Table of Contents
- [Quick Start](#quick-start)
- [API](#api)
- [Documentation](#documentation)

## Quick Start

```rust
use mlautograd::{Tensor, tape};

let mut x = Tensor::from_vec(vec![2.0, 3.0], vec![2])?;
x.set_requires_grad(true);

// Forward pass — ops are recorded automatically on the thread-local tape
let y = x.mul_scalar_raw(3.0);
let loss = Tensor::from_vec(vec![y.sum_all_raw()], vec![1])?;

tape::backward(&loss);

if let Some(grad) = tape::grad(&x) {
    println!("grad x = {:?}", grad.to_vec()); // [3.0, 3.0]
}
```

## API

| Type | Description |
|------|-------------|
| `Tensor` | Autograd-aware tensor with unique `TensorId` and `requires_grad` flag |
| `TensorId` | Monotonic atomic counter used by the tape for tensor identity |
| `BackwardOp` | Implement to define a custom differentiable operation |
| `GradientTape` | Records forward ops as `TapeEntry` entries and replays the backward pass |
| `tape::no_grad` | Run a closure with gradient tracking disabled (inference mode) |
| `tape::backward` | Replay the tape from a loss tensor, accumulating gradients |
| `tape::grad` | Retrieve the accumulated gradient for a `Tensor` |
| `MlError` / `MlResult` | Shared error and result types across the ML stack |

## Documentation

- [Architecture](docs/3-design/architecture.md) - System design

## Related FRs

None — foundational crate, no feature requests.
