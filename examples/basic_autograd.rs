//! Basic autograd example — forward pass + backward pass with AddBackward.
//!
//! Demonstrates creating tensors, recording a gradient tape entry, running
//! backward, and retrieving per-input gradients.

use mlautograd::{AddBackward, TapeContext, TapeEntry, Tensor};

fn main() {
    // Start from a clean tape state.
    TapeContext::clear_tape();

    // Build two input tensors.
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).expect("tensor a");
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3]).expect("tensor b");

    // Simulate c = a + b (shape [3]).
    let c_data: Vec<f32> = a.to_vec().iter().zip(b.to_vec().iter()).map(|(x, y)| x + y).collect();
    let c = Tensor::from_vec(c_data, vec![3]).expect("tensor c");

    // Record the add operation on the tape.
    let entry = TapeEntry {
        output_id: c.id(),
        input_ids: vec![a.id(), b.id()],
        backward_op: Box::new(AddBackward {
            a_shape: a.shape().to_vec(),
            b_shape: b.shape().to_vec(),
        }),
        saved_tensors: vec![],
    };
    TapeContext::record_op(entry);

    // Run backward from c (loss is c itself for this simple demo).
    TapeContext::backward(&c);

    // Retrieve gradients — both should be ones (∂c/∂a = 1, ∂c/∂b = 1).
    let grad_a = TapeContext::grad(&a);
    let grad_b = TapeContext::grad(&b);

    println!("grad_a: {:?}", grad_a.as_ref().map(|g| g.to_vec()));
    println!("grad_b: {:?}", grad_b.as_ref().map(|g| g.to_vec()));

    assert_eq!(
        grad_a.expect("grad_a missing").to_vec(),
        vec![1.0, 1.0, 1.0]
    );
    assert_eq!(
        grad_b.expect("grad_b missing").to_vec(),
        vec![1.0, 1.0, 1.0]
    );

    println!("basic_autograd: OK");
}
