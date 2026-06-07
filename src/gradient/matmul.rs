use crate::tape::BackwardOp;
use crate::tensor::Tensor;

pub struct MatMulBackward;

impl BackwardOp for MatMulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let a = &saved[0];
        let b = &saved[1];

        let b_t = b.transpose_raw(-1, -2).expect("transpose B");
        let grad_a = grad_output.matmul_raw(&b_t).expect("matmul grad_a");

        let a_t = a.transpose_raw(-1, -2).expect("transpose A");
        let grad_b = a_t.matmul_raw(grad_output).expect("matmul grad_b");

        vec![grad_a, grad_b]
    }

    fn name(&self) -> &str {
        "MatMulBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_backward_produces_two_gradients() {
        let op = MatMulBackward;
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 4]);
        let grad_output = Tensor::ones(vec![2, 4]);
        let grads = op.backward(&grad_output, &[a, b]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]);
        assert_eq!(grads[1].shape(), &[3, 4]);
    }
}
