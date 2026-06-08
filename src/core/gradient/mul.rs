use crate::api::gradient::types::mul_backward::MulBackward;
use crate::api::traits::backward_op::BackwardOp;
use crate::api::types::tensor::Tensor;

impl BackwardOp for MulBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let a = &saved[0];
        let b = &saved[1];

        let grad_a_full = grad_output.mul_raw(b).expect("mul grad_a");
        let grad_b_full = grad_output.mul_raw(a).expect("mul grad_b");

        let grad_a = grad_a_full.unbroadcast_to(a.shape());
        let grad_b = grad_b_full.unbroadcast_to(b.shape());

        vec![grad_a, grad_b]
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_backward_produces_correct_gradients() {
        let op = MulBackward;
        let a = Tensor::from_vec(vec![2.0, 3.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0], vec![2]).unwrap();
        let grad_output = Tensor::ones(vec![2]);
        let grads = op.backward(&grad_output, &[a, b]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].to_vec(), vec![4.0, 5.0]);
        assert_eq!(grads[1].to_vec(), vec![2.0, 3.0]);
    }
}
