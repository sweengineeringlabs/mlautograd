use crate::api::gradient::types::add_backward::AddBackward;
use crate::api::traits::backward_op::BackwardOp;
use crate::api::types::tensor::Tensor;

impl BackwardOp for AddBackward {
    fn backward(&self, grad_output: &Tensor, _saved: &[Tensor]) -> Vec<Tensor> {
        let grad_a = grad_output.unbroadcast_to(&self.a_shape);
        let grad_b = grad_output.unbroadcast_to(&self.b_shape);
        vec![grad_a, grad_b]
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_backward_produces_two_gradients() {
        let op = AddBackward {
            a_shape: vec![2, 3],
            b_shape: vec![2, 3],
        };
        let grad = Tensor::ones(vec![2, 3]);
        let grads = op.backward(&grad, &[]);
        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].shape(), &[2, 3]);
        assert_eq!(grads[1].shape(), &[2, 3]);
    }

    #[test]
    fn test_unbroadcast_to_same_shape_returns_clone() {
        let t = Tensor::ones(vec![2, 3]);
        let result = t.unbroadcast_to(&[2, 3]);
        assert_eq!(result.shape(), &[2, 3]);
    }
}
