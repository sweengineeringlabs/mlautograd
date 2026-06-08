use crate::api::gradient::types::re_l_u_backward::ReLUBackward;
use crate::api::traits::backward_op::BackwardOp;
use crate::api::types::tensor::Tensor;

impl BackwardOp for ReLUBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let input = &saved[0];
        let input_data = input.to_vec();
        let mask_data: Vec<f32> = input_data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
        let mask =
            Tensor::from_vec(mask_data, input.shape().to_vec()).expect("relu mask");
        let grad_input = grad_output.mul_raw(&mask).expect("relu backward mul");
        vec![grad_input]
    }


}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_backward_zeroes_grad_for_negative_input() {
        let op = ReLUBackward;
        let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).unwrap();
        let grad_output = Tensor::ones(vec![4]);
        let grads = op.backward(&grad_output, &[input]);
        assert_eq!(grads[0].to_vec(), vec![0.0, 0.0, 1.0, 1.0]);
    }
}
