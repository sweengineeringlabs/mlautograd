use crate::api::backward_op::BackwardOp;
use crate::api::tensor::Tensor;

pub struct TanhBackward;

impl BackwardOp for TanhBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let tanh_out = &saved[0];
        let tanh_sq = tanh_out.pow_raw(2.0);
        let ones = Tensor::ones(tanh_out.shape().to_vec());
        let one_minus_sq = ones.sub_raw(&tanh_sq).expect("tanh 1 - t^2");
        let grad_input = grad_output.mul_raw(&one_minus_sq).expect("tanh backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "TanhBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tanh_backward_at_zero_output_gives_unit_grad() {
        let op = TanhBackward;
        let tanh_out = Tensor::from_vec(vec![0.0], vec![1]).unwrap();
        let grad_output = Tensor::ones(vec![1]);
        let grads = op.backward(&grad_output, &[tanh_out]);
        assert!((grads[0].to_vec()[0] - 1.0).abs() < 1e-6);
    }
}
