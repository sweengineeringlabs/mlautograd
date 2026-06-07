use crate::api::backward_op::BackwardOp;
use crate::api::tensor::Tensor;

pub struct SigmoidBackward;

impl BackwardOp for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let sig = &saved[0];
        let ones = Tensor::ones(sig.shape().to_vec());
        let one_minus_sig = ones.sub_raw(sig).expect("sigmoid 1 - s");
        let local_grad = sig.mul_raw(&one_minus_sig).expect("sigmoid s*(1-s)");
        let grad_input = grad_output.mul_raw(&local_grad).expect("sigmoid backward mul");
        vec![grad_input]
    }

    fn name(&self) -> &str {
        "SigmoidBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_backward_at_half_gives_quarter() {
        let op = SigmoidBackward;
        let sig_output = Tensor::from_vec(vec![0.5], vec![1]).unwrap();
        let grad_output = Tensor::ones(vec![1]);
        let grads = op.backward(&grad_output, &[sig_output]);
        assert!((grads[0].to_vec()[0] - 0.25).abs() < 1e-6);
    }
}
