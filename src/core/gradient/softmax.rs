use crate::api::backward_op::BackwardOp;
use crate::api::tensor::Tensor;

pub(crate) struct SoftmaxBackward {
    pub dim: i64,
}

impl BackwardOp for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor, saved: &[Tensor]) -> Vec<Tensor> {
        let softmax_out = &saved[0];
        let shape = softmax_out.shape().to_vec();
        let ndim = shape.len();

        let dim = if self.dim < 0 {
            (ndim as i64 + self.dim) as usize
        } else {
            self.dim as usize
        };

        let s_data = softmax_out.to_vec();
        let g_data = grad_output.to_vec();
        let total = s_data.len();

        let inner_size: usize = shape[(dim + 1)..].iter().product();
        let dim_size = shape[dim];
        let outer_size: usize = total / (dim_size * inner_size);

        let mut grad_input = vec![0.0f32; total];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let mut dot = 0.0f32;
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    dot += g_data[idx] * s_data[idx];
                }
                for d in 0..dim_size {
                    let idx = outer * dim_size * inner_size + d * inner_size + inner;
                    grad_input[idx] = s_data[idx] * (g_data[idx] - dot);
                }
            }
        }

        let result =
            Tensor::from_vec(grad_input, shape).expect("softmax backward from_vec");
        vec![result]
    }

    fn name(&self) -> &str {
        "SoftmaxBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_backward_output_shape_matches_input() {
        let op = SoftmaxBackward { dim: -1 };
        let softmax_out = Tensor::from_vec(vec![0.25, 0.25, 0.25, 0.25], vec![1, 4]).unwrap();
        let grad_output = Tensor::ones(vec![1, 4]);
        let grads = op.backward(&grad_output, &[softmax_out]);
        assert_eq!(grads[0].shape(), &[1, 4]);
    }
}
