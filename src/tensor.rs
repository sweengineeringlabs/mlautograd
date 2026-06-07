use swe_ml_tensor::{DType, Tensor as CoreTensor, TensorError, TensorResult};
use crate::error::{MlError, MlResult};
pub use crate::tensor_id::TensorId;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) id: TensorId,
    pub(crate) inner: CoreTensor,
    pub(crate) requires_grad: bool,
}

impl Tensor {
    // --- Constructors ---

    pub fn new(inner: CoreTensor, requires_grad: bool) -> Self {
        Self {
            id: TensorId::next(),
            inner,
            requires_grad,
        }
    }

    pub fn zeros(shape: impl Into<swe_ml_tensor::Shape>) -> Self {
        Self::new(CoreTensor::zeros(shape), false)
    }

    pub fn ones(shape: impl Into<swe_ml_tensor::Shape>) -> Self {
        Self::new(CoreTensor::ones(shape), false)
    }

    pub fn randn(shape: impl Into<swe_ml_tensor::Shape>) -> Self {
        Self::new(CoreTensor::randn(shape), false)
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<swe_ml_tensor::Shape>) -> TensorResult<Self> {
        Ok(Self::new(CoreTensor::from_vec(data, shape)?, false))
    }

    pub fn full(shape: impl Into<swe_ml_tensor::Shape>, value: f32) -> Self {
        Self::new(CoreTensor::full(shape, value), false)
    }

    // --- Accessors ---

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn inner(&self) -> &CoreTensor {
        &self.inner
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn data(&self) -> TensorResult<&[f32]> {
        self.inner.data()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    // --- Raw ops (no tape recording) ---

    pub fn matmul_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.matmul(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn add_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.add(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sub_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.sub(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn mul_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.mul(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn mul_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.mul_scalar(scalar), false)
    }

    pub fn transpose_raw(&self, dim0: i64, dim1: i64) -> TensorResult<Tensor> {
        let result = self.inner.transpose(dim0, dim1)?;
        Ok(Tensor::new(result, false))
    }

    pub fn relu_raw(&self) -> Tensor {
        Tensor::new(self.inner.relu(), false)
    }

    pub fn neg_raw(&self) -> Tensor {
        Tensor::new(self.inner.neg(), false)
    }

    pub fn mean_all_raw(&self) -> f32 {
        self.inner.mean_all()
    }

    pub fn sum_all_raw(&self) -> f32 {
        self.inner.sum_all()
    }

    pub fn pow_raw(&self, exp: f32) -> Tensor {
        Tensor::new(self.inner.pow(exp), false)
    }

    pub fn div_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.div(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn div_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.div_scalar(scalar), false)
    }

    pub fn add_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.add_scalar(scalar), false)
    }

    pub fn sqrt_raw(&self) -> Tensor {
        Tensor::new(self.inner.sqrt(), false)
    }

    pub fn reshape_raw(&self, shape: &[usize]) -> TensorResult<Tensor> {
        let result = self.inner.reshape(shape)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sum_raw(&self, dim: i64) -> TensorResult<Tensor> {
        let result = self.inner.sum(dim)?;
        Ok(Tensor::new(result, false))
    }

    // --- Shape ops (no tape recording) ---

    pub fn permute_raw(&self, dims: &[usize]) -> MlResult<Tensor> {
        let result = self.inner.permute(dims).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn squeeze_raw(&self, dim: i64) -> MlResult<Tensor> {
        let result = self.inner.squeeze(dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn unsqueeze_raw(&self, dim: i64) -> MlResult<Tensor> {
        let result = self.inner.unsqueeze(dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn flatten_raw(&self) -> MlResult<Tensor> {
        let n = self.numel();
        let result = self.inner.reshape(&[n]).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn view_raw(&self, shape: &[usize]) -> MlResult<Tensor> {
        let result = self.inner.reshape(shape).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    // --- Indexing ops (no tape recording) ---

    pub fn slice_raw(&self, dim: i64, start: usize, end: usize) -> MlResult<Tensor> {
        let result = self.inner.slice(dim, start, end).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn index_select_raw(&self, dim: i64, indices: &[usize]) -> MlResult<Tensor> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;
        let dim_size = shape[dim_idx];

        for &idx in indices {
            if idx >= dim_size {
                return Err(MlError::TensorError(TensorError::IndexOutOfBounds {
                    dim: dim_idx,
                    index: idx,
                    size: dim_size,
                }));
            }
        }

        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape[dim_idx] = indices.len();
        let out_numel: usize = out_shape.iter().product();

        let src_data = self.to_vec();
        let src_strides = compute_strides(&shape);
        let out_strides = compute_strides(&out_shape);

        let mut out_data = vec![0.0f32; out_numel];
        let mut out_indices = vec![0usize; ndim];

        for flat in 0..out_numel {
            let mut rem = flat;
            for d in 0..ndim {
                out_indices[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = if d == dim_idx {
                    indices[out_indices[d]]
                } else {
                    out_indices[d]
                };
                src_flat += idx * src_strides[d];
            }
            out_data[flat] = src_data[src_flat];
        }

        let result = CoreTensor::from_vec(out_data, out_shape).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn gather_raw(&self, dim: i64, index: &Tensor) -> MlResult<Tensor> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;

        let idx_shape = index.shape();
        if idx_shape.len() != ndim {
            return Err(MlError::TensorError(TensorError::InvalidOperation(
                format!(
                    "gather: index must have same ndim as self ({}), got {}",
                    ndim,
                    idx_shape.len()
                ),
            )));
        }

        let src_data = self.to_vec();
        let idx_data = index.to_vec();
        let src_strides = compute_strides(shape);
        let idx_strides = compute_strides(idx_shape);
        let out_numel: usize = idx_shape.iter().product();

        let mut out_data = vec![0.0f32; out_numel];
        let mut multi_idx = vec![0usize; ndim];

        for flat in 0..out_numel {
            let mut rem = flat;
            for d in 0..ndim {
                multi_idx[d] = rem / idx_strides[d];
                rem %= idx_strides[d];
            }
            let gather_idx = idx_data[flat] as usize;
            if gather_idx >= shape[dim_idx] {
                return Err(MlError::TensorError(TensorError::IndexOutOfBounds {
                    dim: dim_idx,
                    index: gather_idx,
                    size: shape[dim_idx],
                }));
            }
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = if d == dim_idx { gather_idx } else { multi_idx[d] };
                src_flat += idx * src_strides[d];
            }
            out_data[flat] = src_data[src_flat];
        }

        let result =
            CoreTensor::from_vec(out_data, idx_shape.to_vec()).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn masked_select_raw(&self, mask: &Tensor) -> MlResult<Tensor> {
        let self_data = self.to_vec();

        let mask_data = if self.shape() == mask.shape() {
            mask.to_vec()
        } else {
            let target = swe_ml_tensor::Shape::new(self.shape().to_vec());
            let broadcast_mask = mask
                .inner
                .broadcast_to(&target)
                .map_err(MlError::TensorError)?;
            broadcast_mask.to_vec()
        };

        let selected: Vec<f32> = self_data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, m)| **m != 0.0)
            .map(|(v, _)| *v)
            .collect();
        let n = selected.len();
        let result = CoreTensor::from_vec(selected, vec![n]).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    // --- Element-wise math ops (no tape recording) ---

    pub fn exp_raw(&self) -> Tensor {
        Tensor::new(self.inner.exp(), false)
    }

    pub fn log_raw(&self) -> Tensor {
        Tensor::new(self.inner.log(), false)
    }

    pub fn abs_raw(&self) -> Tensor {
        Tensor::new(self.inner.abs(), false)
    }

    // --- Reduction ops (no tape recording) ---

    pub fn mean_raw(&self, dim: i64) -> MlResult<Tensor> {
        let result = self.inner.mean(dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn max_raw(&self, dim: i64) -> MlResult<(Tensor, Tensor)> {
        let (values, indices) = self.inner.max(dim).map_err(MlError::TensorError)?;
        Ok((Tensor::new(values, false), Tensor::new(indices, false)))
    }

    pub fn min_raw(&self, dim: i64) -> MlResult<(Tensor, Tensor)> {
        let (values, indices) = self.inner.min(dim).map_err(MlError::TensorError)?;
        Ok((Tensor::new(values, false), Tensor::new(indices, false)))
    }

    pub fn var_raw(&self, dim: i64) -> MlResult<Tensor> {
        let result = self.inner.var(dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn std_dev_raw(&self, dim: i64) -> MlResult<Tensor> {
        let var = self.inner.var(dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(var.sqrt(), false))
    }

    // --- Joining ops (no tape recording) ---

    pub fn concat_raw(tensors: &[&Tensor], dim: i64) -> MlResult<Tensor> {
        let core_tensors: Vec<&CoreTensor> = tensors.iter().map(|t| &t.inner).collect();
        let result = CoreTensor::cat(&core_tensors, dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn stack_raw(tensors: &[&Tensor], dim: i64) -> MlResult<Tensor> {
        if tensors.is_empty() {
            return Err(MlError::TensorError(TensorError::EmptyTensor));
        }

        let first_shape = tensors[0].shape();
        for t in tensors.iter().skip(1) {
            if t.shape() != first_shape {
                return Err(MlError::TensorError(TensorError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: t.shape().to_vec(),
                }));
            }
        }

        let unsqueezed: Vec<CoreTensor> = tensors
            .iter()
            .map(|t| t.inner.unsqueeze(dim))
            .collect::<Result<Vec<_>, _>>()
            .map_err(MlError::TensorError)?;
        let refs: Vec<&CoreTensor> = unsqueezed.iter().collect();
        let result = CoreTensor::cat(&refs, dim).map_err(MlError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    pub fn split_raw(&self, split_size: usize, dim: i64) -> MlResult<Vec<Tensor>> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;
        let dim_size = shape[dim_idx];

        if split_size == 0 {
            return Err(MlError::TensorError(TensorError::InvalidOperation(
                "split_size must be greater than 0".into(),
            )));
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let end = (start + split_size).min(dim_size);
            let chunk = self
                .inner
                .slice(dim, start, end)
                .map_err(MlError::TensorError)?;
            chunks.push(Tensor::new(chunk, false));
            start = end;
        }

        Ok(chunks)
    }

    /// Replace inner data while preserving TensorId (for optimizer in-place updates).
    pub fn update_data_from(&mut self, other: &Tensor) {
        self.inner = other.inner.clone();
    }
}

// --- Helper functions ---

fn normalize_dim_helper(dim: i64, ndim: usize) -> MlResult<usize> {
    let ndim_i64 = ndim as i64;
    let normalized = if dim < 0 { dim + ndim_i64 } else { dim };
    if normalized >= 0 && normalized < ndim_i64 {
        Ok(normalized as usize)
    } else {
        Err(MlError::TensorError(TensorError::InvalidDimension {
            dim,
            ndim,
        }))
    }
}

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_creates_all_zero_tensor() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_ones_creates_all_one_tensor() {
        let t = Tensor::ones(vec![2, 3]);
        assert!(t.to_vec().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_numel_returns_total_elements() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_add_raw_sums_element_wise() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]).unwrap();
        let c = a.add_raw(&b).unwrap();
        assert_eq!(c.to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_tensor_ids_are_unique() {
        let t1 = Tensor::zeros(vec![1]);
        let t2 = Tensor::zeros(vec![1]);
        assert_ne!(t1.id(), t2.id());
    }

    #[test]
    fn test_set_requires_grad_toggles_flag() {
        let mut t = Tensor::zeros(vec![2]);
        assert!(!t.requires_grad());
        t.set_requires_grad(true);
        assert!(t.requires_grad());
    }

    #[test]
    fn test_compute_strides_for_3d_shape() {
        let strides = compute_strides(&[2, 3, 4]);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_normalize_dim_helper_negative_dim() {
        let result = normalize_dim_helper(-1, 3).unwrap();
        assert_eq!(result, 2);
    }

    #[test]
    fn test_normalize_dim_helper_out_of_bounds() {
        let result = normalize_dim_helper(5, 3);
        assert!(result.is_err());
    }
}
