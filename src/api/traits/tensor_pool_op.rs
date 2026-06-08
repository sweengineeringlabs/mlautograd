pub trait TensorPoolOp {
    fn acquire(&mut self, size: usize) -> Vec<f32>;
    fn release(&mut self, buf: Vec<f32>);
    fn clear(&mut self);
}
