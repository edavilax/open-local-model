use tensor::Tensor;

pub mod cpubackend;

pub trait Backend {
    /// Performs a 2D matrix multiplication, and provides the results to `out`.
    ///
    /// Note that this multiplication is accomplished by performing row-row dot
    /// products. So a normal A\[n,k\] * B\[k,m\] will not work with this function.
    /// Instead, you must first transpose B such that the number of columns
    /// match.
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor);

    /// Normalizes each row of the input tensor by root-mean-square and provides
    /// the results to `out`.
    ///
    /// `w` is the weight tensor. It is a learnable parameter that needs to be
    /// accounted for in the norm calculation, and is applied per element in
    /// the final norm calculation.
    ///
    /// `eps` is an additive factor to prevent divide by zero.
    fn rmsnorm(&self, t: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor);
}
