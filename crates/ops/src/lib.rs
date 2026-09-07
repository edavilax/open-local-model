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
}
