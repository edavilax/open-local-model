use crate::Backend;
use tensor::Tensor;

pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        matmul_assert(a, b, out);
        let m = a.shape.get_dim(0).unwrap_or_default();
        let n = b.shape.get_dim(0).unwrap_or_default();
        // A[m,k] * B[n,k]
        for i in 0..m {
            for j in 0..n {
                let idx = get_index(out, i, j);
                out.data[idx] = dot_product(get_row(a, i), get_row(b, j));
            }
        }
    }

    fn rmsnorm(&self, t: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) {
        rmsnorm_assert(t, w, out);
        let m = t.shape.get_dim(0).unwrap_or_default();
        for i in 0..m {
            let row = get_row(t, i);
            let cnt = row.len() as f32;
            let sq: f32 = row.iter().map(|x| x * x).sum();
            let mean = sq / cnt + eps;
            let rms = f32::sqrt(mean);
            let scale = 1.0 / rms;
            for (j, x) in row.iter().enumerate() {
                let idx = get_index(out, i, j);
                out.data[idx] = x * scale * w.data[j];
            }
        }
    }
}

// Helper functions
fn matmul_assert(a: &Tensor, b: &Tensor, out: &Tensor) {
    assert!(a.is_valid());
    assert!(b.is_valid());
    assert!(out.is_valid());
    assert_eq!(2, a.shape.ndims());
    assert_eq!(2, b.shape.ndims());
    assert_eq!(2, out.shape.ndims());
    assert_eq!(a.shape.get_dim(1), b.shape.get_dim(1));
    assert_eq!(a.shape.get_dim(0), out.shape.get_dim(0));
    assert_eq!(b.shape.get_dim(0), out.shape.get_dim(1));
}

fn rmsnorm_assert(t: &Tensor, w: &Tensor, out: &Tensor) {
    assert!(t.is_valid());
    assert!(out.is_valid());
    assert!(w.is_valid());
    assert_eq!(2, t.shape.ndims());
    assert_eq!(1, w.shape.ndims());
    assert_eq!(t.shape.get_dim(1), w.shape.get_dim(0));
    assert_eq!(t.shape, out.shape);
}

fn get_index(t: &Tensor, i: usize, j: usize) -> usize {
    let m = t.shape.get_dim(1).unwrap_or_default();
    m * i + j
}

fn get_row(t: &Tensor, i: usize) -> &[f32] {
    let start = get_index(t, i, 0);
    let end = start + t.shape.get_dim(1).unwrap_or_default();
    &t.data[start..end]
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor::Shape;
    use testutil::{DEFAULT_TOL, assert_close};

    fn tensor(dims: &[usize], data: &[f32]) -> Tensor {
        Tensor {
            shape: Shape::new(dims),
            data: data.to_vec(),
        }
    }

    /// A[2,3].
    fn a_2x3() -> Tensor {
        tensor(
            &[2, 3],
            &[
                1.5, -2.0, 0.0, // row 0
                -0.5, 3.0, -1.5, // row 1
            ],
        )
    }

    /// B[4,3]. Stored in the [n, k] layout `matmul` expects, i.e. each row
    /// here is a column of the mathematical B — no transpose is performed.
    fn b_4x3() -> Tensor {
        tensor(
            &[4, 3],
            &[
                4.0, 0.0, -2.0, // row 0
                -1.0, 2.5, 0.0, // row 1
                0.0, -3.0, 1.0, // row 2
                2.0, 0.5, -4.0, // row 3
            ],
        )
    }

    #[test]
    fn matmul_valid() {
        let mut out = tensor(&[2, 4], &[0.0; 8]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);

        assert_eq!(out.shape, Shape::new(&[2, 4]));
        assert_close(
            &out.data,
            &[
                6.0, -6.5, 6.0, 2.0, // row 0
                1.0, 8.0, -10.5, 6.5, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn matmul_single_row() {
        let a = tensor(&[1, 3], &[1.5, -2.0, 0.0]);
        let mut out = tensor(&[1, 4], &[0.0; 4]);
        CpuBackend {}.matmul(&a, &b_4x3(), &mut out);

        assert_eq!(out.shape, Shape::new(&[1, 4]));
        assert_close(&out.data, &[6.0, -6.5, 6.0, 2.0], DEFAULT_TOL);
    }

    #[test]
    fn matmul_single_column() {
        let b = tensor(&[1, 3], &[4.0, 0.0, -2.0]);
        let mut out = tensor(&[2, 1], &[0.0; 2]);
        CpuBackend {}.matmul(&a_2x3(), &b, &mut out);

        assert_eq!(out.shape, Shape::new(&[2, 1]));
        assert_close(&out.data, &[6.0, 1.0], DEFAULT_TOL);
    }

    /// k = 1 degenerates the dot product to a single multiply.
    #[test]
    fn matmul_k_of_one() {
        let a = tensor(&[2, 1], &[3.0, -2.0]);
        let b = tensor(&[3, 1], &[1.0, 0.5, -4.0]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.matmul(&a, &b, &mut out);

        assert_close(
            &out.data,
            &[
                3.0, 1.5, -12.0, // row 0
                -2.0, -1.0, 8.0, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    /// Every element of `out` must be written, not accumulated into. A stale
    /// value surviving here would mean the loop skipped a cell.
    #[test]
    fn matmul_overwrites_every_element_of_out() {
        let mut out = tensor(&[2, 4], &[999.0; 8]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);

        assert_close(
            &out.data,
            &[6.0, -6.5, 6.0, 2.0, 1.0, 8.0, -10.5, 6.5],
            DEFAULT_TOL,
        );
    }

    // Rejection cases. Each `expected` pins a value unique to the assertion
    // under test, so a panic from an earlier check cannot satisfy it.

    #[test]
    #[should_panic(expected = "assertion failed: a.is_valid()")]
    fn matmul_rejects_shape_data_disagreement() {
        let a = tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0]); // 5 elems, needs 6
        let mut out = tensor(&[2, 4], &[0.0; 8]);
        CpuBackend {}.matmul(&a, &b_4x3(), &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 1")]
    fn matmul_rejects_non_2d_input() {
        let a = tensor(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut out = tensor(&[2, 4], &[0.0; 8]);
        CpuBackend {}.matmul(&a, &b_4x3(), &mut out);
    }

    #[test]
    #[should_panic(expected = "right: Some(7)")]
    fn matmul_rejects_k_mismatch() {
        let b = tensor(&[4, 7], &[0.0; 28]);
        let mut out = tensor(&[2, 4], &[0.0; 8]);
        CpuBackend {}.matmul(&a_2x3(), &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: Some(5)")]
    fn matmul_rejects_wrong_out_rows() {
        let mut out = tensor(&[5, 4], &[0.0; 20]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);
    }

    #[test]
    #[should_panic(expected = "right: Some(9)")]
    fn matmul_rejects_wrong_out_cols() {
        let mut out = tensor(&[2, 9], &[0.0; 18]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);
    }

    #[test]
    fn rmsnorm_valid() {
        let t = a_2x3();
        let w = tensor(&[3], &[1.0, 2.0, 0.5]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(
            &out.data,
            &[
                1.0392302, -2.7712806, 0.0, // row 0
                -0.2553769, 3.0645231, -0.3830654, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    /// With an all-ones weight the op is pure normalization, which isolates
    /// the RMS computation from the per-feature scale.
    #[test]
    fn rmsnorm_identity_weight() {
        let t = a_2x3();
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(
            &out.data,
            &[
                1.0392302, -1.3856403, 0.0, // row 0
                -0.2553769, 1.5322616, -0.7661308, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    /// The defining property: after normalization each row has RMS 1. This
    /// holds for any input, so it catches a wrong divisor without depending
    /// on hand-computed expectations.
    #[test]
    fn rmsnorm_output_rows_have_unit_rms() {
        let t = a_2x3();
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);

        for i in 0..2 {
            let row = get_row(&out, i);
            let rms = (row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32).sqrt();
            assert_close(&[rms], &[1.0], DEFAULT_TOL);
        }
    }

    /// `eps` exists so an all-zero row divides by `sqrt(eps)` instead of 0.
    /// Without it this row would be 0/0 = NaN.
    #[test]
    fn rmsnorm_zero_row_does_not_produce_nan() {
        let t = tensor(
            &[2, 3],
            &[
                0.0, 0.0, 0.0, // row 0 — degenerate
                3.0, 4.0, 0.0, // row 1
            ],
        );
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);

        assert!(out.data.iter().all(|x| x.is_finite()), "got {:?}", out.data);
        assert_close(
            &out.data,
            &[
                0.0, 0.0, 0.0, // row 0
                1.0392304, 1.3856406, 0.0, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rmsnorm_single_row() {
        let t = tensor(&[1, 3], &[1.5, -2.0, 0.0]);
        let w = tensor(&[3], &[1.0, 2.0, 0.5]);
        let mut out = tensor(&[1, 3], &[0.0; 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(&out.data, &[1.0392302, -2.7712806, 0.0], DEFAULT_TOL);
    }

    #[test]
    fn rmsnorm_overwrites_every_element_of_out() {
        let t = a_2x3();
        let w = tensor(&[3], &[1.0, 2.0, 0.5]);
        let mut out = tensor(&[2, 3], &[999.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(
            &out.data,
            &[
                1.0392302, -2.7712806, 0.0, // row 0
                -0.2553769, 3.0645231, -0.3830654, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    // Rejection cases. Each `expected` pins a value unique to the assertion
    // under test, so a panic from an earlier check cannot satisfy it.

    #[test]
    #[should_panic(expected = "assertion failed: w.is_valid()")]
    fn rmsnorm_rejects_invalid_weight() {
        let w = tensor(&[3], &[1.0, 2.0]); // 2 elems, shape says 3
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 1")]
    fn rmsnorm_rejects_non_2d_input() {
        let t = tensor(&[6], &[1.0; 6]);
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = tensor(&[6], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 2")]
    fn rmsnorm_rejects_non_1d_weight() {
        let w = tensor(&[3, 1], &[1.0; 3]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: Some(7)")]
    fn rmsnorm_rejects_weight_length_mismatch() {
        let w = tensor(&[7], &[1.0; 7]); // must match t's last dim, 3
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out);
    }

    #[test]
    #[should_panic(expected = "dims: [3, 2]")]
    fn rmsnorm_rejects_out_shape_mismatch() {
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = tensor(&[3, 2], &[0.0; 6]); // same element count, wrong shape
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out);
    }
}
