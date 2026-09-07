use std::assert_eq;

use crate::Backend;
use tensor::Tensor;

pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) {
        matmul_assert(a, b, out);
        let n = a.shape.get_dim(0).unwrap_or_default();
        let m = b.shape.get_dim(0).unwrap_or_default();
        // A[n,k] * B[m,k]
        for i in 0..n {
            for j in 0..m {
                let idx = get_index(out, i, j);
                out.data[idx] = dot_product(get_row(a, i), get_row(b, j));
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
}
