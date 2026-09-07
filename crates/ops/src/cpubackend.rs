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
    return m * i + j;
}

fn get_row(t: &Tensor, i: usize) -> &[f32] {
    let start = get_index(t, i, 0);
    let end = start + t.shape.get_dim(1).unwrap_or_default();
    return &t.data[start..end];
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use tensor::Shape;

    use super::*;

    #[test]
    fn matmul_valid() {
        let a = Tensor {
            shape: Shape::new(&[2, 3]),
            data: [
                1.5, -2.0, 0.0, // row 0
                -0.5, 3.0, -1.5, // row 1
            ]
            .to_vec(),
        };
        let b = Tensor {
            shape: Shape::new(&[4, 3]),
            // Note: Data is transposed
            data: [
                4.0, 0.0, -2.0, // col 0
                -1.0, 2.5, 0.0, // col 1
                0.0, -3.0, 1.0, // col 2
                2.0, 0.5, -4.0, // col 3
            ]
            .to_vec(),
        };
        let mut out = Tensor {
            shape: Shape::new(&[2, 4]),
            data: vec![0.0; 8],
        };
        let backend = CpuBackend {};
        backend.matmul(&a, &b, &mut out);
        let expected = Tensor {
            shape: Shape::new(&[2, 4]),
            data: [
                6.0, -6.5, 6.0, 2.0, // row 0
                1.0, 8.0, -10.5, 6.5, // row 1
            ]
            .to_vec(),
        };
        assert_eq!(expected, out);
    }

    #[test]
    fn matmul_single_row() {
        let a = Tensor {
            shape: Shape::new(&[1, 3]),
            data: [
                1.5, -2.0, 0.0, // row 0
            ]
            .to_vec(),
        };
        let b = Tensor {
            shape: Shape::new(&[4, 3]),
            // Note: Data is transposed
            data: [
                4.0, 0.0, -2.0, // col 0
                -1.0, 2.5, 0.0, // col 1
                0.0, -3.0, 1.0, // col 2
                2.0, 0.5, -4.0, // col 3
            ]
            .to_vec(),
        };
        let mut out = Tensor {
            shape: Shape::new(&[1, 4]),
            data: vec![0.0; 4],
        };
        let backend = CpuBackend {};
        backend.matmul(&a, &b, &mut out);
        let expected = Tensor {
            shape: Shape::new(&[1, 4]),
            data: [
                6.0, -6.5, 6.0, 2.0, // row 0
            ]
            .to_vec(),
        };
        assert_eq!(expected, out);
    }

    #[should_panic]
    #[test]
    fn matmul_invalid() {
        let a = Tensor {
            shape: Shape::new(&[2, 3]),
            data: [
                1.5, -2.0, 0.0, // row 0
                -0.5, 3.0, -1.5, // row 1
            ]
            .to_vec(),
        };
        let b = Tensor {
            shape: Shape::new(&[4, 2]),
            // Note: Data is transposed
            data: [
                4.0, 0.0, // col 0
                -1.0, 2.5, // col 1
                0.0, -3.0, // col 2
                2.0, 0.5, // col 3
            ]
            .to_vec(),
        };
        let mut out = Tensor {
            shape: Shape::new(&[2, 4]),
            data: vec![0.0; 8],
        };
        let backend = CpuBackend {};
        backend.matmul(&a, &b, &mut out);
    }
}
