use std::assert_eq;

use crate::{Backend, RopeTable, get_index};
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

    fn rope(&self, t: &Tensor, table: &RopeTable, m_start: usize, out: &mut Tensor) {
        rope_assert(t, table, m_start, out);
        let m_end = t.shape.get_dim(0).unwrap_or_default() + m_start;
        for m in m_start..m_end {
            let i = m - m_start;
            let data_row = get_row(t, i);
            let cos_row = get_row(&table.cos, m);
            let sin_row = get_row(&table.sin, m);
            for j_left in 0..cos_row.len() {
                let cos = cos_row[j_left];
                let sin = sin_row[j_left];
                let j_right = j_left + cos_row.len();
                let out_idx_left = get_index(out, i, j_left);
                let out_idx_right = get_index(out, i, j_right);
                out.data[out_idx_left] = cos * data_row[j_left] - sin * data_row[j_right];
                out.data[out_idx_right] = sin * data_row[j_left] + cos * data_row[j_right];
            }
        }
    }

    fn softmax(&self, t: &Tensor, out: &mut Tensor) {
        softmax_assert(t, out);
        let m = t.shape.get_dim(0).unwrap_or_default();
        for i in 0..m {
            let data_row = get_row(t, i);
            let max = data_row
                .iter()
                .copied()
                .reduce(f32::max)
                .unwrap_or_default();
            if max == f32::NEG_INFINITY {
                for j in 0..data_row.len() {
                    let idx = get_index(out, i, j);
                    out.data[idx] = f32::NAN;
                }
            } else {
                for (j, x) in data_row.iter().enumerate() {
                    let idx = get_index(out, i, j);
                    let y = (x - max).exp();
                    out.data[idx] = y;
                }
                let sum: f32 = get_row(out, i).iter().sum();
                for j in 0..data_row.len() {
                    let idx = get_index(out, i, j);
                    out.data[idx] /= sum;
                }
            }
        }
    }

    fn silu(&self, t: &Tensor, out: &mut Tensor) {
        silu_assert(t, out);
        for (i, x) in t.data.iter().enumerate() {
            out.data[i] = x / (1.0 + (-1.0 * x).exp());
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

fn rope_assert(t: &Tensor, table: &RopeTable, m_start: usize, out: &Tensor) {
    assert!(t.is_valid());
    assert!(out.is_valid());
    assert!(table.cos.is_valid());
    assert!(table.sin.is_valid());
    assert_eq!(table.cos.shape, table.sin.shape);
    assert_eq!(t.shape, out.shape);
    assert_eq!(2, t.shape.ndims());
    assert_eq!(2, table.cos.shape.ndims());
    assert_eq!(2, table.sin.shape.ndims());
    assert_eq!(
        t.shape.get_dim(1).unwrap_or_default(),
        2 * table.cos.shape.get_dim(1).unwrap_or_default()
    );
    assert!(
        table.cos.shape.get_dim(0).unwrap_or_default()
            >= m_start + t.shape.get_dim(0).unwrap_or_default()
    );
}

fn softmax_assert(t: &Tensor, out: &Tensor) {
    assert!(t.is_valid());
    assert!(out.is_valid());
    assert_eq!(t.shape, out.shape);
    assert_eq!(2, t.shape.ndims());
}

fn silu_assert(t: &Tensor, out: &Tensor) {
    assert!(t.is_valid());
    assert!(out.is_valid());
    assert_eq!(t.shape, out.shape);
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
                -0.2553769, 3.064523, -0.3830654, // row 1
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
                -0.2553769, 3.064523, -0.3830654, // row 1
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

    #[test]
    fn rope_valid_using_vector() {
        const HEAD_DIM: usize = 4;
        let t = tensor(&[1, HEAD_DIM], &[0.497, -0.138, 0.648, 1.523]);
        let mut out = tensor(&[1, HEAD_DIM], &[0.0; 4]);
        let table = RopeTable::new(HEAD_DIM, 10, 10000.0);
        // M = 0
        CpuBackend {}.rope(&t, &table, 0, &mut out);
        assert_close(&out.data, &[0.497, -0.138, 0.648, 1.523], DEFAULT_TOL);
        // M = 1
        CpuBackend {}.rope(&t, &table, 1, &mut out);
        assert_close(
            &out.data,
            &[-0.276743, -0.153223, 0.768327, 1.521544],
            DEFAULT_TOL,
        );
        // M = 2
        CpuBackend {}.rope(&t, &table, 2, &mut out);
        assert_close(
            &out.data,
            &[-0.796050, -0.168430, 0.182258, 1.519936],
            DEFAULT_TOL,
        );
        // M = 3
        CpuBackend {}.rope(&t, &table, 3, &mut out);
        assert_close(
            &out.data,
            &[-0.583472, -0.183621, -0.571378, 1.518175],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rope_valid_using_matrix() {
        const HEAD_DIM: usize = 6;
        const NUM_ROWS: usize = 5;
        let t = tensor(
            &[NUM_ROWS, HEAD_DIM],
            &[
                0.497, -0.138, 0.648, 1.523, -0.234, 0.812, // m = 0
                -0.345, 0.781, -0.112, 0.452, 1.104, -0.673, // m = 1
                0.912, -0.543, 0.321, -0.801, 0.219, -0.456, // m = 2
                -0.123, 0.654, -0.987, 0.314, -0.876, 0.543, // m = 3
                0.765, -0.432, 0.198, -0.541, 0.632, -0.879, // m = 4
            ],
        );
        let mut out = tensor(&[NUM_ROWS, HEAD_DIM], &[0.0; HEAD_DIM * NUM_ROWS]);
        let table = RopeTable::new(HEAD_DIM, 10, 10000.0);
        CpuBackend {}.rope(&t, &table, 0, &mut out);
        assert_close(
            &out.data,
            &[
                0.497,
                -0.138,
                0.648,
                1.523,
                -0.234,
                0.812, // m = 0
                -0.56674916,
                0.7289341,
                -0.11054981,
                -0.04609085,
                1.1390488,
                -0.6732397, // m = 1
                0.3488213,
                -0.5609629,
                0.32296187,
                1.1626129,
                0.16772175,
                -0.4546126, // m = 2
                0.0774574,
                0.7692569,
                -0.99048895,
                -0.3282154,
                -0.776747,
                0.5366094, // m = 3
                -0.9094675,
                -0.541242,
                0.20556755,
                -0.2253327,
                0.5413918,
                -0.87726104, // m = 4
            ],
            DEFAULT_TOL,
        );
    }

    /// [4, 8] input shared by the rope property tests below.
    fn rope_input_4x8() -> Tensor {
        tensor(
            &[4, 8],
            &[
                0.497, -0.138, 0.648, 1.523, -0.234, 0.812, -0.345, 0.781, // row 0
                -0.112, 0.452, 1.104, -0.673, 0.912, -0.543, 0.321, -0.801, // row 1
                0.219, -0.456, -0.123, 0.654, -0.987, 0.314, -0.876, 0.543, // row 2
                0.765, -0.432, 0.198, -0.541, 0.632, -0.879, 0.111, -0.222, // row 3
            ],
        )
    }

    /// Copies row `i` of a 2-D tensor into its own [1, cols] tensor.
    fn row_tensor(t: &Tensor, i: usize) -> Tensor {
        let cols = t.shape.get_dim(1).unwrap();
        tensor(&[1, cols], get_row(t, i))
    }

    /// Prefill and decode must agree: rotating several rows at once from
    /// `m_start` gives the same result as rotating each row alone at its own
    /// position. Phase 4's KV cache relies on this, and it catches mixing up
    /// the input row `i` with the table row `m`. The existing tests only
    /// exercise multi-row input at `m_start = 0`, where `i == m`.
    #[test]
    fn rope_multi_row_matches_per_row_decode() {
        let t = rope_input_4x8();
        let table = RopeTable::new(8, 16, 10000.0);
        let m_start = 3;
        let mut batched = tensor(&[4, 8], &[0.0; 32]);
        CpuBackend {}.rope(&t, &table, m_start, &mut batched);

        for i in 0..4 {
            let mut single = tensor(&[1, 8], &[0.0; 8]);
            CpuBackend {}.rope(&row_tensor(&t, i), &table, m_start + i, &mut single);
            assert_close(get_row(&batched, i), &single.data, DEFAULT_TOL);
        }
    }

    /// Each (j, j + n/2) pair is rotated as a 2-D vector, so its length must
    /// not change. This is also a convention check: interleaved RoPE preserves
    /// (2j, 2j + 1) pairs instead, and would fail here.
    #[test]
    fn rope_preserves_pair_lengths() {
        let t = rope_input_4x8();
        let table = RopeTable::new(8, 16, 10000.0);
        let mut out = tensor(&[4, 8], &[0.0; 32]);
        CpuBackend {}.rope(&t, &table, 9, &mut out);

        for i in 0..4 {
            let (x, y) = (get_row(&t, i), get_row(&out, i));
            for j in 0..4 {
                let before = x[j] * x[j] + x[j + 4] * x[j + 4];
                let after = y[j] * y[j] + y[j + 4] * y[j + 4];
                assert_close(&[after], &[before], DEFAULT_TOL);
            }
        }
    }

    /// The point of RoPE: the dot product of a rotated query and key depends
    /// only on the distance between their positions.
    #[test]
    fn rope_dot_product_depends_only_on_relative_position() {
        let base = rope_input_4x8();
        let (q, k) = (row_tensor(&base, 0), row_tensor(&base, 1));
        let table = RopeTable::new(8, 16, 10000.0);

        let rotated_dot = |m_q: usize, m_k: usize| -> f32 {
            let mut rq = tensor(&[1, 8], &[0.0; 8]);
            let mut rk = tensor(&[1, 8], &[0.0; 8]);
            CpuBackend {}.rope(&q, &table, m_q, &mut rq);
            CpuBackend {}.rope(&k, &table, m_k, &mut rk);
            dot_product(&rq.data, &rk.data)
        };

        // Every pair here is 2 positions apart.
        let reference = rotated_dot(2, 0);
        assert_close(&[rotated_dot(5, 3)], &[reference], DEFAULT_TOL);
        assert_close(&[rotated_dot(13, 11)], &[reference], DEFAULT_TOL);
        // A different distance must give a different result, or the checks
        // above prove nothing.
        assert!((rotated_dot(3, 0) - reference).abs() > 1e-3);
    }

    /// Rotate-half pairs element j with j + n/2. A one-hot input at index 1
    /// may only produce output at indices 1 and 5; interleaved RoPE would
    /// produce output at 0 and 1 instead.
    #[test]
    fn rope_one_hot_pairs_j_with_j_plus_half() {
        let mut data = [0.0; 8];
        data[1] = 1.0;
        let t = tensor(&[1, 8], &data);
        let table = RopeTable::new(8, 4, 10000.0);
        let mut out = tensor(&[1, 8], &[0.0; 8]);
        CpuBackend {}.rope(&t, &table, 1, &mut out);

        // Table row 1, column 1: pair 1 at position 1.
        let (c, s) = (get_row(&table.cos, 1)[1], get_row(&table.sin, 1)[1]);
        assert_close(
            &out.data,
            &[0.0, c, 0.0, 0.0, 0.0, s, 0.0, 0.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rope_overwrites_every_element_of_out() {
        let t = rope_input_4x8();
        let table = RopeTable::new(8, 16, 10000.0);
        let mut expected = tensor(&[4, 8], &[0.0; 32]);
        CpuBackend {}.rope(&t, &table, 2, &mut expected);

        let mut out = tensor(&[4, 8], &[999.0; 32]);
        CpuBackend {}.rope(&t, &table, 2, &mut out);
        assert_close(&out.data, &expected.data, DEFAULT_TOL);
    }

    /// Decoding at the last table row is allowed: `m_start + rows == max_seq`.
    /// Pins the bounds check as `>=` rather than `>`.
    #[test]
    fn rope_accepts_last_table_row() {
        let t = row_tensor(&rope_input_4x8(), 0);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = tensor(&[1, 8], &[0.0; 8]);
        CpuBackend {}.rope(&t, &table, 6, &mut out);
        assert!(out.data.iter().any(|x| *x != 0.0));
    }

    // Rejection cases. Each `expected` pins a value unique to the assertion
    // under test, so a panic from an earlier check cannot satisfy it.

    /// The case you'll actually hit in practice: decode runs past `max_seq`.
    #[test]
    #[should_panic(expected = "m_start + t.shape.get_dim(0)")]
    fn rope_rejects_positions_past_table_end() {
        let t = row_tensor(&rope_input_4x8(), 0);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = tensor(&[1, 8], &[0.0; 8]);
        CpuBackend {}.rope(&t, &table, 7, &mut out); // table rows are 0..=6
    }

    #[test]
    #[should_panic(expected = "right: 8")]
    fn rope_rejects_head_dim_table_mismatch() {
        let t = tensor(&[1, 6], &[0.0; 6]);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = tensor(&[1, 6], &[0.0; 6]);
        CpuBackend {}.rope(&t, &table, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "dims: [8, 1]")]
    fn rope_rejects_out_shape_mismatch() {
        let t = row_tensor(&rope_input_4x8(), 0); // [1, 8]
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = tensor(&[8, 1], &[0.0; 8]); // same element count, wrong shape
        CpuBackend {}.rope(&t, &table, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 1")]
    fn rope_rejects_non_2d_input() {
        let t = tensor(&[8], &[0.0; 8]);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = tensor(&[8], &[0.0; 8]);
        CpuBackend {}.rope(&t, &table, 0, &mut out);
    }

    #[test]
    fn softmax_valid_using_vector() {
        let t = tensor(&[1, 3], &[1.0, 2.0, 3.0]);
        let mut out = tensor(&[1, 3], &[0.0; 3]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_eq!(out.shape, Shape::new(&[1, 3]));
        assert_close(
            &out.data,
            &[
                0.09003057, 0.24472847, 0.66524096, // row 0
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_valid_using_matrix() {
        let t = tensor(
            &[5, 6],
            &[
                0.5, -1.2, 3.3, 0.0, 2.1, -0.7, // row 0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // row 1
                -2.0, -3.5, -0.25, -1.0, -4.0, -0.5, // row 2
                1e+01, 9.5, 8.0, 10.5, 7.25, 9.0, // row 3
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // row 4
            ],
        );
        let mut out = tensor(&[5, 6], &[0.0; 30]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_eq!(out.shape, Shape::new(&[5, 6]));
        assert_close(
            &out.data,
            &[
                0.042574773,
                0.0077777096,
                0.7001271,
                0.025822905,
                0.21087423,
                0.012823275, // row 0
                0.16666667,
                0.16666667,
                0.16666667,
                0.16666667,
                0.16666667,
                0.16666667, // row 1
                0.06986637,
                0.015589293,
                0.40205317,
                0.18991646,
                0.009455384,
                0.31311932, // row 2
                0.2616161,
                0.15867819,
                0.03540589,
                0.43133205,
                0.016724559,
                0.09624319, // row 3
                0.12792667,
                0.14138083,
                0.15624998,
                0.17268294,
                0.19084416,
                0.21091542, // row 4
            ],
            DEFAULT_TOL,
        );
    }

    /// Every output row is a probability distribution: entries in (0, 1),
    /// summing to 1.
    #[test]
    fn softmax_rows_sum_to_one() {
        let t = rope_input_4x8();
        let mut out = tensor(&[4, 8], &[0.0; 32]);
        CpuBackend {}.softmax(&t, &mut out);
        for i in 0..4 {
            let row = get_row(&out, i);
            assert!(row.iter().all(|p| *p > 0.0 && *p < 1.0), "row {i}: {row:?}");
            assert_close(&[row.iter().sum::<f32>()], &[1.0], DEFAULT_TOL);
        }
    }

    /// Softmax is monotonic: a larger input always gets a larger probability.
    /// Catches a flipped exponent such as `exp(max - x)`.
    #[test]
    fn softmax_preserves_order_within_a_row() {
        let t = rope_input_4x8();
        let mut out = tensor(&[4, 8], &[0.0; 32]);
        CpuBackend {}.softmax(&t, &mut out);
        for i in 0..4 {
            let (x, y) = (get_row(&t, i), get_row(&out, i));
            for j in 0..8 {
                for k in 0..8 {
                    if x[j] < x[k] {
                        assert!(y[j] < y[k], "row {i}: x[{j}] < x[{k}] but y[{j}] >= y[{k}]");
                    }
                }
            }
        }
    }

    /// Adding a constant to a row must not change its softmax. With logits
    /// near +/-1000, `exp` overflows or underflows f32 (overflow starts around
    /// 88.7) unless the row max is subtracted first, so this also proves the
    /// implementation is numerically stable.
    #[test]
    fn softmax_is_shift_invariant_and_stable_for_large_logits() {
        let t = tensor(
            &[3, 3],
            &[
                1.0, 2.0, 3.0, // row 0: reference
                1000.0, 1001.0, 1002.0, // row 1: naive exp overflows to inf
                -1000.0, -999.0, -998.0, // row 2: naive exp underflows to 0
            ],
        );
        let mut out = tensor(&[3, 3], &[0.0; 9]);
        CpuBackend {}.softmax(&t, &mut out);

        let reference = [0.09003057, 0.24472847, 0.66524096];
        for i in 0..3 {
            assert_close(get_row(&out, i), &reference, DEFAULT_TOL);
        }
    }

    /// The shape softmax actually sees in attention: a score matrix with -inf
    /// above the diagonal. Masked entries must be exactly 0.0, and the rest
    /// of each row must renormalize among themselves.
    #[test]
    fn softmax_causal_mask() {
        let ninf = f32::NEG_INFINITY;
        let t = tensor(
            &[3, 3],
            &[
                0.5, ninf, ninf, // query 0 sees key 0
                1.0, 2.0, ninf, // query 1 sees keys 0..=1
                1.0, 2.0, 3.0, // query 2 sees keys 0..=2
            ],
        );
        let mut out = tensor(&[3, 3], &[0.0; 9]);
        CpuBackend {}.softmax(&t, &mut out);

        // Exact, not approximate: a masked position must contribute nothing.
        for idx in [1, 2, 5] {
            assert_eq!(out.data[idx], 0.0, "masked index {idx}");
        }
        assert_close(
            &out.data,
            &[
                1.0, 0.0, 0.0, // row 0
                0.26894142, 0.7310586, 0.0, // row 1
                0.09003057, 0.24472847, 0.66524096, // row 2
            ],
            DEFAULT_TOL,
        );
    }

    /// With one column there is only one choice, so it gets all the mass.
    #[test]
    fn softmax_single_column_is_one() {
        let t = tensor(&[3, 1], &[-5.0, 0.0, 42.0]);
        let mut out = tensor(&[3, 1], &[0.0; 3]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_close(&out.data, &[1.0, 1.0, 1.0], DEFAULT_TOL);
    }

    /// A NaN input must not be silently dropped. `f32::max` ignores NaN, so
    /// the row max is still finite here; the NaN has to show up through the
    /// sum. PyTorch returns an all-NaN row in this case.
    #[test]
    fn softmax_propagates_nan() {
        let t = tensor(&[2, 3], &[1.0, f32::NAN, 2.0, 1.0, 2.0, 3.0]);
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.softmax(&t, &mut out);
        assert!(
            get_row(&out, 0).iter().all(|p| p.is_nan()),
            "row 0: {:?}",
            get_row(&out, 0)
        );
        // The NaN must not leak into the next row.
        assert_close(
            get_row(&out, 1),
            &[0.09003057, 0.24472847, 0.66524096],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_overwrites_every_element_of_out() {
        let t = rope_input_4x8();
        let mut expected = tensor(&[4, 8], &[0.0; 32]);
        CpuBackend {}.softmax(&t, &mut expected);

        let mut out = tensor(&[4, 8], &[999.0; 32]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_close(&out.data, &expected.data, DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "assertion failed: t.is_valid()")]
    fn softmax_rejects_invalid_input() {
        let t = tensor(&[2, 3], &[1.0; 5]); // 5 elements, shape needs 6
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.softmax(&t, &mut out);
    }

    #[test]
    #[should_panic(expected = "dims: [3, 2]")]
    fn softmax_rejects_out_shape_mismatch() {
        let t = tensor(&[2, 3], &[1.0; 6]);
        let mut out = tensor(&[3, 2], &[0.0; 6]); // same element count, wrong shape
        CpuBackend {}.softmax(&t, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 1")]
    fn softmax_rejects_non_2d_input() {
        let t = tensor(&[6], &[1.0; 6]);
        let mut out = tensor(&[6], &[0.0; 6]);
        CpuBackend {}.softmax(&t, &mut out);
    }

    #[test]
    fn silu_valid_using_vector() {
        let t = tensor(&[1, 3], &[0.0, 1.0, -1.0]);
        let mut out = tensor(&[1, 3], &[0.0; 3]);
        CpuBackend {}.silu(&t, &mut out);
        assert_eq!(out.shape, Shape::new(&[1, 3]));
        assert_close(&out.data, &[0.0, 0.7310586, -0.26894143], DEFAULT_TOL);
    }

    #[test]
    fn silu_valid_using_matrix() {
        let t = tensor(
            &[4, 6],
            &[
                -1.2785, -1.0, -0.5, 0.0, 0.5, 1.0, // row 0
                2.0, 3.0, -2.0, -3.0, 4.0, -4.0, // row 1
                0.1, -0.1, 0.25, -0.25, 10.0, -10.0, // row 2
                5.5, -5.5, 0.75, -0.75, 1.5, -1.5, // row 3
            ],
        );
        let mut out = tensor(&[4, 6], &[0.0; 24]);
        CpuBackend {}.silu(&t, &mut out);
        assert_eq!(out.shape, Shape::new(&[4, 6]));
        assert_close(
            &out.data,
            &[
                -0.27846456,
                -0.26894143,
                -0.18877034,
                0.0,
                0.31122968,
                0.7310586, // row 0
                1.7615942,
                2.8577223,
                -0.23840584,
                -0.14227761,
                3.928055,
                -0.07194484, // row 1
                0.05249792,
                -0.04750208,
                0.14054413,
                -0.109455876,
                9.999546,
                -0.0004539787, // row 2
                5.4776144,
                -0.022385757,
                0.50938404,
                -0.24061598,
                1.2263618,
                -0.27363828, // row 3
            ],
            DEFAULT_TOL,
        );
    }

    /// Large magnitudes must not produce inf or NaN. `exp(-x)` overflows f32
    /// for x below about -88.7, and `x / inf` gives the correct limit of 0.
    #[test]
    fn silu_handles_large_magnitudes() {
        let t = tensor(&[1, 6], &[100.0, -100.0, 88.0, -88.0, 20.0, -20.0]);
        let mut out = tensor(&[1, 6], &[0.0; 6]);
        CpuBackend {}.silu(&t, &mut out);

        assert!(out.data.iter().all(|y| y.is_finite()), "{:?}", out.data);
        // Large positive passes through; large negative decays to zero.
        assert_close(&out.data, &[100.0, 0.0, 88.0, 0.0, 20.0, 0.0], DEFAULT_TOL);
    }

    /// SiLU is not monotonic: it dips to about -0.2785 near x = -1.2785, then
    /// climbs back toward 0. Points further out in either direction must sit
    /// above the dip.
    #[test]
    fn silu_has_a_minimum_near_negative_1_2785() {
        let t = tensor(&[1, 5], &[-6.0, -3.0, -1.2785, -0.5, -0.1]);
        let mut out = tensor(&[1, 5], &[0.0; 5]);
        CpuBackend {}.silu(&t, &mut out);

        let dip = out.data[2];
        assert_close(&[dip], &[-0.27846456], DEFAULT_TOL);
        for (i, y) in out.data.iter().enumerate() {
            if i != 2 {
                assert!(*y > dip, "index {i} = {y} should exceed the dip {dip}");
            }
        }
    }

    /// Elementwise means output `i` depends only on input `i`, so reversing the
    /// input must reverse the output. Catches an indexing mistake that a
    /// symmetric input would hide.
    #[test]
    fn silu_is_elementwise() {
        let forward_in = tensor(&[2, 3], &[-2.0, -0.5, 0.0, 0.75, 1.5, 3.0]);
        let mut forward_out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.silu(&forward_in, &mut forward_out);

        let mut reversed: Vec<f32> = forward_in.data.clone();
        reversed.reverse();
        let reversed_in = tensor(&[2, 3], &reversed);
        let mut reversed_out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.silu(&reversed_in, &mut reversed_out);

        let mut expected = forward_out.data.clone();
        expected.reverse();
        assert_close(&reversed_out.data, &expected, DEFAULT_TOL);
    }

    #[test]
    fn silu_overwrites_every_element_of_out() {
        let t = tensor(&[2, 3], &[-2.0, -0.5, 0.0, 0.75, 1.5, 3.0]);
        let mut expected = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.silu(&t, &mut expected);

        let mut out = tensor(&[2, 3], &[999.0; 6]);
        CpuBackend {}.silu(&t, &mut out);
        assert_close(&out.data, &expected.data, DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "assertion failed: t.is_valid()")]
    fn silu_rejects_invalid_input() {
        let t = tensor(&[2, 3], &[1.0; 5]); // 5 elements, shape needs 6
        let mut out = tensor(&[2, 3], &[0.0; 6]);
        CpuBackend {}.silu(&t, &mut out);
    }

    #[test]
    #[should_panic(expected = "dims: [3, 2]")]
    fn silu_rejects_out_shape_mismatch() {
        let t = tensor(&[2, 3], &[1.0; 6]);
        let mut out = tensor(&[3, 2], &[0.0; 6]); // same element count, wrong shape
        CpuBackend {}.silu(&t, &mut out);
    }
}
