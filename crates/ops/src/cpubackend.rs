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
}
