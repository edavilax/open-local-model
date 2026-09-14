use std::assert_eq;

use crate::{Backend, RopeTable};
use tensor::{Matrix, Tensor, Vector};

pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn add<const R: usize>(&self, a: &Tensor<R>, b: &Tensor<R>, out: &mut Tensor<R>) {
        same_shape_assert(&[a, b, out]);
        a.data_iter()
            .zip(b.data_iter())
            .zip(out.mut_data_iter())
            .for_each(|((&a_val, &b_val), c_ref)| *c_ref = a_val + b_val);
    }

    fn matmul(&self, a: &Matrix, b: &Matrix, out: &mut Matrix) {
        matmul_assert(a, b, out);
        let m = a.shape()[0];
        let n = b.shape()[0];
        // A[m,k] * B[n,k]
        for i in 0..m {
            for j in 0..n {
                let row_a = a.row(i);
                let row_b = b.row(j);
                assert!(out.set(&[i, j], dot_product(row_a, row_b)).is_ok());
            }
        }
    }

    fn hadamard_product<const R: usize>(&self, a: &Tensor<R>, b: &Tensor<R>, out: &mut Tensor<R>) {
        same_shape_assert(&[a, b, out]);
        a.data_iter()
            .zip(b.data_iter())
            .zip(out.mut_data_iter())
            .for_each(|((&a_val, &b_val), c_ref)| *c_ref = a_val * b_val);
    }

    fn rmsnorm(&self, t: &Matrix, w: &Vector, eps: f32, out: &mut Matrix) {
        rmsnorm_assert(t, w, out);
        let m = t.shape()[0];
        for i in 0..m {
            let row = t.row(i);
            let cnt = row.len() as f32;
            let sq: f32 = row.iter().map(|x| x * x).sum();
            let mean = sq / cnt + eps;
            let rms = f32::sqrt(mean);
            let scale = 1.0 / rms;
            let rms_row: Vec<f32> = row
                .iter()
                .zip(w.data_iter())
                .map(|(&x, &w_val)| x * scale * w_val)
                .collect();
            out.mut_row(i).clone_from_slice(&rms_row);
        }
    }

    fn rope(&self, t: &Matrix, table: &RopeTable, m_start: usize, out: &mut Matrix) {
        rope_assert(t, table, m_start, out);
        let m_end = t.shape()[0] + m_start;
        for m in m_start..m_end {
            let i = m - m_start;
            let data_row = t.row(i);
            let cos_row = table.cos.row(m);
            let sin_row = table.sin.row(m);
            for j_left in 0..cos_row.len() {
                let cos = cos_row[j_left];
                let sin = sin_row[j_left];
                let j_right = j_left + cos_row.len();
                assert!(
                    out.set(
                        &[i, j_left],
                        cos * data_row[j_left] - sin * data_row[j_right]
                    )
                    .is_ok()
                );
                assert!(
                    out.set(
                        &[i, j_right],
                        sin * data_row[j_left] + cos * data_row[j_right]
                    )
                    .is_ok()
                )
            }
        }
    }

    fn softmax(&self, t: &Matrix, out: &mut Matrix) {
        same_shape_assert(&[t, out]);
        let m = t.shape()[0];
        for i in 0..m {
            let data_row = t.row(i);
            let max = data_row
                .iter()
                .copied()
                .reduce(f32::max)
                .unwrap_or_default();
            if max == f32::NEG_INFINITY {
                out.mut_row(i).iter_mut().for_each(|y| *y = f32::NAN);
            } else {
                data_row
                    .iter()
                    .zip(out.mut_row(i))
                    .for_each(|(x, y)| *y = (x - max).exp());
                let sum: f32 = out.row(i).iter().sum();
                out.mut_row(i).iter_mut().for_each(|y| *y /= sum);
            }
        }
    }

    fn silu<const R: usize>(&self, t: &Tensor<R>, out: &mut Tensor<R>) {
        same_shape_assert(&[t, out]);
        t.data_iter()
            .zip(out.mut_data_iter())
            .for_each(|(x, y)| *y = x / (1.0 + (-1.0 * x).exp()));
    }

    fn embedding_lookup(&self, ids: &[u32], embed: &Matrix, out: &mut Matrix) {
        embedding_lookup_assert(ids, embed, out);
        for (i, id) in ids.iter().enumerate() {
            let ref_row = embed.row(*id as usize);
            out.mut_row(i).copy_from_slice(ref_row);
        }
    }
}

// Helper functions
fn matmul_assert(a: &Matrix, b: &Matrix, out: &Matrix) {
    assert_eq!(a.shape()[1], b.shape()[1]);
    assert_eq!(a.shape()[0], out.shape()[0]);
    assert_eq!(b.shape()[0], out.shape()[1]);
}

fn rmsnorm_assert(t: &Matrix, w: &Vector, out: &Matrix) {
    same_shape_assert(&[t, out]);
    assert_eq!(t.shape()[1], w.shape()[0]);
}

fn rope_assert(t: &Matrix, table: &RopeTable, m_start: usize, out: &Matrix) {
    same_shape_assert(&[&table.cos, &table.sin]);
    same_shape_assert(&[t, out]);
    assert_eq!(t.shape()[1], 2 * table.cos.shape()[1]);
    assert!(table.cos.shape()[0] >= m_start + t.shape()[0]);
}

fn same_shape_assert<const R: usize>(tensors: &[&Tensor<R>]) {
    for i in 1..tensors.len() {
        assert_eq!(tensors[i - 1].shape(), tensors[i].shape());
    }
}

fn embedding_lookup_assert(ids: &[u32], embed: &Matrix, out: &Matrix) {
    assert_eq!(&[ids.len(), embed.shape()[1]], out.shape());
    assert!(embed.shape()[0] > ids.iter().cloned().max().unwrap_or_default() as usize);
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use testutil::{DEFAULT_TOL, assert_close};

    fn mat(shape: [usize; 2], data: &[f32]) -> Matrix {
        Matrix::new(shape, data.to_vec()).unwrap()
    }

    fn vector(data: &[f32]) -> Vector {
        Vector::new([data.len()], data.to_vec()).unwrap()
    }

    fn data<const R: usize>(t: &Tensor<R>) -> Vec<f32> {
        t.data_iter().copied().collect()
    }

    /// A[2,3].
    fn a_2x3() -> Matrix {
        mat(
            [2, 3],
            &[
                1.5, -2.0, 0.0, // row 0
                -0.5, 3.0, -1.5, // row 1
            ],
        )
    }

    /// B[4,3]. Stored in the [n, k] layout `matmul` expects, i.e. each row
    /// here is a column of the mathematical B — no transpose is performed.
    fn b_4x3() -> Matrix {
        mat(
            [4, 3],
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
        let mut out = Matrix::zeros([2, 4]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);

        assert_eq!(out.shape(), &[2, 4]);
        assert_close(
            &data(&out),
            &[
                6.0, -6.5, 6.0, 2.0, // row 0
                1.0, 8.0, -10.5, 6.5, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn matmul_single_row() {
        let a = mat([1, 3], &[1.5, -2.0, 0.0]);
        let mut out = Matrix::zeros([1, 4]);
        CpuBackend {}.matmul(&a, &b_4x3(), &mut out);

        assert_eq!(out.shape(), &[1, 4]);
        assert_close(&data(&out), &[6.0, -6.5, 6.0, 2.0], DEFAULT_TOL);
    }

    #[test]
    fn matmul_single_column() {
        let b = mat([1, 3], &[4.0, 0.0, -2.0]);
        let mut out = Matrix::zeros([2, 1]);
        CpuBackend {}.matmul(&a_2x3(), &b, &mut out);

        assert_eq!(out.shape(), &[2, 1]);
        assert_close(&data(&out), &[6.0, 1.0], DEFAULT_TOL);
    }

    /// k = 1 degenerates the dot product to a single multiply.
    #[test]
    fn matmul_k_of_one() {
        let a = mat([2, 1], &[3.0, -2.0]);
        let b = mat([3, 1], &[1.0, 0.5, -4.0]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.matmul(&a, &b, &mut out);

        assert_close(
            &data(&out),
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
        let mut out = mat([2, 4], &[999.0; 8]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);

        assert_close(
            &data(&out),
            &[6.0, -6.5, 6.0, 2.0, 1.0, 8.0, -10.5, 6.5],
            DEFAULT_TOL,
        );
    }

    // Rejection cases. Each `expected` pins a value unique to the assertion
    // under test, so a panic from an earlier check cannot satisfy it.
    //
    // Shape/data disagreement and wrong rank are no longer runtime cases:
    // `Matrix::new` rejects the former (tested in the tensor crate) and the
    // type system rejects the latter.

    #[test]
    #[should_panic(expected = "right: 7")]
    fn matmul_rejects_k_mismatch() {
        let b = Matrix::zeros([4, 7]);
        let mut out = Matrix::zeros([2, 4]);
        CpuBackend {}.matmul(&a_2x3(), &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 5")]
    fn matmul_rejects_wrong_out_rows() {
        let mut out = Matrix::zeros([5, 4]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);
    }

    #[test]
    #[should_panic(expected = "right: 9")]
    fn matmul_rejects_wrong_out_cols() {
        let mut out = Matrix::zeros([2, 9]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out);
    }

    #[test]
    fn rmsnorm_valid() {
        let t = a_2x3();
        let w = vector(&[1.0, 2.0, 0.5]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(
            &data(&out),
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
        let w = vector(&[1.0; 3]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(
            &data(&out),
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
        let w = vector(&[1.0; 3]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);

        for i in 0..2 {
            let row = out.row(i);
            let rms = (row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32).sqrt();
            assert_close(&[rms], &[1.0], DEFAULT_TOL);
        }
    }

    /// `eps` exists so an all-zero row divides by `sqrt(eps)` instead of 0.
    /// Without it this row would be 0/0 = NaN.
    #[test]
    fn rmsnorm_zero_row_does_not_produce_nan() {
        let t = mat(
            [2, 3],
            &[
                0.0, 0.0, 0.0, // row 0 — degenerate
                3.0, 4.0, 0.0, // row 1
            ],
        );
        let w = vector(&[1.0; 3]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);

        assert!(out.data_iter().all(|x| x.is_finite()), "got {:?}", data(&out));
        assert_close(
            &data(&out),
            &[
                0.0, 0.0, 0.0, // row 0
                1.0392304, 1.3856406, 0.0, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rmsnorm_single_row() {
        let t = mat([1, 3], &[1.5, -2.0, 0.0]);
        let w = vector(&[1.0, 2.0, 0.5]);
        let mut out = Matrix::zeros([1, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(&data(&out), &[1.0392302, -2.7712806, 0.0], DEFAULT_TOL);
    }

    #[test]
    fn rmsnorm_overwrites_every_element_of_out() {
        let t = a_2x3();
        let w = vector(&[1.0, 2.0, 0.5]);
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out);
        assert_close(
            &data(&out),
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
    #[should_panic(expected = "right: 7")]
    fn rmsnorm_rejects_weight_length_mismatch() {
        let w = vector(&[1.0; 7]); // must match t's last dim, 3
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out);
    }

    #[test]
    #[should_panic(expected = "[3, 2]")]
    fn rmsnorm_rejects_out_shape_mismatch() {
        let w = vector(&[1.0; 3]);
        let mut out = Matrix::zeros([3, 2]); // same element count, wrong shape
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out);
    }

    #[test]
    fn rope_valid_using_vector() {
        const HEAD_DIM: usize = 4;
        let t = mat([1, HEAD_DIM], &[0.497, -0.138, 0.648, 1.523]);
        let mut out = Matrix::zeros([1, HEAD_DIM]);
        let table = RopeTable::new(HEAD_DIM, 10, 10000.0);
        // M = 0
        CpuBackend {}.rope(&t, &table, 0, &mut out);
        assert_close(&data(&out), &[0.497, -0.138, 0.648, 1.523], DEFAULT_TOL);
        // M = 1
        CpuBackend {}.rope(&t, &table, 1, &mut out);
        assert_close(
            &data(&out),
            &[-0.276743, -0.153223, 0.768327, 1.521544],
            DEFAULT_TOL,
        );
        // M = 2
        CpuBackend {}.rope(&t, &table, 2, &mut out);
        assert_close(
            &data(&out),
            &[-0.796050, -0.168430, 0.182258, 1.519936],
            DEFAULT_TOL,
        );
        // M = 3
        CpuBackend {}.rope(&t, &table, 3, &mut out);
        assert_close(
            &data(&out),
            &[-0.583472, -0.183621, -0.571378, 1.518175],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rope_valid_using_matrix() {
        const HEAD_DIM: usize = 6;
        const NUM_ROWS: usize = 5;
        let t = mat(
            [NUM_ROWS, HEAD_DIM],
            &[
                0.497, -0.138, 0.648, 1.523, -0.234, 0.812, // m = 0
                -0.345, 0.781, -0.112, 0.452, 1.104, -0.673, // m = 1
                0.912, -0.543, 0.321, -0.801, 0.219, -0.456, // m = 2
                -0.123, 0.654, -0.987, 0.314, -0.876, 0.543, // m = 3
                0.765, -0.432, 0.198, -0.541, 0.632, -0.879, // m = 4
            ],
        );
        let mut out = Matrix::zeros([NUM_ROWS, HEAD_DIM]);
        let table = RopeTable::new(HEAD_DIM, 10, 10000.0);
        CpuBackend {}.rope(&t, &table, 0, &mut out);
        assert_close(
            &data(&out),
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
    fn rope_input_4x8() -> Matrix {
        mat(
            [4, 8],
            &[
                0.497, -0.138, 0.648, 1.523, -0.234, 0.812, -0.345, 0.781, // row 0
                -0.112, 0.452, 1.104, -0.673, 0.912, -0.543, 0.321, -0.801, // row 1
                0.219, -0.456, -0.123, 0.654, -0.987, 0.314, -0.876, 0.543, // row 2
                0.765, -0.432, 0.198, -0.541, 0.632, -0.879, 0.111, -0.222, // row 3
            ],
        )
    }

    /// Copies row `i` of a matrix into its own [1, cols] matrix.
    fn row_matrix(t: &Matrix, i: usize) -> Matrix {
        mat([1, t.num_cols()], t.row(i))
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
        let mut batched = Matrix::zeros([4, 8]);
        CpuBackend {}.rope(&t, &table, m_start, &mut batched);

        for i in 0..4 {
            let mut single = Matrix::zeros([1, 8]);
            CpuBackend {}.rope(&row_matrix(&t, i), &table, m_start + i, &mut single);
            assert_close(batched.row(i), &data(&single), DEFAULT_TOL);
        }
    }

    /// Each (j, j + n/2) pair is rotated as a 2-D vector, so its length must
    /// not change. This is also a convention check: interleaved RoPE preserves
    /// (2j, 2j + 1) pairs instead, and would fail here.
    #[test]
    fn rope_preserves_pair_lengths() {
        let t = rope_input_4x8();
        let table = RopeTable::new(8, 16, 10000.0);
        let mut out = Matrix::zeros([4, 8]);
        CpuBackend {}.rope(&t, &table, 9, &mut out);

        for i in 0..4 {
            let (x, y) = (t.row(i), out.row(i));
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
        let (q, k) = (row_matrix(&base, 0), row_matrix(&base, 1));
        let table = RopeTable::new(8, 16, 10000.0);

        let rotated_dot = |m_q: usize, m_k: usize| -> f32 {
            let mut rq = Matrix::zeros([1, 8]);
            let mut rk = Matrix::zeros([1, 8]);
            CpuBackend {}.rope(&q, &table, m_q, &mut rq);
            CpuBackend {}.rope(&k, &table, m_k, &mut rk);
            dot_product(rq.row(0), rk.row(0))
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
        let mut input = [0.0; 8];
        input[1] = 1.0;
        let t = mat([1, 8], &input);
        let table = RopeTable::new(8, 4, 10000.0);
        let mut out = Matrix::zeros([1, 8]);
        CpuBackend {}.rope(&t, &table, 1, &mut out);

        // Table row 1, column 1: pair 1 at position 1.
        let (c, s) = (table.cos.row(1)[1], table.sin.row(1)[1]);
        assert_close(
            &data(&out),
            &[0.0, c, 0.0, 0.0, 0.0, s, 0.0, 0.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rope_overwrites_every_element_of_out() {
        let t = rope_input_4x8();
        let table = RopeTable::new(8, 16, 10000.0);
        let mut expected = Matrix::zeros([4, 8]);
        CpuBackend {}.rope(&t, &table, 2, &mut expected);

        let mut out = mat([4, 8], &[999.0; 32]);
        CpuBackend {}.rope(&t, &table, 2, &mut out);
        assert_close(&data(&out), &data(&expected), DEFAULT_TOL);
    }

    /// Decoding at the last table row is allowed: `m_start + rows == max_seq`.
    /// Pins the bounds check as `>=` rather than `>`.
    #[test]
    fn rope_accepts_last_table_row() {
        let t = row_matrix(&rope_input_4x8(), 0);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros([1, 8]);
        CpuBackend {}.rope(&t, &table, 6, &mut out);
        assert!(out.data_iter().any(|x| *x != 0.0));
    }

    // Rejection cases. Each `expected` pins a value unique to the assertion
    // under test, so a panic from an earlier check cannot satisfy it.

    /// The case you'll actually hit in practice: decode runs past `max_seq`.
    #[test]
    #[should_panic(expected = "m_start + t.shape()[0]")]
    fn rope_rejects_positions_past_table_end() {
        let t = row_matrix(&rope_input_4x8(), 0);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros([1, 8]);
        CpuBackend {}.rope(&t, &table, 7, &mut out); // table rows are 0..=6
    }

    #[test]
    #[should_panic(expected = "right: 8")]
    fn rope_rejects_head_dim_table_mismatch() {
        let t = Matrix::zeros([1, 6]);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros([1, 6]);
        CpuBackend {}.rope(&t, &table, 0, &mut out);
    }

    #[test]
    #[should_panic(expected = "[8, 1]")]
    fn rope_rejects_out_shape_mismatch() {
        let t = row_matrix(&rope_input_4x8(), 0); // [1, 8]
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros([8, 1]); // same element count, wrong shape
        CpuBackend {}.rope(&t, &table, 0, &mut out);
    }

    #[test]
    fn softmax_valid_using_vector() {
        let t = mat([1, 3], &[1.0, 2.0, 3.0]);
        let mut out = Matrix::zeros([1, 3]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_eq!(out.shape(), &[1, 3]);
        assert_close(
            &data(&out),
            &[
                0.09003057, 0.24472847, 0.66524096, // row 0
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_valid_using_matrix() {
        let t = mat(
            [5, 6],
            &[
                0.5, -1.2, 3.3, 0.0, 2.1, -0.7, // row 0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // row 1
                -2.0, -3.5, -0.25, -1.0, -4.0, -0.5, // row 2
                1e+01, 9.5, 8.0, 10.5, 7.25, 9.0, // row 3
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // row 4
            ],
        );
        let mut out = Matrix::zeros([5, 6]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_eq!(out.shape(), &[5, 6]);
        assert_close(
            &data(&out),
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
        let mut out = Matrix::zeros([4, 8]);
        CpuBackend {}.softmax(&t, &mut out);
        for i in 0..4 {
            let row = out.row(i);
            assert!(row.iter().all(|p| *p > 0.0 && *p < 1.0), "row {i}: {row:?}");
            assert_close(&[row.iter().sum::<f32>()], &[1.0], DEFAULT_TOL);
        }
    }

    /// Softmax is monotonic: a larger input always gets a larger probability.
    /// Catches a flipped exponent such as `exp(max - x)`.
    #[test]
    fn softmax_preserves_order_within_a_row() {
        let t = rope_input_4x8();
        let mut out = Matrix::zeros([4, 8]);
        CpuBackend {}.softmax(&t, &mut out);
        for i in 0..4 {
            let (x, y) = (t.row(i), out.row(i));
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
        let t = mat(
            [3, 3],
            &[
                1.0, 2.0, 3.0, // row 0: reference
                1000.0, 1001.0, 1002.0, // row 1: naive exp overflows to inf
                -1000.0, -999.0, -998.0, // row 2: naive exp underflows to 0
            ],
        );
        let mut out = Matrix::zeros([3, 3]);
        CpuBackend {}.softmax(&t, &mut out);

        let reference = [0.09003057, 0.24472847, 0.66524096];
        for i in 0..3 {
            assert_close(out.row(i), &reference, DEFAULT_TOL);
        }
    }

    /// The shape softmax actually sees in attention: a score matrix with -inf
    /// above the diagonal. Masked entries must be exactly 0.0, and the rest
    /// of each row must renormalize among themselves.
    #[test]
    fn softmax_causal_mask() {
        let ninf = f32::NEG_INFINITY;
        let t = mat(
            [3, 3],
            &[
                0.5, ninf, ninf, // query 0 sees key 0
                1.0, 2.0, ninf, // query 1 sees keys 0..=1
                1.0, 2.0, 3.0, // query 2 sees keys 0..=2
            ],
        );
        let mut out = Matrix::zeros([3, 3]);
        CpuBackend {}.softmax(&t, &mut out);

        // Exact, not approximate: a masked position must contribute nothing.
        for coords in [[0, 1], [0, 2], [1, 2]] {
            assert_eq!(out.get(&coords).unwrap(), 0.0, "masked {coords:?}");
        }
        assert_close(
            &data(&out),
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
        let t = mat([3, 1], &[-5.0, 0.0, 42.0]);
        let mut out = Matrix::zeros([3, 1]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_close(&data(&out), &[1.0, 1.0, 1.0], DEFAULT_TOL);
    }

    /// A NaN input must not be silently dropped. `f32::max` ignores NaN, so
    /// the row max is still finite here; the NaN has to show up through the
    /// sum. PyTorch returns an all-NaN row in this case.
    #[test]
    fn softmax_propagates_nan() {
        let t = mat([2, 3], &[1.0, f32::NAN, 2.0, 1.0, 2.0, 3.0]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.softmax(&t, &mut out);
        assert!(
            out.row(0).iter().all(|p| p.is_nan()),
            "row 0: {:?}",
            out.row(0)
        );
        // The NaN must not leak into the next row.
        assert_close(
            out.row(1),
            &[0.09003057, 0.24472847, 0.66524096],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_overwrites_every_element_of_out() {
        let t = rope_input_4x8();
        let mut expected = Matrix::zeros([4, 8]);
        CpuBackend {}.softmax(&t, &mut expected);

        let mut out = mat([4, 8], &[999.0; 32]);
        CpuBackend {}.softmax(&t, &mut out);
        assert_close(&data(&out), &data(&expected), DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "[3, 2]")]
    fn softmax_rejects_out_shape_mismatch() {
        let t = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros([3, 2]); // same element count, wrong shape
        CpuBackend {}.softmax(&t, &mut out);
    }

    #[test]
    fn silu_valid_using_vector() {
        let t = vector(&[0.0, 1.0, -1.0]);
        let mut out = Vector::zeros([3]);
        CpuBackend {}.silu(&t, &mut out);
        assert_eq!(out.shape(), &[3]);
        assert_close(&data(&out), &[0.0, 0.7310586, -0.26894143], DEFAULT_TOL);
    }

    #[test]
    fn silu_valid_using_matrix() {
        let t = mat(
            [4, 6],
            &[
                -1.2785, -1.0, -0.5, 0.0, 0.5, 1.0, // row 0
                2.0, 3.0, -2.0, -3.0, 4.0, -4.0, // row 1
                0.1, -0.1, 0.25, -0.25, 10.0, -10.0, // row 2
                5.5, -5.5, 0.75, -0.75, 1.5, -1.5, // row 3
            ],
        );
        let mut out = Matrix::zeros([4, 6]);
        CpuBackend {}.silu(&t, &mut out);
        assert_eq!(out.shape(), &[4, 6]);
        assert_close(
            &data(&out),
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
        let t = mat([1, 6], &[100.0, -100.0, 88.0, -88.0, 20.0, -20.0]);
        let mut out = Matrix::zeros([1, 6]);
        CpuBackend {}.silu(&t, &mut out);

        assert!(out.data_iter().all(|y| y.is_finite()), "{:?}", data(&out));
        // Large positive passes through; large negative decays to zero.
        assert_close(&data(&out), &[100.0, 0.0, 88.0, 0.0, 20.0, 0.0], DEFAULT_TOL);
    }

    /// SiLU is not monotonic: it dips to about -0.2785 near x = -1.2785, then
    /// climbs back toward 0. Points further out in either direction must sit
    /// above the dip.
    #[test]
    fn silu_has_a_minimum_near_negative_1_2785() {
        let t = mat([1, 5], &[-6.0, -3.0, -1.2785, -0.5, -0.1]);
        let mut out = Matrix::zeros([1, 5]);
        CpuBackend {}.silu(&t, &mut out);

        let out = data(&out);
        let dip = out[2];
        assert_close(&[dip], &[-0.27846456], DEFAULT_TOL);
        for (i, y) in out.iter().enumerate() {
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
        let forward_in = mat([2, 3], &[-2.0, -0.5, 0.0, 0.75, 1.5, 3.0]);
        let mut forward_out = Matrix::zeros([2, 3]);
        CpuBackend {}.silu(&forward_in, &mut forward_out);

        let mut reversed: Vec<f32> = data(&forward_in);
        reversed.reverse();
        let reversed_in = mat([2, 3], &reversed);
        let mut reversed_out = Matrix::zeros([2, 3]);
        CpuBackend {}.silu(&reversed_in, &mut reversed_out);

        let mut expected = data(&forward_out);
        expected.reverse();
        assert_close(&data(&reversed_out), &expected, DEFAULT_TOL);
    }

    #[test]
    fn silu_overwrites_every_element_of_out() {
        let t = mat([2, 3], &[-2.0, -0.5, 0.0, 0.75, 1.5, 3.0]);
        let mut expected = Matrix::zeros([2, 3]);
        CpuBackend {}.silu(&t, &mut expected);

        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.silu(&t, &mut out);
        assert_close(&data(&out), &data(&expected), DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "[3, 2]")]
    fn silu_rejects_out_shape_mismatch() {
        let t = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros([3, 2]); // same element count, wrong shape
        CpuBackend {}.silu(&t, &mut out);
    }

    #[test]
    fn add_valid_using_vector() {
        let a = vector(&[1.0, -2.0, 0.5]);
        let b = vector(&[0.25, 2.0, -1.5]);
        let mut out = Vector::zeros([3]);
        CpuBackend {}.add(&a, &b, &mut out);
        assert_eq!(out.shape(), &[3]);
        assert_close(&data(&out), &[1.25, 0.0, -1.0], DEFAULT_TOL);
    }

    #[test]
    fn add_valid_using_matrix() {
        let a = mat(
            [3, 4],
            &[
                0.0, 1.0, -1.0, 0.5, // row 0
                2.5, -0.75, 4.0, -8.0, // row 1
                100.0, -0.125, 3.25, 6.0, // row 2
            ],
        );
        let b = mat(
            [3, 4],
            &[
                0.0, -1.0, -2.0, 0.25, // row 0
                -2.5, 0.75, 0.5, 8.0, // row 1
                0.5, 0.125, -3.25, -12.0, // row 2
            ],
        );
        let mut out = Matrix::zeros([3, 4]);
        CpuBackend {}.add(&a, &b, &mut out);
        assert_eq!(out.shape(), &[3, 4]);
        assert_close(
            &data(&out),
            &[
                0.0, 0.0, -3.0, 0.75, // row 0
                0.0, 0.0, 4.5, 0.0, // row 1
                100.5, 0.0, 0.0, -6.0, // row 2
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn add_is_commutative() {
        let a = mat([2, 3], &[1.0, -2.0, 0.5, 3.25, -0.125, 0.0]);
        let b = mat([2, 3], &[0.25, 2.0, -1.5, -3.25, 8.0, 4.0]);
        let mut ab = Matrix::zeros([2, 3]);
        let mut ba = Matrix::zeros([2, 3]);
        CpuBackend {}.add(&a, &b, &mut ab);
        CpuBackend {}.add(&b, &a, &mut ba);
        assert_close(&data(&ab), &data(&ba), DEFAULT_TOL);
    }

    /// Adding zeros must leave the input untouched.
    #[test]
    fn add_zero_is_identity() {
        let a = mat([2, 3], &[1.0, -2.0, 0.5, 3.25, -0.125, 0.0]);
        let zeros = Matrix::zeros([2, 3]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.add(&a, &zeros, &mut out);
        assert_close(&data(&out), &data(&a), DEFAULT_TOL);
    }

    #[test]
    fn add_overwrites_every_element_of_out() {
        let a = mat([2, 3], &[1.0, -2.0, 0.5, 3.25, -0.125, 0.0]);
        let b = mat([2, 3], &[0.25, 2.0, -1.5, -3.25, 8.0, 4.0]);
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.add(&a, &b, &mut out);
        assert_close(&data(&out), &[1.25, 0.0, -1.0, 0.0, 7.875, 4.0], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "[3, 2]")]
    fn add_rejects_mismatched_input_shapes() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([3, 2], &[1.0; 6]); // same element count, wrong shape
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.add(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "[6, 1]")]
    fn add_rejects_out_shape_mismatch() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros([6, 1]);
        CpuBackend {}.add(&a, &b, &mut out);
    }

    #[test]
    fn hadamard_product_valid_using_vector() {
        let a = vector(&[2.0, -3.0, 0.5]);
        let b = vector(&[0.25, 0.5, -4.0]);
        let mut out = Vector::zeros([3]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out);
        assert_eq!(out.shape(), &[3]);
        assert_close(&data(&out), &[0.5, -1.5, -2.0], DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_valid_using_matrix() {
        let a = mat(
            [3, 4],
            &[
                0.0, 1.0, -1.0, 0.5, // row 0
                2.5, -0.75, 4.0, -8.0, // row 1
                1.5, -0.125, 3.25, 6.0, // row 2
            ],
        );
        let b = mat(
            [3, 4],
            &[
                3.0, -1.0, -2.0, 0.25, // row 0
                -2.0, 4.0, 0.5, 0.125, // row 1
                0.5, 8.0, -4.0, -0.5, // row 2
            ],
        );
        let mut out = Matrix::zeros([3, 4]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out);
        assert_eq!(out.shape(), &[3, 4]);
        assert_close(
            &data(&out),
            &[
                0.0, -1.0, 2.0, 0.125, // row 0
                -5.0, -3.0, 2.0, -1.0, // row 1
                0.75, -1.0, -13.0, -3.0, // row 2
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn hadamard_product_is_commutative() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 0.0]);
        let b = mat([2, 3], &[0.25, 0.5, -4.0, 8.0, -2.0, 3.0]);
        let mut ab = Matrix::zeros([2, 3]);
        let mut ba = Matrix::zeros([2, 3]);
        CpuBackend {}.hadamard_product(&a, &b, &mut ab);
        CpuBackend {}.hadamard_product(&b, &a, &mut ba);
        assert_close(&data(&ab), &data(&ba), DEFAULT_TOL);
    }

    /// Multiplying by ones leaves the input untouched; multiplying by zeros
    /// erases it. Together these pin that it is elementwise and not a matrix
    /// product, which would not satisfy either for non-square shapes.
    #[test]
    fn hadamard_product_ones_and_zeros() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 0.0]);
        let mut out = Matrix::zeros([2, 3]);

        CpuBackend {}.hadamard_product(&a, &mat([2, 3], &[1.0; 6]), &mut out);
        assert_close(&data(&out), &data(&a), DEFAULT_TOL);

        CpuBackend {}.hadamard_product(&a, &Matrix::zeros([2, 3]), &mut out);
        assert_close(&data(&out), &[0.0; 6], DEFAULT_TOL);
    }

    /// Output `i` must depend only on input `i`, so reversing both inputs must
    /// reverse the output. Catches an index mistake that symmetric data hides.
    #[test]
    fn hadamard_product_is_elementwise() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 4.0]);
        let b = mat([2, 3], &[0.25, 0.5, -4.0, 8.0, -2.0, 3.0]);
        let mut forward = Matrix::zeros([2, 3]);
        CpuBackend {}.hadamard_product(&a, &b, &mut forward);

        let rev = |t: &Matrix| {
            let mut d = data(t);
            d.reverse();
            mat([2, 3], &d)
        };
        let mut reversed = Matrix::zeros([2, 3]);
        CpuBackend {}.hadamard_product(&rev(&a), &rev(&b), &mut reversed);

        let mut expected = data(&forward);
        expected.reverse();
        assert_close(&data(&reversed), &expected, DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_overwrites_every_element_of_out() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 0.0]);
        let b = mat([2, 3], &[0.25, 0.5, -4.0, 8.0, -2.0, 3.0]);
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out);
        assert_close(&data(&out), &[0.5, -1.5, -2.0, 10.0, 1.0, 0.0], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "[3, 2]")]
    fn hadamard_product_rejects_mismatched_input_shapes() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([3, 2], &[1.0; 6]);
        let mut out = Matrix::zeros([2, 3]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out);
    }

    #[test]
    #[should_panic(expected = "[6, 1]")]
    fn hadamard_product_rejects_out_shape_mismatch() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros([6, 1]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out);
    }

    /// embed[i] = [i*10, i*10+1, i*10+2], so each row is recognizable.
    fn embed_5x3() -> Matrix {
        mat(
            [5, 3],
            &[
                0.0, 1.0, 2.0, // id 0
                10.0, 11.0, 12.0, // id 1
                20.0, 21.0, 22.0, // id 2
                30.0, 31.0, 32.0, // id 3
                40.0, 41.0, 42.0, // id 4
            ],
        )
    }

    /// Out-of-order ids with a repeat. Sequential ids would hide a bug that
    /// ignores `ids` and uses the output position instead.
    #[test]
    fn embedding_lookup_gathers_rows() {
        let embed = embed_5x3();
        let ids = [3u32, 0, 3, 4];
        let mut out = Matrix::zeros([4, 3]);
        CpuBackend {}.embedding_lookup(&ids, &embed, &mut out);
        assert_close(
            &data(&out),
            &[
                30.0, 31.0, 32.0, // id 3
                0.0, 1.0, 2.0, // id 0
                30.0, 31.0, 32.0, // id 3 again
                40.0, 41.0, 42.0, // id 4
            ],
            DEFAULT_TOL,
        );
    }

    /// Token ids range over the vocabulary, not the hidden size. The toy has
    /// vocab 1024 and hidden 64, and its real prompt starts with ids
    /// [392, 425, 672, ...] — all far larger than hidden.
    #[test]
    fn embedding_lookup_accepts_ids_larger_than_hidden_size() {
        let table: Vec<f32> = (0..400).map(|i| i as f32).collect();
        let embed = mat([100, 4], &table); // vocab 100, hidden 4
        let ids = [99u32, 64, 0];
        let mut out = Matrix::zeros([3, 4]);
        CpuBackend {}.embedding_lookup(&ids, &embed, &mut out);
        assert_close(
            &data(&out),
            &[
                396.0, 397.0, 398.0, 399.0, // id 99
                256.0, 257.0, 258.0, 259.0, // id 64
                0.0, 1.0, 2.0, 3.0, // id 0
            ],
            DEFAULT_TOL,
        );
    }

    /// The decode shape: one token at a time.
    #[test]
    fn embedding_lookup_single_id() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros([1, 3]);
        CpuBackend {}.embedding_lookup(&[2u32], &embed, &mut out);
        assert_close(&data(&out), &[20.0, 21.0, 22.0], DEFAULT_TOL);
    }

    /// A repeated token copies the same row again; no deduplication.
    #[test]
    fn embedding_lookup_repeats_rows() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros([3, 3]);
        CpuBackend {}.embedding_lookup(&[2u32, 2, 2], &embed, &mut out);
        assert_close(
            &data(&out),
            &[20.0, 21.0, 22.0, 20.0, 21.0, 22.0, 20.0, 21.0, 22.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn embedding_lookup_empty_ids() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros([0, 3]);
        CpuBackend {}.embedding_lookup(&[], &embed, &mut out);
        assert_eq!(out.data_iter().count(), 0);
    }

    #[test]
    fn embedding_lookup_overwrites_every_element_of_out() {
        let embed = embed_5x3();
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.embedding_lookup(&[1u32, 0], &embed, &mut out);
        assert_close(&data(&out), &[10.0, 11.0, 12.0, 0.0, 1.0, 2.0], DEFAULT_TOL);
    }

    /// An id at or past `vocab_size` means a tokenizer/vocab mismatch and must
    /// be rejected.
    #[test]
    #[should_panic(expected = "embed.shape()[0] >")]
    fn embedding_lookup_rejects_id_past_vocab() {
        let table: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let embed = mat([5, 8], &table); // vocab 5, hidden 8
        let mut out = Matrix::zeros([1, 8]);
        CpuBackend {}.embedding_lookup(&[5u32], &embed, &mut out); // ids are 0..=4
    }

    #[test]
    #[should_panic(expected = "right: [3, 2]")]
    fn embedding_lookup_rejects_out_shape_mismatch() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros([3, 2]); // should be [2, 3]
        CpuBackend {}.embedding_lookup(&[1u32, 0], &embed, &mut out);
    }
}
