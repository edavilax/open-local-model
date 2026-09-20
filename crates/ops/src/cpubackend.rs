use crate::{Backend, OpsError, RopeTable};
use itertools::izip;
use tensor::{Matrix, Tensor, Vector};

pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn add<const R: usize>(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        out: &mut Tensor<R>,
    ) -> Result<(), OpsError> {
        check_same_shape(&[a, b, out])?;
        izip!(out.as_mut_f32()?, a.as_f32()?, b.as_f32()?)
            .for_each(|(out_val, a_val, b_val)| *out_val = a_val + b_val);
        Ok(())
    }

    fn matmul(&self, a: &Matrix, b: &Matrix, out: &mut Matrix) -> Result<(), OpsError> {
        check_matmul(a, b, out)?;
        let k = a.shape()[1];
        let a_data = a.as_f32()?;
        let b_data = b.as_f32()?;
        let out_data = out.as_mut_f32()?;
        let mut idx = 0;
        // A[m,k] * B[n,k]
        for a_row in a_data.chunks_exact(k) {
            for b_col in b_data.chunks_exact(k) {
                out_data[idx] = dot_product(a_row, b_col);
                idx += 1;
            }
        }
        Ok(())
    }

    fn hadamard_product<const R: usize>(
        &self,
        a: &Tensor<R>,
        b: &Tensor<R>,
        out: &mut Tensor<R>,
    ) -> Result<(), OpsError> {
        check_same_shape(&[a, b, out])?;
        izip!(out.as_mut_f32()?, a.as_f32()?, b.as_f32()?)
            .for_each(|(out_val, a_val, b_val)| *out_val = a_val * b_val);
        Ok(())
    }

    fn rmsnorm(&self, t: &Matrix, w: &Vector, eps: f32, out: &mut Matrix) -> Result<(), OpsError> {
        check_rmsnorm(t, w, out)?;
        let m = t.shape()[0];
        for i in 0..m {
            let row = t.row_f32(i)?;
            let cnt = row.len() as f32;
            let sq: f32 = row.iter().map(|x| x * x).sum();
            let mean = sq / cnt + eps;
            let rms = f32::sqrt(mean);
            let scale = 1.0 / rms;
            izip!(out.row_f32_mut(i)?, row, w.as_f32()?)
                .for_each(|(out_val, &x, &w_val)| *out_val = x * scale * w_val);
        }
        Ok(())
    }

    fn rope(
        &self,
        t: &Matrix,
        table: &RopeTable,
        m_start: usize,
        out: &mut Matrix,
    ) -> Result<(), OpsError> {
        check_rope(t, table, m_start, out)?;
        let m_end = t.shape()[0] + m_start;
        for m in m_start..m_end {
            let i = m - m_start;
            let data_row = t.row_f32(i)?;
            let (data_left, data_right) = data_row.split_at(data_row.len() / 2);
            let cos_row = table.cos.row_f32(m)?;
            let sin_row = table.sin.row_f32(m)?;
            let out_row = out.row_f32_mut(i)?;
            let (out_left, out_right) = out_row.split_at_mut(out_row.len() / 2);
            izip!(out_left, data_left, data_right, cos_row, sin_row)
                .for_each(|(out_val, l, r, cos, sin)| *out_val = cos * l - sin * r);
            izip!(out_right, data_left, data_right, cos_row, sin_row)
                .for_each(|(out_val, l, r, cos, sin)| *out_val = sin * l + cos * r);
        }
        Ok(())
    }

    fn softmax(&self, t: &Matrix, out: &mut Matrix) -> Result<(), OpsError> {
        check_same_shape(&[t, out])?;
        let m = t.shape()[0];
        for i in 0..m {
            let data_row = t.row_f32(i)?;
            let max = data_row
                .iter()
                .copied()
                .reduce(f32::max)
                .unwrap_or_default();
            if max == f32::NEG_INFINITY {
                out.row_f32_mut(i)?.iter_mut().for_each(|y| *y = f32::NAN);
            } else {
                data_row
                    .iter()
                    .zip(out.row_f32_mut(i)?)
                    .for_each(|(x, y)| *y = (x - max).exp());
                let sum: f32 = out.row_f32(i)?.iter().sum();
                out.row_f32_mut(i)?.iter_mut().for_each(|y| *y /= sum);
            }
        }
        Ok(())
    }

    fn silu<const R: usize>(&self, t: &Tensor<R>, out: &mut Tensor<R>) -> Result<(), OpsError> {
        check_same_shape(&[t, out])?;
        t.as_f32()?
            .iter()
            .zip(out.as_mut_f32()?)
            .for_each(|(x, y)| *y = x / (1.0 + (-1.0 * x).exp()));
        Ok(())
    }

    fn embedding_lookup(
        &self,
        ids: &[u32],
        embed: &Matrix,
        out: &mut Matrix,
    ) -> Result<(), OpsError> {
        check_embedding_lookup(ids, embed, out)?;
        for (i, id) in ids.iter().enumerate() {
            let ref_row = embed.row_f32(*id as usize)?;
            out.row_f32_mut(i)?.copy_from_slice(ref_row);
        }
        Ok(())
    }
}

// Helper functions
fn check_matmul(a: &Matrix, b: &Matrix, out: &Matrix) -> Result<(), OpsError> {
    if a.shape()[1] != b.shape()[1] {
        return Err(OpsError::ShapeMismatch);
    }
    if a.shape()[0] != out.shape()[0] {
        return Err(OpsError::ShapeMismatch);
    }
    if b.shape()[0] != out.shape()[1] {
        return Err(OpsError::ShapeMismatch);
    }
    Ok(())
}

fn check_rmsnorm(t: &Matrix, w: &Vector, out: &Matrix) -> Result<(), OpsError> {
    check_same_shape(&[t, out])?;
    if t.shape()[1] != w.shape()[0] {
        return Err(OpsError::ShapeMismatch);
    }
    Ok(())
}

fn check_rope(t: &Matrix, table: &RopeTable, m_start: usize, out: &Matrix) -> Result<(), OpsError> {
    check_same_shape(&[&table.cos, &table.sin])?;
    check_same_shape(&[t, out])?;
    if t.shape()[1] != 2 * table.cos.shape()[1] {
        return Err(OpsError::ShapeMismatch);
    }
    if table.cos.shape()[0] < m_start + t.shape()[0] {
        return Err(OpsError::ShapeMismatch);
    }
    Ok(())
}

fn check_same_shape<const R: usize>(tensors: &[&Tensor<R>]) -> Result<(), OpsError> {
    for i in 1..tensors.len() {
        if tensors[i - 1].shape() != tensors[i].shape() {
            return Err(OpsError::ShapeMismatch);
        }
    }
    Ok(())
}

fn check_embedding_lookup(ids: &[u32], embed: &Matrix, out: &Matrix) -> Result<(), OpsError> {
    if &[ids.len(), embed.shape()[1]] != out.shape() {
        return Err(OpsError::ShapeMismatch);
    }
    if embed.shape()[0] <= ids.iter().cloned().max().unwrap_or_default() as usize {
        return Err(OpsError::ShapeMismatch);
    }
    Ok(())
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    izip!(a, b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor::{Dtype, Storage};
    use testutil::{DEFAULT_TOL, assert_close};

    fn mat(shape: [usize; 2], data: &[f32]) -> Matrix {
        Matrix::new(shape, Dtype::F32, Storage::Heap(data.to_vec())).unwrap()
    }

    fn vector(data: &[f32]) -> Vector {
        Vector::new([data.len()], Dtype::F32, Storage::Heap(data.to_vec())).unwrap()
    }

    fn data<const R: usize>(t: &Tensor<R>) -> Vec<f32> {
        t.as_f32().unwrap().to_vec()
    }

    /// Asserts that an op refused its arguments because of their shapes.
    #[track_caller]
    fn assert_shape_mismatch(result: Result<(), OpsError>) {
        assert!(
            matches!(result, Err(OpsError::ShapeMismatch)),
            "expected Err(ShapeMismatch), got {result:?}"
        );
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
        let mut out = Matrix::zeros_f32([2, 4]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out).unwrap();

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
        let mut out = Matrix::zeros_f32([1, 4]);
        CpuBackend {}.matmul(&a, &b_4x3(), &mut out).unwrap();

        assert_eq!(out.shape(), &[1, 4]);
        assert_close(&data(&out), &[6.0, -6.5, 6.0, 2.0], DEFAULT_TOL);
    }

    #[test]
    fn matmul_single_column() {
        let b = mat([1, 3], &[4.0, 0.0, -2.0]);
        let mut out = Matrix::zeros_f32([2, 1]);
        CpuBackend {}.matmul(&a_2x3(), &b, &mut out).unwrap();

        assert_eq!(out.shape(), &[2, 1]);
        assert_close(&data(&out), &[6.0, 1.0], DEFAULT_TOL);
    }

    /// k = 1 degenerates the dot product to a single multiply.
    #[test]
    fn matmul_k_of_one() {
        let a = mat([2, 1], &[3.0, -2.0]);
        let b = mat([3, 1], &[1.0, 0.5, -4.0]);
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.matmul(&a, &b, &mut out).unwrap();

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
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out).unwrap();

        assert_close(
            &data(&out),
            &[6.0, -6.5, 6.0, 2.0, 1.0, 8.0, -10.5, 6.5],
            DEFAULT_TOL,
        );
    }

    // Rejection cases. Each one breaks exactly one shape rule and satisfies the
    // rest, so the error can only come from the check under test.
    //
    // Shape/data disagreement and wrong rank are no longer runtime cases:
    // `Matrix::new` rejects the former (tested in the tensor crate) and the
    // type system rejects the latter.

    #[test]
    fn matmul_rejects_k_mismatch() {
        let b = Matrix::zeros_f32([4, 7]);
        let mut out = Matrix::zeros_f32([2, 4]);
        assert_shape_mismatch(CpuBackend {}.matmul(&a_2x3(), &b, &mut out));
    }

    #[test]
    fn matmul_rejects_wrong_out_rows() {
        let mut out = Matrix::zeros_f32([5, 4]);
        assert_shape_mismatch(CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out));
    }

    #[test]
    fn matmul_rejects_wrong_out_cols() {
        let mut out = Matrix::zeros_f32([2, 9]);
        assert_shape_mismatch(CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out));
    }

    #[test]
    fn rmsnorm_valid() {
        let t = a_2x3();
        let w = vector(&[1.0, 2.0, 0.5]);
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out).unwrap();

        for i in 0..2 {
            let row = out.row_f32(i).unwrap();
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
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out).unwrap();

        assert!(
            out.as_f32().unwrap().iter().all(|x| x.is_finite()),
            "got {:?}",
            data(&out)
        );
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
        let mut out = Matrix::zeros_f32([1, 3]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out).unwrap();
        assert_close(&data(&out), &[1.0392302, -2.7712806, 0.0], DEFAULT_TOL);
    }

    #[test]
    fn rmsnorm_overwrites_every_element_of_out() {
        let t = a_2x3();
        let w = vector(&[1.0, 2.0, 0.5]);
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.rmsnorm(&t, &w, 1e-6, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[
                1.0392302, -2.7712806, 0.0, // row 0
                -0.2553769, 3.064523, -0.3830654, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    // Rejection cases. Each one breaks exactly one shape rule and satisfies the
    // rest, so the error can only come from the check under test.

    #[test]
    fn rmsnorm_rejects_weight_length_mismatch() {
        let w = vector(&[1.0; 7]); // must match t's last dim, 3
        let mut out = Matrix::zeros_f32([2, 3]);
        assert_shape_mismatch(CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out));
    }

    #[test]
    fn rmsnorm_rejects_out_shape_mismatch() {
        let w = vector(&[1.0; 3]);
        let mut out = Matrix::zeros_f32([3, 2]); // same element count, wrong shape
        assert_shape_mismatch(CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out));
    }

    #[test]
    fn rope_valid_using_vector() {
        const HEAD_DIM: usize = 4;
        let t = mat([1, HEAD_DIM], &[0.497, -0.138, 0.648, 1.523]);
        let mut out = Matrix::zeros_f32([1, HEAD_DIM]);
        let table = RopeTable::new(HEAD_DIM, 10, 10000.0);
        // M = 0
        CpuBackend {}.rope(&t, &table, 0, &mut out).unwrap();
        assert_close(&data(&out), &[0.497, -0.138, 0.648, 1.523], DEFAULT_TOL);
        // M = 1
        CpuBackend {}.rope(&t, &table, 1, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[-0.276743, -0.153223, 0.768327, 1.521544],
            DEFAULT_TOL,
        );
        // M = 2
        CpuBackend {}.rope(&t, &table, 2, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[-0.796050, -0.168430, 0.182258, 1.519936],
            DEFAULT_TOL,
        );
        // M = 3
        CpuBackend {}.rope(&t, &table, 3, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([NUM_ROWS, HEAD_DIM]);
        let table = RopeTable::new(HEAD_DIM, 10, 10000.0);
        CpuBackend {}.rope(&t, &table, 0, &mut out).unwrap();
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
        mat([1, t.num_cols()], t.row_f32(i).unwrap())
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
        let mut batched = Matrix::zeros_f32([4, 8]);
        CpuBackend {}
            .rope(&t, &table, m_start, &mut batched)
            .unwrap();

        for i in 0..4 {
            let mut single = Matrix::zeros_f32([1, 8]);
            CpuBackend {}
                .rope(&row_matrix(&t, i), &table, m_start + i, &mut single)
                .unwrap();
            assert_close(batched.row_f32(i).unwrap(), &data(&single), DEFAULT_TOL);
        }
    }

    /// Each (j, j + n/2) pair is rotated as a 2-D vector, so its length must
    /// not change. This is also a convention check: interleaved RoPE preserves
    /// (2j, 2j + 1) pairs instead, and would fail here.
    #[test]
    fn rope_preserves_pair_lengths() {
        let t = rope_input_4x8();
        let table = RopeTable::new(8, 16, 10000.0);
        let mut out = Matrix::zeros_f32([4, 8]);
        CpuBackend {}.rope(&t, &table, 9, &mut out).unwrap();

        for i in 0..4 {
            let (x, y) = (t.row_f32(i).unwrap(), out.row_f32(i).unwrap());
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
            let mut rq = Matrix::zeros_f32([1, 8]);
            let mut rk = Matrix::zeros_f32([1, 8]);
            CpuBackend {}.rope(&q, &table, m_q, &mut rq).unwrap();
            CpuBackend {}.rope(&k, &table, m_k, &mut rk).unwrap();
            dot_product(rq.row_f32(0).unwrap(), rk.row_f32(0).unwrap())
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
        let mut out = Matrix::zeros_f32([1, 8]);
        CpuBackend {}.rope(&t, &table, 1, &mut out).unwrap();

        // Table row 1, column 1: pair 1 at position 1.
        let (c, s) = (
            table.cos.row_f32(1).unwrap()[1],
            table.sin.row_f32(1).unwrap()[1],
        );
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
        let mut expected = Matrix::zeros_f32([4, 8]);
        CpuBackend {}.rope(&t, &table, 2, &mut expected).unwrap();

        let mut out = mat([4, 8], &[999.0; 32]);
        CpuBackend {}.rope(&t, &table, 2, &mut out).unwrap();
        assert_close(&data(&out), &data(&expected), DEFAULT_TOL);
    }

    /// Decoding at the last table row is allowed: `m_start + rows == max_seq`.
    /// Pins the table bound as inclusive of its last row.
    #[test]
    fn rope_accepts_last_table_row() {
        let t = row_matrix(&rope_input_4x8(), 0);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros_f32([1, 8]);
        CpuBackend {}.rope(&t, &table, 6, &mut out).unwrap();
        assert!(out.as_f32().unwrap().iter().any(|x| *x != 0.0));
    }

    // Rejection cases. Each one breaks exactly one shape rule and satisfies the
    // rest, so the error can only come from the check under test.

    /// The case you'll actually hit in practice: decode runs past `max_seq`.
    #[test]
    fn rope_rejects_positions_past_table_end() {
        let t = row_matrix(&rope_input_4x8(), 0);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros_f32([1, 8]);
        assert_shape_mismatch(CpuBackend {}.rope(&t, &table, 7, &mut out)); // table rows are 0..=6
    }

    #[test]
    fn rope_rejects_head_dim_table_mismatch() {
        let t = Matrix::zeros_f32([1, 6]);
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros_f32([1, 6]);
        assert_shape_mismatch(CpuBackend {}.rope(&t, &table, 0, &mut out));
    }

    #[test]
    fn rope_rejects_out_shape_mismatch() {
        let t = row_matrix(&rope_input_4x8(), 0); // [1, 8]
        let table = RopeTable::new(8, 7, 10000.0);
        let mut out = Matrix::zeros_f32([8, 1]); // same element count, wrong shape
        assert_shape_mismatch(CpuBackend {}.rope(&t, &table, 0, &mut out));
    }

    #[test]
    fn softmax_valid_using_vector() {
        let t = mat([1, 3], &[1.0, 2.0, 3.0]);
        let mut out = Matrix::zeros_f32([1, 3]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([5, 6]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([4, 8]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
        for i in 0..4 {
            let row = out.row_f32(i).unwrap();
            assert!(row.iter().all(|p| *p > 0.0 && *p < 1.0), "row {i}: {row:?}");
            assert_close(&[row.iter().sum::<f32>()], &[1.0], DEFAULT_TOL);
        }
    }

    /// Softmax is monotonic: a larger input always gets a larger probability.
    /// Catches a flipped exponent such as `exp(max - x)`.
    #[test]
    fn softmax_preserves_order_within_a_row() {
        let t = rope_input_4x8();
        let mut out = Matrix::zeros_f32([4, 8]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
        for i in 0..4 {
            let (x, y) = (t.row_f32(i).unwrap(), out.row_f32(i).unwrap());
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
        let mut out = Matrix::zeros_f32([3, 3]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();

        let reference = [0.09003057, 0.24472847, 0.66524096];
        for i in 0..3 {
            assert_close(out.row_f32(i).unwrap(), &reference, DEFAULT_TOL);
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
        let mut out = Matrix::zeros_f32([3, 3]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();

        // Exact, not approximate: a masked position must contribute nothing.
        for coords in [[0, 1], [0, 2], [1, 2]] {
            assert_eq!(
                out.row_f32(coords[0]).unwrap()[coords[1]],
                0.0,
                "masked {coords:?}"
            );
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
        let mut out = Matrix::zeros_f32([3, 1]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
        assert_close(&data(&out), &[1.0, 1.0, 1.0], DEFAULT_TOL);
    }

    /// A NaN input must not be silently dropped. `f32::max` ignores NaN, so
    /// the row max is still finite here; the NaN has to show up through the
    /// sum. PyTorch returns an all-NaN row in this case.
    #[test]
    fn softmax_propagates_nan() {
        let t = mat([2, 3], &[1.0, f32::NAN, 2.0, 1.0, 2.0, 3.0]);
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
        assert!(
            out.row_f32(0).unwrap().iter().all(|p| p.is_nan()),
            "row 0: {:?}",
            out.row_f32(0).unwrap()
        );
        // The NaN must not leak into the next row.
        assert_close(
            out.row_f32(1).unwrap(),
            &[0.09003057, 0.24472847, 0.66524096],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_overwrites_every_element_of_out() {
        let t = rope_input_4x8();
        let mut expected = Matrix::zeros_f32([4, 8]);
        CpuBackend {}.softmax(&t, &mut expected).unwrap();

        let mut out = mat([4, 8], &[999.0; 32]);
        CpuBackend {}.softmax(&t, &mut out).unwrap();
        assert_close(&data(&out), &data(&expected), DEFAULT_TOL);
    }

    #[test]
    fn softmax_rejects_out_shape_mismatch() {
        let t = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros_f32([3, 2]); // same element count, wrong shape
        assert_shape_mismatch(CpuBackend {}.softmax(&t, &mut out));
    }

    #[test]
    fn silu_valid_using_vector() {
        let t = vector(&[0.0, 1.0, -1.0]);
        let mut out = Vector::zeros_f32([3]);
        CpuBackend {}.silu(&t, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([4, 6]);
        CpuBackend {}.silu(&t, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([1, 6]);
        CpuBackend {}.silu(&t, &mut out).unwrap();

        assert!(
            out.as_f32().unwrap().iter().all(|y| y.is_finite()),
            "{:?}",
            data(&out)
        );
        // Large positive passes through; large negative decays to zero.
        assert_close(
            &data(&out),
            &[100.0, 0.0, 88.0, 0.0, 20.0, 0.0],
            DEFAULT_TOL,
        );
    }

    /// SiLU is not monotonic: it dips to about -0.2785 near x = -1.2785, then
    /// climbs back toward 0. Points further out in either direction must sit
    /// above the dip.
    #[test]
    fn silu_has_a_minimum_near_negative_1_2785() {
        let t = mat([1, 5], &[-6.0, -3.0, -1.2785, -0.5, -0.1]);
        let mut out = Matrix::zeros_f32([1, 5]);
        CpuBackend {}.silu(&t, &mut out).unwrap();

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
        let mut forward_out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.silu(&forward_in, &mut forward_out).unwrap();

        let mut reversed: Vec<f32> = data(&forward_in);
        reversed.reverse();
        let reversed_in = mat([2, 3], &reversed);
        let mut reversed_out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.silu(&reversed_in, &mut reversed_out).unwrap();

        let mut expected = data(&forward_out);
        expected.reverse();
        assert_close(&data(&reversed_out), &expected, DEFAULT_TOL);
    }

    #[test]
    fn silu_overwrites_every_element_of_out() {
        let t = mat([2, 3], &[-2.0, -0.5, 0.0, 0.75, 1.5, 3.0]);
        let mut expected = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.silu(&t, &mut expected).unwrap();

        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.silu(&t, &mut out).unwrap();
        assert_close(&data(&out), &data(&expected), DEFAULT_TOL);
    }

    #[test]
    fn silu_rejects_out_shape_mismatch() {
        let t = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros_f32([3, 2]); // same element count, wrong shape
        assert_shape_mismatch(CpuBackend {}.silu(&t, &mut out));
    }

    #[test]
    fn add_valid_using_vector() {
        let a = vector(&[1.0, -2.0, 0.5]);
        let b = vector(&[0.25, 2.0, -1.5]);
        let mut out = Vector::zeros_f32([3]);
        CpuBackend {}.add(&a, &b, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([3, 4]);
        CpuBackend {}.add(&a, &b, &mut out).unwrap();
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
        let mut ab = Matrix::zeros_f32([2, 3]);
        let mut ba = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.add(&a, &b, &mut ab).unwrap();
        CpuBackend {}.add(&b, &a, &mut ba).unwrap();
        assert_close(&data(&ab), &data(&ba), DEFAULT_TOL);
    }

    /// Adding zeros must leave the input untouched.
    #[test]
    fn add_zero_is_identity() {
        let a = mat([2, 3], &[1.0, -2.0, 0.5, 3.25, -0.125, 0.0]);
        let zeros = Matrix::zeros_f32([2, 3]);
        let mut out = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.add(&a, &zeros, &mut out).unwrap();
        assert_close(&data(&out), &data(&a), DEFAULT_TOL);
    }

    #[test]
    fn add_overwrites_every_element_of_out() {
        let a = mat([2, 3], &[1.0, -2.0, 0.5, 3.25, -0.125, 0.0]);
        let b = mat([2, 3], &[0.25, 2.0, -1.5, -3.25, 8.0, 4.0]);
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.add(&a, &b, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[1.25, 0.0, -1.0, 0.0, 7.875, 4.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn add_rejects_mismatched_input_shapes() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([3, 2], &[1.0; 6]); // same element count, wrong shape
        let mut out = Matrix::zeros_f32([2, 3]);
        assert_shape_mismatch(CpuBackend {}.add(&a, &b, &mut out));
    }

    #[test]
    fn add_rejects_out_shape_mismatch() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros_f32([6, 1]);
        assert_shape_mismatch(CpuBackend {}.add(&a, &b, &mut out));
    }

    #[test]
    fn hadamard_product_valid_using_vector() {
        let a = vector(&[2.0, -3.0, 0.5]);
        let b = vector(&[0.25, 0.5, -4.0]);
        let mut out = Vector::zeros_f32([3]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out).unwrap();
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
        let mut out = Matrix::zeros_f32([3, 4]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out).unwrap();
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
        let mut ab = Matrix::zeros_f32([2, 3]);
        let mut ba = Matrix::zeros_f32([2, 3]);
        CpuBackend {}.hadamard_product(&a, &b, &mut ab).unwrap();
        CpuBackend {}.hadamard_product(&b, &a, &mut ba).unwrap();
        assert_close(&data(&ab), &data(&ba), DEFAULT_TOL);
    }

    /// Multiplying by ones leaves the input untouched; multiplying by zeros
    /// erases it. Together these pin that it is elementwise and not a matrix
    /// product, which would not satisfy either for non-square shapes.
    #[test]
    fn hadamard_product_ones_and_zeros() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 0.0]);
        let mut out = Matrix::zeros_f32([2, 3]);

        CpuBackend {}
            .hadamard_product(&a, &mat([2, 3], &[1.0; 6]), &mut out)
            .unwrap();
        assert_close(&data(&out), &data(&a), DEFAULT_TOL);

        CpuBackend {}
            .hadamard_product(&a, &Matrix::zeros_f32([2, 3]), &mut out)
            .unwrap();
        assert_close(&data(&out), &[0.0; 6], DEFAULT_TOL);
    }

    /// Output `i` must depend only on input `i`, so reversing both inputs must
    /// reverse the output. Catches an index mistake that symmetric data hides.
    #[test]
    fn hadamard_product_is_elementwise() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 4.0]);
        let b = mat([2, 3], &[0.25, 0.5, -4.0, 8.0, -2.0, 3.0]);
        let mut forward = Matrix::zeros_f32([2, 3]);
        CpuBackend {}
            .hadamard_product(&a, &b, &mut forward)
            .unwrap();

        let rev = |t: &Matrix| {
            let mut d = data(t);
            d.reverse();
            mat([2, 3], &d)
        };
        let mut reversed = Matrix::zeros_f32([2, 3]);
        CpuBackend {}
            .hadamard_product(&rev(&a), &rev(&b), &mut reversed)
            .unwrap();

        let mut expected = data(&forward);
        expected.reverse();
        assert_close(&data(&reversed), &expected, DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_overwrites_every_element_of_out() {
        let a = mat([2, 3], &[2.0, -3.0, 0.5, 1.25, -0.5, 0.0]);
        let b = mat([2, 3], &[0.25, 0.5, -4.0, 8.0, -2.0, 3.0]);
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}.hadamard_product(&a, &b, &mut out).unwrap();
        assert_close(&data(&out), &[0.5, -1.5, -2.0, 10.0, 1.0, 0.0], DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_rejects_mismatched_input_shapes() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([3, 2], &[1.0; 6]);
        let mut out = Matrix::zeros_f32([2, 3]);
        assert_shape_mismatch(CpuBackend {}.hadamard_product(&a, &b, &mut out));
    }

    #[test]
    fn hadamard_product_rejects_out_shape_mismatch() {
        let a = mat([2, 3], &[1.0; 6]);
        let b = mat([2, 3], &[1.0; 6]);
        let mut out = Matrix::zeros_f32([6, 1]);
        assert_shape_mismatch(CpuBackend {}.hadamard_product(&a, &b, &mut out));
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
        let mut out = Matrix::zeros_f32([4, 3]);
        CpuBackend {}
            .embedding_lookup(&ids, &embed, &mut out)
            .unwrap();
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
        let mut out = Matrix::zeros_f32([3, 4]);
        CpuBackend {}
            .embedding_lookup(&ids, &embed, &mut out)
            .unwrap();
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
        let mut out = Matrix::zeros_f32([1, 3]);
        CpuBackend {}
            .embedding_lookup(&[2u32], &embed, &mut out)
            .unwrap();
        assert_close(&data(&out), &[20.0, 21.0, 22.0], DEFAULT_TOL);
    }

    /// A repeated token copies the same row again; no deduplication.
    #[test]
    fn embedding_lookup_repeats_rows() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros_f32([3, 3]);
        CpuBackend {}
            .embedding_lookup(&[2u32, 2, 2], &embed, &mut out)
            .unwrap();
        assert_close(
            &data(&out),
            &[20.0, 21.0, 22.0, 20.0, 21.0, 22.0, 20.0, 21.0, 22.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn embedding_lookup_empty_ids() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros_f32([0, 3]);
        CpuBackend {}
            .embedding_lookup(&[], &embed, &mut out)
            .unwrap();
        assert_eq!(out.as_f32().unwrap().iter().count(), 0);
    }

    #[test]
    fn embedding_lookup_overwrites_every_element_of_out() {
        let embed = embed_5x3();
        let mut out = mat([2, 3], &[999.0; 6]);
        CpuBackend {}
            .embedding_lookup(&[1u32, 0], &embed, &mut out)
            .unwrap();
        assert_close(&data(&out), &[10.0, 11.0, 12.0, 0.0, 1.0, 2.0], DEFAULT_TOL);
    }

    /// An id at or past `vocab_size` means a tokenizer/vocab mismatch and must
    /// be rejected.
    #[test]
    fn embedding_lookup_rejects_id_past_vocab() {
        let table: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let embed = mat([5, 8], &table); // vocab 5, hidden 8
        let mut out = Matrix::zeros_f32([1, 8]);
        let result = CpuBackend {}.embedding_lookup(&[5u32], &embed, &mut out); // ids are 0..=4
        assert!(result.is_err(), "got {result:?}");
    }

    #[test]
    fn embedding_lookup_rejects_out_shape_mismatch() {
        let embed = embed_5x3();
        let mut out = Matrix::zeros_f32([3, 2]); // should be [2, 3]
        assert_shape_mismatch(CpuBackend {}.embedding_lookup(&[1u32, 0], &embed, &mut out));
    }
}
