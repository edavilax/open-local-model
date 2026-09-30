use crate::{Backend, OpsError, RopeTable};
use anyhow::Result;
use itertools::izip;
use tensor::Tensor;

pub struct CpuBackend {}

impl Backend for CpuBackend {
    fn add(&self, y: &mut Tensor, x: &Tensor) -> Result<()> {
        check_same_shape("add x must match y", &[y, x])?;
        izip!(y.as_mut_f32()?, x.as_f32()?).for_each(|(y_val, x_val)| *y_val += x_val);
        Ok(())
    }

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        check_matmul(a, b, out)?;
        let k = a.dim(1)?;
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

    fn hadamard_product(&self, y: &mut Tensor, x: &Tensor) -> Result<()> {
        check_same_shape("hadamard_product x must match y", &[y, x])?;
        izip!(y.as_mut_f32()?, x.as_f32()?).for_each(|(y_val, x_val)| *y_val *= x_val);
        Ok(())
    }

    fn rmsnorm(&self, t: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()> {
        check_rmsnorm(t, w, out)?;
        let n = t.dim(1)?;
        for (t_row, out_row) in izip!(
            t.as_f32()?.chunks_exact(n),
            out.as_mut_f32()?.chunks_exact_mut(n)
        ) {
            let sq: f32 = t_row.iter().map(|x| x * x).sum();
            let cnt = n as f32;
            let mean = sq / cnt + eps;
            let rms = f32::sqrt(mean);
            let scale = 1.0 / rms;
            izip!(out_row, t_row, w.as_f32()?)
                .for_each(|(out_val, x, w_val)| *out_val = x * scale * w_val);
        }
        Ok(())
    }

    fn rope(&self, y: &mut Tensor, table: &RopeTable, m_start: usize) -> Result<()> {
        check_rope(y, table, m_start)?;
        let num_heads = y.dim(1)?;
        let num_head_elems = y.dim(2)?;
        let num_head_half = num_head_elems / 2;
        let num_row_elems = num_heads * num_head_elems;
        for (y_row, cos_row, sin_row) in izip!(
            y.as_mut_f32()?.chunks_exact_mut(num_row_elems),
            table
                .cos
                .as_f32()?
                .chunks_exact(num_head_half)
                .skip(m_start),
            table
                .sin
                .as_f32()?
                .chunks_exact(num_head_half)
                .skip(m_start),
        ) {
            for y_head in izip!(y_row.chunks_exact_mut(num_head_elems)) {
                let (y_left, y_right) = y_head.split_at_mut(num_head_half);
                izip!(y_left, y_right, cos_row, sin_row).for_each(|(l, r, cos, sin)| {
                    let l_orig = *l;
                    *l = *cos * *l - *sin * *r;
                    *r = *sin * l_orig + *cos * *r;
                });
            }
        }
        Ok(())
    }

    fn softmax(&self, y: &mut Tensor) -> Result<()> {
        check_rank(y, "softmax y", 2)?;
        let n = y.dim(1)?;
        if n == 0 {
            return Ok(());
        }
        for row in y.as_mut_f32()?.chunks_exact_mut(n) {
            let max = row.iter().copied().reduce(f32::max).unwrap_or_default();
            if max == f32::NEG_INFINITY {
                row.fill(f32::NAN);
            } else {
                let mut sum: f32 = 0.0;
                for x in row.iter_mut() {
                    *x = (*x - max).exp();
                    sum += *x;
                }
                row.iter_mut().for_each(|x| *x /= sum);
            }
        }
        Ok(())
    }

    fn silu(&self, y: &mut Tensor) -> Result<()> {
        y.as_mut_f32()?
            .iter_mut()
            .for_each(|y| *y = *y / (1.0 + (-1.0 * *y).exp()));
        Ok(())
    }

    fn embedding_lookup(&self, ids: &[u32], embed: &Tensor, out: &mut Tensor) -> Result<()> {
        check_embedding_lookup(ids, embed, out)?;
        let n = embed.dim(1)?;
        if n == 0 {
            return Ok(());
        }
        let embed_data = embed.as_f32()?;
        for (id, out_row) in izip!(ids, out.as_mut_f32()?.chunks_exact_mut(n)) {
            let start = *id as usize * n;
            out_row.copy_from_slice(&embed_data[start..start + n]);
        }
        Ok(())
    }
}

// Helper functions
fn check_matmul(a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
    check_rank(a, "matmul a", 2)?;
    check_rank(b, "matmul b", 2)?;
    check_rank(out, "matmul out", 2)?;
    if a.dim(1)? != b.dim(1)? {
        return Err(shape_mismatch(
            "matmul b must be [n, k] with k from a",
            &[b.dim(0)?, a.dim(1)?],
            b.shape(),
        ));
    }
    if a.dim(0)? != out.dim(0)? || b.dim(0)? != out.dim(1)? {
        return Err(shape_mismatch(
            "matmul out must be [a rows, b rows]",
            &[a.dim(0)?, b.dim(0)?],
            out.shape(),
        ));
    }
    Ok(())
}

fn check_rmsnorm(t: &Tensor, w: &Tensor, out: &Tensor) -> Result<()> {
    check_rank(t, "rmsnorm t", 2)?;
    check_rank(w, "rmsnorm w", 1)?;
    let n0 = t.dim(1)?;
    let n1 = w.dim(0)?;
    out.dim(1)?;
    check_same_shape("rmsnorm out must match t", &[t, out])?;
    if n0 != n1 {
        return Err(shape_mismatch(
            "rmsnorm w must match the row width of t",
            &[n0],
            w.shape(),
        ));
    }
    Ok(())
}

fn check_rope(y: &Tensor, table: &RopeTable, m_start: usize) -> Result<()> {
    check_rank(y, "rope y", 3)?;
    check_same_shape("rope table sin must match cos", &[&table.cos, &table.sin])?;
    if y.dim(2)? != 2 * table.cos.dim(1)? {
        return Err(shape_mismatch(
            "rope y head_dim must match the table",
            &[2 * table.cos.dim(1)?],
            &[y.dim(2)?],
        ));
    }
    if table.cos.dim(0)? < m_start + y.dim(0)? {
        return Err(shape_mismatch(
            "rope positions m_start + rows must fit in the table",
            &[table.cos.dim(0)?],
            &[m_start + y.dim(0)?],
        ));
    }
    Ok(())
}

fn check_same_shape(desc: &str, tensors: &[&Tensor]) -> Result<()> {
    for i in 1..tensors.len() {
        if tensors[i - 1].shape() != tensors[i].shape() {
            return Err(shape_mismatch(
                desc,
                tensors[i - 1].shape(),
                tensors[i].shape(),
            ));
        }
    }
    Ok(())
}

fn check_rank(x: &Tensor, name: &str, rank: usize) -> Result<()> {
    if x.shape().len() != rank {
        return Err(OpsError::RankMismatch {
            name: name.to_string(),
            expected: rank,
            actual: x.shape().len(),
        }
        .into());
    }
    Ok(())
}

fn check_embedding_lookup(ids: &[u32], embed: &Tensor, out: &Tensor) -> Result<()> {
    check_rank(embed, "embedding_lookup embed", 2)?;
    check_rank(out, "embedding_lookup out", 2)?;
    let (vocab, n) = (embed.dim(0)?, embed.dim(1)?);
    if (out.dim(0)?, out.dim(1)?) != (ids.len(), n) {
        return Err(shape_mismatch(
            "embedding_lookup out must be [ids, embed columns]",
            &[ids.len(), n],
            out.shape(),
        ));
    }
    if let Some(id) = ids.iter().find(|id| **id as usize >= vocab) {
        return Err(shape_mismatch(
            "embedding_lookup token id must be below vocab_size",
            &[vocab],
            &[*id as usize],
        ));
    }
    Ok(())
}

fn shape_mismatch(desc: &str, expected: &[usize], actual: &[usize]) -> anyhow::Error {
    OpsError::ShapeMismatch {
        desc: desc.to_string(),
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    }
    .into()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    izip!(a, b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor::{Dtype, Storage};
    use testutil::{DEFAULT_TOL, assert_close, mmap_f32};

    fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), Dtype::F32, Storage::Heap(data.to_vec())).unwrap()
    }

    fn mapped(shape: &[usize], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), Dtype::F32, Storage::Mmap(mmap_f32(data))).unwrap()
    }

    fn data(t: &Tensor) -> Vec<f32> {
        t.as_f32().unwrap().to_vec()
    }

    // Row `i` of dimension 0, with the trailing dimensions flattened.
    fn row(t: &Tensor, i: usize) -> &[f32] {
        let n: usize = t.shape()[1..].iter().product();
        &t.as_f32().unwrap()[i * n..(i + 1) * n]
    }

    #[track_caller]
    fn err(result: Result<()>) -> anyhow::Error {
        match result {
            Ok(()) => panic!("succeeded, but it should have failed"),
            Err(e) => e,
        }
    }

    fn a_2x3() -> Tensor {
        tensor(
            &[2, 3],
            &[
                1.5, -2.0, 0.0, // row 0
                -0.5, 3.0, -1.5, // row 1
            ],
        )
    }

    // Stored [n, k]: each row is a column of the mathematical B.
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

    fn input_4x8() -> Tensor {
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

    // matmul

    #[test]
    fn matmul_valid() {
        let mut out = Tensor::zeros_f32(vec![2, 4]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out).unwrap();
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
        let a = tensor(&[1, 3], &[1.5, -2.0, 0.0]);
        let mut out = Tensor::zeros_f32(vec![1, 4]);
        CpuBackend {}.matmul(&a, &b_4x3(), &mut out).unwrap();
        assert_close(&data(&out), &[6.0, -6.5, 6.0, 2.0], DEFAULT_TOL);
    }

    #[test]
    fn matmul_single_column() {
        let b = tensor(&[1, 3], &[4.0, 0.0, -2.0]);
        let mut out = Tensor::zeros_f32(vec![2, 1]);
        CpuBackend {}.matmul(&a_2x3(), &b, &mut out).unwrap();
        assert_close(&data(&out), &[6.0, 1.0], DEFAULT_TOL);
    }

    #[test]
    fn matmul_k_of_one() {
        let a = tensor(&[2, 1], &[3.0, -2.0]);
        let b = tensor(&[3, 1], &[1.0, 0.5, -4.0]);
        let mut out = Tensor::zeros_f32(vec![2, 3]);
        CpuBackend {}.matmul(&a, &b, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[3.0, 1.5, -12.0, -2.0, -1.0, 8.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn matmul_overwrites_every_element_of_out() {
        let mut out = tensor(&[2, 4], &[999.0; 8]);
        CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut out).unwrap();
        assert_close(
            &data(&out),
            &[6.0, -6.5, 6.0, 2.0, 1.0, 8.0, -10.5, 6.5],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn matmul_rejects_shape_mismatches() {
        let cases = [
            (Tensor::zeros_f32(vec![4, 7]), Tensor::zeros_f32(vec![2, 4])), // k
            (b_4x3(), Tensor::zeros_f32(vec![5, 4])),                       // out rows
            (b_4x3(), Tensor::zeros_f32(vec![2, 9])),                       // out cols
        ];
        for (b, mut out) in cases {
            let e = err(CpuBackend {}.matmul(&a_2x3(), &b, &mut out));
            assert!(
                matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
                "{e}"
            );
        }
    }

    #[test]
    fn matmul_rejects_rank_1() {
        let v = tensor(&[3], &[1.0, 2.0, 3.0]);
        let e = err(CpuBackend {}.matmul(&v, &b_4x3(), &mut Tensor::zeros_f32(vec![2, 4])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
        let e = err(CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut Tensor::zeros_f32(vec![8])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
    }

    // A [T, heads, head_dim] buffer that was never reshaped back. Each shape
    // below lines up with the other operands in its first two dimensions.
    #[test]
    fn matmul_rejects_rank_3() {
        let a3 = Tensor::zeros_f32(vec![2, 3, 1]);
        let e = err(CpuBackend {}.matmul(&a3, &b_4x3(), &mut Tensor::zeros_f32(vec![2, 4])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 3, .. })
            ),
            "{e}"
        );

        let b3 = Tensor::zeros_f32(vec![4, 3, 1]);
        let e = err(CpuBackend {}.matmul(&a_2x3(), &b3, &mut Tensor::zeros_f32(vec![2, 4])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 3, .. })
            ),
            "{e}"
        );

        let e =
            err(CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut Tensor::zeros_f32(vec![2, 4, 1])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 3, .. })
            ),
            "{e}"
        );
    }

    // rmsnorm

    #[test]
    fn rmsnorm_valid() {
        let w = tensor(&[3], &[1.0, 2.0, 0.5]);
        let mut out = Tensor::zeros_f32(vec![2, 3]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[
                1.0392302, -2.7712806, 0.0, // row 0
                -0.2553769, 3.064523, -0.3830654, // row 1
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rmsnorm_output_rows_have_unit_rms() {
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = Tensor::zeros_f32(vec![2, 3]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out).unwrap();
        for i in 0..2 {
            let r = row(&out, i);
            let rms = (r.iter().map(|x| x * x).sum::<f32>() / r.len() as f32).sqrt();
            assert_close(&[rms], &[1.0], DEFAULT_TOL);
        }
    }

    // Without eps this row would be 0/0.
    #[test]
    fn rmsnorm_zero_row_does_not_produce_nan() {
        let x = tensor(&[2, 3], &[0.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
        let w = tensor(&[3], &[1.0; 3]);
        let mut out = Tensor::zeros_f32(vec![2, 3]);
        CpuBackend {}.rmsnorm(&x, &w, 1e-6, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[0.0, 0.0, 0.0, 1.0392304, 1.3856406, 0.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rmsnorm_single_row() {
        let x = tensor(&[1, 3], &[1.5, -2.0, 0.0]);
        let w = tensor(&[3], &[1.0, 2.0, 0.5]);
        let mut out = Tensor::zeros_f32(vec![1, 3]);
        CpuBackend {}.rmsnorm(&x, &w, 1e-6, &mut out).unwrap();
        assert_close(&data(&out), &[1.0392302, -2.7712806, 0.0], DEFAULT_TOL);
    }

    #[test]
    fn rmsnorm_overwrites_every_element_of_out() {
        let w = tensor(&[3], &[1.0, 2.0, 0.5]);
        let mut out = tensor(&[2, 3], &[999.0; 6]);
        CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[1.0392302, -2.7712806, 0.0, -0.2553769, 3.064523, -0.3830654],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rmsnorm_rejects_shape_mismatches() {
        let e = err(CpuBackend {}.rmsnorm(
            &a_2x3(),
            &tensor(&[7], &[1.0; 7]),
            1e-6,
            &mut Tensor::zeros_f32(vec![2, 3]),
        ));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );

        let e = err(CpuBackend {}.rmsnorm(
            &a_2x3(),
            &tensor(&[3], &[1.0; 3]),
            1e-6,
            &mut Tensor::zeros_f32(vec![3, 2]),
        ));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    #[test]
    fn rmsnorm_rejects_wrong_rank() {
        let w = tensor(&[1, 3], &[1.0; 3]);
        let e = err(CpuBackend {}.rmsnorm(&a_2x3(), &w, 1e-6, &mut Tensor::zeros_f32(vec![2, 3])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 2, .. })
            ),
            "{e}"
        );

        let v = tensor(&[3], &[1.0, 2.0, 3.0]);
        let w = tensor(&[3], &[1.0; 3]);
        let e = err(CpuBackend {}.rmsnorm(&v, &w, 1e-6, &mut Tensor::zeros_f32(vec![3])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
    }

    // rope: input is [rows, heads, head_dim]

    fn roped(x: &Tensor, table: &RopeTable, m_start: usize) -> Tensor {
        let mut y = tensor(x.shape(), &data(x));
        CpuBackend {}.rope(&mut y, table, m_start).unwrap();
        y
    }

    fn head(t: &Tensor, i: usize, h: usize) -> &[f32] {
        let (n_heads, head_dim) = (t.shape()[1], t.shape()[2]);
        let start = (i * n_heads + h) * head_dim;
        &t.as_f32().unwrap()[start..start + head_dim]
    }

    #[test]
    fn rope_valid_at_each_position() {
        let x = tensor(&[1, 1, 4], &[0.497, -0.138, 0.648, 1.523]);
        let table = RopeTable::new(4, 10, 10000.0);
        let cases = [
            (0, [0.497, -0.138, 0.648, 1.523]),
            (1, [-0.276743, -0.153223, 0.768327, 1.521544]),
            (2, [-0.796050, -0.168430, 0.182258, 1.519936]),
            (3, [-0.583472, -0.183621, -0.571378, 1.518175]),
        ];
        for (m, expected) in cases {
            assert_close(&data(&roped(&x, &table, m)), &expected, DEFAULT_TOL);
        }
    }

    #[test]
    fn rope_valid_over_rows() {
        let x = tensor(
            &[5, 1, 6],
            &[
                0.497, -0.138, 0.648, 1.523, -0.234, 0.812, // m = 0
                -0.345, 0.781, -0.112, 0.452, 1.104, -0.673, // m = 1
                0.912, -0.543, 0.321, -0.801, 0.219, -0.456, // m = 2
                -0.123, 0.654, -0.987, 0.314, -0.876, 0.543, // m = 3
                0.765, -0.432, 0.198, -0.541, 0.632, -0.879, // m = 4
            ],
        );
        let table = RopeTable::new(6, 10, 10000.0);
        assert_close(
            &data(&roped(&x, &table, 0)),
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

    // Every head in row i sits at position m_start + i. Rotating the whole
    // tensor must equal rotating each head alone at that position.
    #[test]
    fn rope_matches_rotating_each_head_alone() {
        let values: Vec<f32> = (0..72)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
            .collect();
        let x = tensor(&[3, 3, 8], &values);
        let table = RopeTable::new(8, 16, 10000.0);
        let m_start = 2;
        let all = roped(&x, &table, m_start);
        for i in 0..3 {
            for h in 0..3 {
                let alone = roped(&tensor(&[1, 1, 8], head(&x, i, h)), &table, m_start + i);
                assert_close(head(&all, i, h), &data(&alone), DEFAULT_TOL);
            }
        }
    }

    // One-hot at head 1, index 1 (flat 9). Its pair is flat 13, inside the
    // same head. Pairing across the whole 16-wide row would touch flat 1.
    #[test]
    fn rope_keeps_pairs_inside_their_head() {
        let mut values = [0.0; 16];
        values[9] = 1.0;
        let x = tensor(&[1, 2, 8], &values);
        let table = RopeTable::new(8, 4, 10000.0);
        let (c, s) = (row(&table.cos, 1)[1], row(&table.sin, 1)[1]);
        let mut expected = [0.0; 16];
        expected[9] = c;
        expected[13] = s;
        assert_close(&data(&roped(&x, &table, 1)), &expected, DEFAULT_TOL);
    }

    #[test]
    fn rope_one_hot_pairs_j_with_j_plus_half() {
        let mut values = [0.0; 8];
        values[1] = 1.0;
        let x = tensor(&[1, 1, 8], &values);
        let table = RopeTable::new(8, 4, 10000.0);
        let (c, s) = (row(&table.cos, 1)[1], row(&table.sin, 1)[1]);
        assert_close(
            &data(&roped(&x, &table, 1)),
            &[0.0, c, 0.0, 0.0, 0.0, s, 0.0, 0.0],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn rope_preserves_pair_lengths() {
        let x = tensor(&[2, 2, 8], &data(&input_4x8()));
        let table = RopeTable::new(8, 16, 10000.0);
        let y = roped(&x, &table, 9);
        for i in 0..2 {
            for h in 0..2 {
                let (a, b) = (head(&x, i, h), head(&y, i, h));
                for j in 0..4 {
                    let before = a[j] * a[j] + a[j + 4] * a[j + 4];
                    let after = b[j] * b[j] + b[j + 4] * b[j + 4];
                    assert_close(&[after], &[before], DEFAULT_TOL);
                }
            }
        }
    }

    #[test]
    fn rope_dot_product_depends_only_on_relative_position() {
        let q = tensor(&[1, 1, 8], row(&input_4x8(), 0));
        let k = tensor(&[1, 1, 8], row(&input_4x8(), 1));
        let table = RopeTable::new(8, 16, 10000.0);
        let dot = |m_q, m_k| {
            dot_product(
                &data(&roped(&q, &table, m_q)),
                &data(&roped(&k, &table, m_k)),
            )
        };

        let reference = dot(2, 0);
        assert_close(&[dot(5, 3)], &[reference], DEFAULT_TOL);
        assert_close(&[dot(13, 11)], &[reference], DEFAULT_TOL);
        assert!((dot(3, 0) - reference).abs() > 1e-3);
    }

    // q and k share one table but have different head counts.
    #[test]
    fn rope_accepts_any_head_count() {
        let table = RopeTable::new(8, 4, 10000.0);
        for n_heads in [1, 2, 3, 4] {
            let mut y = Tensor::zeros_f32(vec![2, n_heads, 8]);
            CpuBackend {}.rope(&mut y, &table, 0).unwrap();
        }
    }

    #[test]
    fn rope_accepts_last_table_row() {
        let table = RopeTable::new(8, 7, 10000.0);
        let y = roped(&tensor(&[1, 1, 8], row(&input_4x8(), 0)), &table, 6);
        assert!(y.as_f32().unwrap().iter().any(|v| *v != 0.0));
    }

    #[test]
    fn rope_rejects_positions_past_table_end() {
        let table = RopeTable::new(8, 7, 10000.0);
        let e = err(CpuBackend {}.rope(&mut Tensor::zeros_f32(vec![1, 1, 8]), &table, 7));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
        let e = err(CpuBackend {}.rope(&mut Tensor::zeros_f32(vec![3, 1, 8]), &table, 5));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    // 8 heads of width 6 against a table for width 8: a check against the
    // head count instead of the head width would pass.
    #[test]
    fn rope_rejects_head_dim_table_mismatch() {
        let table = RopeTable::new(8, 7, 10000.0);
        let e = err(CpuBackend {}.rope(&mut Tensor::zeros_f32(vec![1, 8, 6]), &table, 0));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    #[test]
    fn rope_rejects_wrong_rank() {
        let table = RopeTable::new(8, 4, 10000.0);
        let e = err(CpuBackend {}.rope(&mut Tensor::zeros_f32(vec![1, 8]), &table, 0));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 2, .. })
            ),
            "{e}"
        );
        let e = err(CpuBackend {}.rope(&mut Tensor::zeros_f32(vec![8]), &table, 0));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
    }

    #[test]
    fn rope_refuses_a_mapped_tensor() {
        let table = RopeTable::new(8, 4, 10000.0);
        let e = err(CpuBackend {}.rope(&mut mapped(&[1, 1, 8], &[0.0; 8]), &table, 0));
        assert!(e.to_string().contains("read-only"), "{e}");
    }

    // softmax

    fn softmaxed(x: &Tensor) -> Tensor {
        let mut y = tensor(x.shape(), &data(x));
        CpuBackend {}.softmax(&mut y).unwrap();
        y
    }

    #[test]
    fn softmax_valid_single_row() {
        let y = softmaxed(&tensor(&[1, 3], &[1.0, 2.0, 3.0]));
        assert_close(
            &data(&y),
            &[0.09003057, 0.24472847, 0.66524096],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_valid_over_rows() {
        let x = tensor(
            &[5, 6],
            &[
                0.5, -1.2, 3.3, 0.0, 2.1, -0.7, // row 0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // row 1
                -2.0, -3.5, -0.25, -1.0, -4.0, -0.5, // row 2
                1e+01, 9.5, 8.0, 10.5, 7.25, 9.0, // row 3
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, // row 4
            ],
        );
        assert_close(
            &data(&softmaxed(&x)),
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

    #[test]
    fn softmax_rows_sum_to_one() {
        let y = softmaxed(&input_4x8());
        for i in 0..4 {
            let r = row(&y, i);
            assert!(r.iter().all(|p| *p > 0.0 && *p < 1.0), "row {i}: {r:?}");
            assert_close(&[r.iter().sum::<f32>()], &[1.0], DEFAULT_TOL);
        }
    }

    #[test]
    fn softmax_preserves_order_within_a_row() {
        let x = input_4x8();
        let y = softmaxed(&x);
        for i in 0..4 {
            let (a, b) = (row(&x, i), row(&y, i));
            for j in 0..8 {
                for k in 0..8 {
                    if a[j] < a[k] {
                        assert!(b[j] < b[k], "row {i}: x[{j}] < x[{k}] but y[{j}] >= y[{k}]");
                    }
                }
            }
        }
    }

    // exp overflows f32 near 88.7 unless the row max is subtracted first.
    #[test]
    fn softmax_is_shift_invariant_and_stable_for_large_logits() {
        let x = tensor(
            &[3, 3],
            &[
                1.0, 2.0, 3.0, 1000.0, 1001.0, 1002.0, -1000.0, -999.0, -998.0,
            ],
        );
        let y = softmaxed(&x);
        for i in 0..3 {
            assert_close(
                row(&y, i),
                &[0.09003057, 0.24472847, 0.66524096],
                DEFAULT_TOL,
            );
        }
    }

    #[test]
    fn softmax_causal_mask() {
        let ninf = f32::NEG_INFINITY;
        let x = tensor(&[3, 3], &[0.5, ninf, ninf, 1.0, 2.0, ninf, 1.0, 2.0, 3.0]);
        let y = softmaxed(&x);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            assert_eq!(row(&y, i)[j], 0.0, "masked ({i}, {j})");
        }
        assert_close(
            &data(&y),
            &[
                1.0, 0.0, 0.0, 0.26894142, 0.7310586, 0.0, 0.09003057, 0.24472847, 0.66524096,
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_single_column_is_one() {
        let y = softmaxed(&tensor(&[3, 1], &[-5.0, 0.0, 42.0]));
        assert_close(&data(&y), &[1.0, 1.0, 1.0], DEFAULT_TOL);
    }

    #[test]
    fn softmax_propagates_nan() {
        let y = softmaxed(&tensor(&[2, 3], &[1.0, f32::NAN, 2.0, 1.0, 2.0, 3.0]));
        assert!(
            row(&y, 0).iter().all(|p| p.is_nan()),
            "row 0: {:?}",
            row(&y, 0)
        );
        assert_close(
            row(&y, 1),
            &[0.09003057, 0.24472847, 0.66524096],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn softmax_rejects_wrong_rank() {
        let e = err(CpuBackend {}.softmax(&mut tensor(&[3], &[1.0, 2.0, 3.0])));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
    }

    #[test]
    fn softmax_refuses_a_mapped_tensor() {
        let e = err(CpuBackend {}.softmax(&mut mapped(&[1, 3], &[1.0, 2.0, 3.0])));
        assert!(e.to_string().contains("read-only"), "{e}");
    }

    // silu

    fn silued(x: &Tensor) -> Tensor {
        let mut y = tensor(x.shape(), &data(x));
        CpuBackend {}.silu(&mut y).unwrap();
        y
    }

    #[test]
    fn silu_valid() {
        let x = tensor(
            &[4, 6],
            &[
                -1.2785, -1.0, -0.5, 0.0, 0.5, 1.0, // row 0
                2.0, 3.0, -2.0, -3.0, 4.0, -4.0, // row 1
                0.1, -0.1, 0.25, -0.25, 10.0, -10.0, // row 2
                5.5, -5.5, 0.75, -0.75, 1.5, -1.5, // row 3
            ],
        );
        assert_close(
            &data(&silued(&x)),
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

    #[test]
    fn silu_accepts_a_vector() {
        let y = silued(&tensor(&[3], &[0.0, 1.0, -1.0]));
        assert_close(&data(&y), &[0.0, 0.7310586, -0.26894143], DEFAULT_TOL);
    }

    // exp(-x) overflows for x below about -88.7; x / inf is the correct limit, 0.
    #[test]
    fn silu_handles_large_magnitudes() {
        let y = silued(&tensor(&[1, 6], &[100.0, -100.0, 88.0, -88.0, 20.0, -20.0]));
        assert!(
            y.as_f32().unwrap().iter().all(|v| v.is_finite()),
            "{:?}",
            data(&y)
        );
        assert_close(&data(&y), &[100.0, 0.0, 88.0, 0.0, 20.0, 0.0], DEFAULT_TOL);
    }

    #[test]
    fn silu_has_a_minimum_near_negative_1_2785() {
        let y = data(&silued(&tensor(
            &[1, 5],
            &[-6.0, -3.0, -1.2785, -0.5, -0.1],
        )));
        let dip = y[2];
        assert_close(&[dip], &[-0.27846456], DEFAULT_TOL);
        for (i, v) in y.iter().enumerate() {
            if i != 2 {
                assert!(*v > dip, "index {i} = {v} should exceed the dip {dip}");
            }
        }
    }

    #[test]
    fn silu_is_elementwise() {
        let forward = data(&silued(&tensor(
            &[2, 3],
            &[-2.0, -0.5, 0.0, 0.75, 1.5, 3.0],
        )));
        let reversed = data(&silued(&tensor(
            &[2, 3],
            &[3.0, 1.5, 0.75, 0.0, -0.5, -2.0],
        )));
        let mut expected = forward.clone();
        expected.reverse();
        assert_close(&reversed, &expected, DEFAULT_TOL);
    }

    #[test]
    fn silu_refuses_a_mapped_tensor() {
        let e = err(CpuBackend {}.silu(&mut mapped(&[3], &[0.0, 1.0, -1.0])));
        assert!(e.to_string().contains("read-only"), "{e}");
    }

    // add: y += x

    #[test]
    fn add_valid() {
        let mut y = tensor(
            &[3, 4],
            &[
                0.0, 1.0, -1.0, 0.5, // row 0
                2.5, -0.75, 4.0, -8.0, // row 1
                100.0, -0.125, 3.25, 6.0, // row 2
            ],
        );
        let x = tensor(
            &[3, 4],
            &[
                0.0, -1.0, -2.0, 0.25, // row 0
                -2.5, 0.75, 0.5, 8.0, // row 1
                0.5, 0.125, -3.25, -12.0, // row 2
            ],
        );
        CpuBackend {}.add(&mut y, &x).unwrap();
        assert_close(
            &data(&y),
            &[
                0.0, 0.0, -3.0, 0.75, // row 0
                0.0, 0.0, 4.5, 0.0, // row 1
                100.5, 0.0, 0.0, -6.0, // row 2
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn add_accepts_vectors() {
        let mut y = tensor(&[3], &[1.0, -2.0, 0.5]);
        CpuBackend {}
            .add(&mut y, &tensor(&[3], &[0.25, 2.0, -1.5]))
            .unwrap();
        assert_close(&data(&y), &[1.25, 0.0, -1.0], DEFAULT_TOL);
    }

    #[test]
    fn add_is_commutative() {
        let a = [1.0, -2.0, 0.5, 3.25, -0.125, 0.0];
        let b = [0.25, 2.0, -1.5, -3.25, 8.0, 4.0];
        let mut ab = tensor(&[2, 3], &a);
        let mut ba = tensor(&[2, 3], &b);
        CpuBackend {}.add(&mut ab, &tensor(&[2, 3], &b)).unwrap();
        CpuBackend {}.add(&mut ba, &tensor(&[2, 3], &a)).unwrap();
        assert_close(&data(&ab), &data(&ba), DEFAULT_TOL);
    }

    #[test]
    fn add_zero_is_identity() {
        let a = [1.0, -2.0, 0.5, 3.25, -0.125, 0.0];
        let mut y = tensor(&[2, 3], &a);
        CpuBackend {}
            .add(&mut y, &Tensor::zeros_f32(vec![2, 3]))
            .unwrap();
        assert_close(&data(&y), &a, DEFAULT_TOL);
    }

    #[test]
    fn add_rejects_mismatched_shapes() {
        let e =
            err(CpuBackend {}.add(&mut tensor(&[2, 3], &[1.0; 6]), &tensor(&[3, 2], &[1.0; 6])));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    // hadamard_product: y *= x

    #[test]
    fn hadamard_product_valid() {
        let mut y = tensor(
            &[3, 4],
            &[
                0.0, 1.0, -1.0, 0.5, // row 0
                2.5, -0.75, 4.0, -8.0, // row 1
                1.5, -0.125, 3.25, 6.0, // row 2
            ],
        );
        let x = tensor(
            &[3, 4],
            &[
                3.0, -1.0, -2.0, 0.25, // row 0
                -2.0, 4.0, 0.5, 0.125, // row 1
                0.5, 8.0, -4.0, -0.5, // row 2
            ],
        );
        CpuBackend {}.hadamard_product(&mut y, &x).unwrap();
        assert_close(
            &data(&y),
            &[
                0.0, -1.0, 2.0, 0.125, // row 0
                -5.0, -3.0, 2.0, -1.0, // row 1
                0.75, -1.0, -13.0, -3.0, // row 2
            ],
            DEFAULT_TOL,
        );
    }

    #[test]
    fn hadamard_product_accepts_vectors() {
        let mut y = tensor(&[3], &[2.0, -3.0, 0.5]);
        CpuBackend {}
            .hadamard_product(&mut y, &tensor(&[3], &[0.25, 0.5, -4.0]))
            .unwrap();
        assert_close(&data(&y), &[0.5, -1.5, -2.0], DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_is_commutative() {
        let a = [2.0, -3.0, 0.5, 1.25, -0.5, 0.0];
        let b = [0.25, 0.5, -4.0, 8.0, -2.0, 3.0];
        let mut ab = tensor(&[2, 3], &a);
        let mut ba = tensor(&[2, 3], &b);
        CpuBackend {}
            .hadamard_product(&mut ab, &tensor(&[2, 3], &b))
            .unwrap();
        CpuBackend {}
            .hadamard_product(&mut ba, &tensor(&[2, 3], &a))
            .unwrap();
        assert_close(&data(&ab), &data(&ba), DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_ones_and_zeros() {
        let a = [2.0, -3.0, 0.5, 1.25, -0.5, 0.0];
        let mut y = tensor(&[2, 3], &a);
        CpuBackend {}
            .hadamard_product(&mut y, &tensor(&[2, 3], &[1.0; 6]))
            .unwrap();
        assert_close(&data(&y), &a, DEFAULT_TOL);
        CpuBackend {}
            .hadamard_product(&mut y, &Tensor::zeros_f32(vec![2, 3]))
            .unwrap();
        assert_close(&data(&y), &[0.0; 6], DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_is_elementwise() {
        let a = [2.0, -3.0, 0.5, 1.25, -0.5, 4.0];
        let b = [0.25, 0.5, -4.0, 8.0, -2.0, 3.0];
        let mut forward = tensor(&[2, 3], &a);
        CpuBackend {}
            .hadamard_product(&mut forward, &tensor(&[2, 3], &b))
            .unwrap();

        let (mut ra, mut rb) = (a, b);
        ra.reverse();
        rb.reverse();
        let mut reversed = tensor(&[2, 3], &ra);
        CpuBackend {}
            .hadamard_product(&mut reversed, &tensor(&[2, 3], &rb))
            .unwrap();

        let mut expected = data(&forward);
        expected.reverse();
        assert_close(&data(&reversed), &expected, DEFAULT_TOL);
    }

    #[test]
    fn hadamard_product_rejects_mismatched_shapes() {
        let e = err(CpuBackend {}
            .hadamard_product(&mut tensor(&[2, 3], &[1.0; 6]), &tensor(&[3, 2], &[1.0; 6])));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    #[test]
    fn elementwise_ops_accept_rank_3() {
        let a = [1.0, -2.0, 0.5, 4.0];
        let b = tensor(&[2, 1, 2], &[0.5, 2.0, -0.5, 1.0]);

        let mut y = tensor(&[2, 1, 2], &a);
        CpuBackend {}.add(&mut y, &b).unwrap();
        assert_close(&data(&y), &[1.5, 0.0, 0.0, 5.0], DEFAULT_TOL);

        let mut y = tensor(&[2, 1, 2], &a);
        CpuBackend {}.hadamard_product(&mut y, &b).unwrap();
        assert_close(&data(&y), &[0.5, -4.0, -0.25, 4.0], DEFAULT_TOL);

        let e = err(CpuBackend {}.add(&mut tensor(&[2, 1, 2], &a), &tensor(&[4], &[0.0; 4])));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    // embedding_lookup

    // embed[i] = [i*10, i*10+1, i*10+2], so each row is recognizable.
    fn embed_5x3() -> Tensor {
        tensor(
            &[5, 3],
            &[
                0.0, 1.0, 2.0, // id 0
                10.0, 11.0, 12.0, // id 1
                20.0, 21.0, 22.0, // id 2
                30.0, 31.0, 32.0, // id 3
                40.0, 41.0, 42.0, // id 4
            ],
        )
    }

    #[test]
    fn embedding_lookup_gathers_rows() {
        let mut out = Tensor::zeros_f32(vec![4, 3]);
        CpuBackend {}
            .embedding_lookup(&[3, 0, 3, 4], &embed_5x3(), &mut out)
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

    #[test]
    fn embedding_lookup_accepts_ids_larger_than_hidden_size() {
        let values: Vec<f32> = (0..400).map(|i| i as f32).collect();
        let embed = tensor(&[100, 4], &values);
        let mut out = Tensor::zeros_f32(vec![3, 4]);
        CpuBackend {}
            .embedding_lookup(&[99, 64, 0], &embed, &mut out)
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

    #[test]
    fn embedding_lookup_single_id() {
        let mut out = Tensor::zeros_f32(vec![1, 3]);
        CpuBackend {}
            .embedding_lookup(&[2], &embed_5x3(), &mut out)
            .unwrap();
        assert_close(&data(&out), &[20.0, 21.0, 22.0], DEFAULT_TOL);
    }

    #[test]
    fn embedding_lookup_empty_ids() {
        let mut out = Tensor::zeros_f32(vec![0, 3]);
        CpuBackend {}
            .embedding_lookup(&[], &embed_5x3(), &mut out)
            .unwrap();
        assert!(out.as_f32().unwrap().is_empty());
    }

    #[test]
    fn embedding_lookup_overwrites_every_element_of_out() {
        let mut out = tensor(&[2, 3], &[999.0; 6]);
        CpuBackend {}
            .embedding_lookup(&[1, 0], &embed_5x3(), &mut out)
            .unwrap();
        assert_close(&data(&out), &[10.0, 11.0, 12.0, 0.0, 1.0, 2.0], DEFAULT_TOL);
    }

    #[test]
    fn embedding_lookup_rejects_id_past_vocab() {
        let mut out = Tensor::zeros_f32(vec![1, 3]);
        assert!(
            CpuBackend {}
                .embedding_lookup(&[5], &embed_5x3(), &mut out)
                .is_err()
        );
    }

    #[test]
    fn embedding_lookup_rejects_out_shape_mismatch() {
        let mut out = Tensor::zeros_f32(vec![3, 2]);
        let e = err(CpuBackend {}.embedding_lookup(&[1, 0], &embed_5x3(), &mut out));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }

    #[test]
    fn embedding_lookup_rejects_wrong_rank() {
        let e = err(CpuBackend {}.embedding_lookup(
            &[0],
            &tensor(&[15], &[0.0; 15]),
            &mut Tensor::zeros_f32(vec![1, 3]),
        ));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
        let e = err(CpuBackend {}.embedding_lookup(
            &[0],
            &embed_5x3(),
            &mut Tensor::zeros_f32(vec![3]),
        ));
        assert!(
            matches!(
                e.downcast_ref(),
                Some(OpsError::RankMismatch { actual: 1, .. })
            ),
            "{e}"
        );
    }

    // Mapped storage: weights arrive mapped, so reads must work and writes
    // must fail cleanly.

    #[test]
    fn ops_read_mapped_inputs() {
        let mut out = Tensor::zeros_f32(vec![4, 3]);
        let embed = mapped(&[5, 3], &data(&embed_5x3()));
        CpuBackend {}
            .embedding_lookup(&[3, 0, 3, 4], &embed, &mut out)
            .unwrap();
        assert_eq!(row(&out, 0), &[30.0, 31.0, 32.0]);

        let mut out = Tensor::zeros_f32(vec![2, 4]);
        let a = mapped(&[2, 3], &data(&a_2x3()));
        let b = mapped(&[4, 3], &data(&b_4x3()));
        CpuBackend {}.matmul(&a, &b, &mut out).unwrap();
        assert_close(
            &data(&out),
            &[6.0, -6.5, 6.0, 2.0, 1.0, 8.0, -10.5, 6.5],
            DEFAULT_TOL,
        );

        let mut out = Tensor::zeros_f32(vec![2, 3]);
        CpuBackend {}
            .rmsnorm(&a_2x3(), &mapped(&[3], &[1.0, 2.0, 0.5]), 1e-6, &mut out)
            .unwrap();
        assert_close(
            &data(&out),
            &[1.0392302, -2.7712806, 0.0, -0.2553769, 3.064523, -0.3830654],
            DEFAULT_TOL,
        );

        let mut y = tensor(&[3], &[1.0, -2.0, 0.5]);
        CpuBackend {}
            .add(&mut y, &mapped(&[3], &[0.25, 2.0, -1.5]))
            .unwrap();
        assert_close(&data(&y), &[1.25, 0.0, -1.0], DEFAULT_TOL);
    }

    #[test]
    fn ops_refuse_mapped_outputs() {
        let e = err(CpuBackend {}.add(&mut mapped(&[3], &[0.0; 3]), &tensor(&[3], &[1.0; 3])));
        assert!(e.to_string().contains("read-only"), "{e}");
        let e = err(CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut mapped(&[2, 4], &[0.0; 8])));
        assert!(e.to_string().contains("read-only"), "{e}");
        let e = err(CpuBackend {}.embedding_lookup(
            &[1, 0],
            &embed_5x3(),
            &mut mapped(&[2, 3], &[0.0; 6]),
        ));
        assert!(e.to_string().contains("read-only"), "{e}");
    }

    #[test]
    fn shape_mismatch_carries_both_shapes() {
        let b = Tensor::zeros_f32(vec![4, 7]);
        let e = err(CpuBackend {}.matmul(&a_2x3(), &b, &mut Tensor::zeros_f32(vec![2, 4])));
        match e.downcast::<OpsError>() {
            Ok(OpsError::ShapeMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, [4, 3]);
                assert_eq!(actual, [4, 7]);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn rank_mismatch_names_the_tensor() {
        let b = Tensor::zeros_f32(vec![4, 3, 1]);
        let e = err(CpuBackend {}.matmul(&a_2x3(), &b, &mut Tensor::zeros_f32(vec![2, 4])));
        match e.downcast::<OpsError>() {
            Ok(OpsError::RankMismatch {
                name,
                expected,
                actual,
            }) => {
                assert_eq!(name, "matmul b");
                assert_eq!((expected, actual), (2, 3));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn shape_mismatch_is_reported_before_read_only() {
        let e = err(CpuBackend {}.matmul(&a_2x3(), &b_4x3(), &mut mapped(&[3, 2], &[0.0; 6])));
        assert!(
            matches!(e.downcast_ref(), Some(OpsError::ShapeMismatch { .. })),
            "{e}"
        );
    }
}
