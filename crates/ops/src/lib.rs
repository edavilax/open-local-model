use tensor::{Shape, Tensor};

pub mod cpubackend;

pub struct RopeTable {
    cos: Tensor,
    sin: Tensor,
}

impl RopeTable {
    /// Builds a new table to pre-calculate the cosine/sine values
    /// used in the RoPE calculations.
    pub fn new(head_dim: usize, max_seq: usize, theta: f32) -> Self {
        assert!(head_dim % 2 == 0 && head_dim > 0);
        assert!(theta > 0.0);
        let m = max_seq;
        let n = head_dim / 2;
        let out_shape = Shape::new(&[m, n]);
        let num_elems = out_shape.num_elems();
        let mut cos = Tensor {
            shape: out_shape.clone(),
            data: vec![0.0; num_elems],
        };
        let mut sin = Tensor {
            shape: out_shape,
            data: vec![0.0; num_elems],
        };
        // Calculate frequencies.
        let mut freqs: Vec<f32> = Vec::new();
        for i in 0..n {
            freqs.push(1.0 / (theta.powf(2.0 * (i as f32) / (head_dim as f32))));
        }
        // Calculate cosines/sines for each step.
        for i in 0..m {
            let step = i as f32;
            for (j, freq) in freqs.iter().enumerate() {
                let idx = get_index(&cos, i, j);
                cos.data[idx] = (step * freq).cos();
                sin.data[idx] = (step * freq).sin();
            }
        }
        RopeTable { cos, sin }
    }
}

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

    /// Calculates the RoPE (Rotary Position Embeddings) of the input tensor,
    /// and provides the result to `out`.
    ///
    /// This method implements the rotate-half variant of RoPE for optimal
    /// cache locality. Hugging Face and Llama models are trained with this, so
    /// must use rotate-half.
    ///
    /// `table` is the set of cosine/sine tables used for the RoPE calculation.
    ///
    /// `m_start` is the starting index for the token offset. An input tensor of
    /// N row will provide output for token positions [m_start, m_start + N).
    fn rope(&self, t: &Tensor, table: &RopeTable, m_start: usize, out: &mut Tensor);
}

// Helper functions

fn get_index(t: &Tensor, i: usize, j: usize) -> usize {
    let m = t.shape.get_dim(1).unwrap_or_default();
    m * i + j
}

#[cfg(test)]
mod test {
    use std::{assert_eq, vec};

    use testutil::{DEFAULT_TOL, assert_close};

    use super::*;

    #[test]
    fn rope_table() {
        let table = RopeTable::new(8, 7, 10000.0);
        let expected_shape = Shape::new(&[7, 4]);

        let expected_cos = Tensor {
            shape: expected_shape.clone(),
            data: vec![
                1.0,
                1.0,
                1.0,
                1.0, // m = 0
                0.5403023,
                0.9950042,
                0.99995,
                0.9999995, // m = 1
                -0.41614684,
                0.9800666,
                0.9998,
                0.999998, // m = 2
                -0.9899925,
                0.9553365,
                0.99955004,
                0.9999955, // m = 3
                -0.6536436,
                0.921061,
                0.9992001,
                0.999992, // m = 4
                0.2836622,
                0.87758255,
                0.99875027,
                0.9999875, // m = 5
                0.96017027,
                0.8253356,
                0.99820054,
                0.999982, // m = 6
            ],
        };
        assert_eq!(table.cos.shape, expected_cos.shape);
        assert_close(
            table.cos.data.as_slice(),
            expected_cos.data.as_slice(),
            DEFAULT_TOL,
        );

        let expected_sin = Tensor {
            shape: expected_shape.clone(),
            data: vec![
                0.0,
                0.0,
                0.0,
                0.0, // m = 0
                0.84147096,
                0.09983342,
                0.009999833,
                0.0009999999, // m = 1
                0.9092974,
                0.19866933,
                0.019998666,
                0.0019999987, // m = 2
                0.14112,
                0.29552022,
                0.0299955,
                0.0029999956, // m = 3
                -0.7568025,
                0.38941833,
                0.039989334,
                0.0039999895, // m = 4
                -0.9589243,
                0.47942555,
                0.04997917,
                0.0049999794, // m = 5
                -0.2794155,
                0.5646425,
                0.059964005,
                0.005999964, // m = 6
            ],
        };
        assert_eq!(table.sin.shape, expected_sin.shape);
        assert_close(
            table.sin.data.as_slice(),
            expected_sin.data.as_slice(),
            DEFAULT_TOL,
        );
    }

    #[test]
    #[should_panic(expected = "head_dim % 2 == 0")]
    fn rope_table_rejects_odd_head_dim() {
        RopeTable::new(7, 4, 10000.0);
    }

    /// cos^2 + sin^2 = 1 everywhere. Fails if cos and sin were built from
    /// different angles, which spot-checking individual values easily misses.
    #[test]
    fn rope_table_entries_lie_on_unit_circle() {
        let table = RopeTable::new(16, 64, 10000.0);
        assert_eq!(table.cos.shape, table.sin.shape);
        let norms: Vec<f32> = table
            .cos
            .data
            .iter()
            .zip(&table.sin.data)
            .map(|(c, s)| c * c + s * s)
            .collect();
        assert_close(&norms, &vec![1.0; norms.len()], DEFAULT_TOL);
    }

    /// At position 1, column j holds angle theta^(-2j/head_dim), so each
    /// column's angle is the previous one times theta^(-2/head_dim). Checking
    /// that ratio verifies the exponent and that `theta` is really used. It
    /// uses theta = 1e6 (Qwen2.5's real value) so a hardcoded 10000 fails.
    #[test]
    fn rope_table_frequencies_follow_theta() {
        let (head_dim, theta) = (8, 1_000_000.0_f32);
        let table = RopeTable::new(head_dim, 2, theta);
        let angle = |j: usize| {
            let idx = get_index(&table.cos, 1, j);
            table.sin.data[idx].atan2(table.cos.data[idx])
        };
        let ratio = theta.powf(-2.0 / head_dim as f32);
        assert_close(&[angle(0)], &[1.0], DEFAULT_TOL);
        for j in 0..head_dim / 2 - 1 {
            assert_close(&[angle(j + 1) / angle(j)], &[ratio], DEFAULT_TOL);
        }
    }
}
