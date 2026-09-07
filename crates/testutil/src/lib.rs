pub mod numpy;

use std::path::{Path, PathBuf};

use numpy::{Array, Element};

/// Absolute tolerance for comparing f32 output.
pub const DEFAULT_TOL: f32 = 1e-5;

/// Show this many failing indices.
const MAX_REPORTED: usize = 5;

pub fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(rel)
}

/// Load a `.npy` fixture by relative path.
///
/// Same as calling numpy::load, but panics.
#[track_caller]
pub fn load_fixture<T: Element>(rel: &str) -> Array<T> {
    let path = fixture(rel);
    match numpy::load::<T>(&path) {
        Ok(a) => a,
        Err(e) => panic!("load_fixture({rel:?}) failed at {}: {e:?}", path.display()),
    }
}

/// Borrow row `r` of a 2-D array.
#[track_caller]
pub fn row<T>(a: &Array<T>, r: usize) -> &[T] {
    assert_eq!(
        a.shape.len(),
        2,
        "row() needs a 2-D array, got shape {:?}",
        a.shape
    );
    assert!(
        r < a.shape[0],
        "row {r} out of range for shape {:?}",
        a.shape
    );
    let cols = a.shape[1];
    &a.data[r * cols..(r + 1) * cols]
}

/// Assert every element of `actual` is within `tol` (absolute) of `expected`.
///
/// On failure the panic names how many elements are out of tolerance, the
/// single worst one with its absolute and relative error, and the first few
/// failing indices — enough to tell "one bad element" from "everything is
/// off by a constant" without re-running under a debugger.
///
/// NaN never compares within tolerance, and is always reported as the worst
/// offender so it surfaces ahead of merely-large deltas.
#[track_caller]
pub fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "assert_close: length mismatch"
    );

    let mut n_bad = 0usize;
    let mut first_bad: Vec<usize> = Vec::new();
    let mut worst: Option<(usize, f32, f32, f32)> = None;
    let mut worst_key = f32::NEG_INFINITY;

    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let delta = (a - e).abs();
        // NaN is checked first: every comparison against NaN is false, so
        // `delta > tol` alone would silently pass a NaN through.
        if delta.is_nan() || delta > tol {
            n_bad += 1;
            if first_bad.len() < MAX_REPORTED {
                first_bad.push(i);
            }
            // Rank NaN above any finite delta so it wins "worst".
            let key = if delta.is_nan() { f32::INFINITY } else { delta };
            if key > worst_key {
                worst_key = key;
                worst = Some((i, a, e, delta));
            }
        }
    }

    if let Some((i, a, e, delta)) = worst {
        let rel = if e == 0.0 { f32::NAN } else { delta / e.abs() };
        let more = if n_bad > first_bad.len() {
            format!(", and {} more", n_bad - first_bad.len())
        } else {
            String::new()
        };
        panic!(
            "assert_close: {n_bad} of {} elements exceed tol {tol:e}\n  \
             worst at index {i}: actual {a}, expected {e}, abs {delta:e}, rel {rel:e}\n  \
             failing indices: {first_bad:?}{more}",
            actual.len()
        );
    }
}

/// Assert two arrays have the same shape and near-equal data.
#[track_caller]
pub fn assert_array_close(actual: &Array<f32>, expected: &Array<f32>, tol: f32) {
    assert_eq!(
        actual.shape, expected.shape,
        "assert_array_close: shape mismatch"
    );
    assert_close(&actual.data, &expected.data, tol);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_u32_flat() -> Result<(), numpy::NpyError> {
        let a = numpy::load::<u32>(fixture("toy-f32/oracle/token_ids.npy"))?;
        assert_eq!(a.shape, [29]);
        assert_eq!(a.data.len(), 29);
        assert_eq!(&a.data[..5], &[392, 425, 672, 271, 681]);
        Ok(())
    }

    #[test]
    fn load_f32_flat() -> Result<(), numpy::NpyError> {
        let a = numpy::load::<f32>(fixture("toy-f32/oracle/logits_last.npy"))?;
        assert_eq!(a.shape, [1024]);
        assert_eq!(a.data.len(), 1024);
        assert_eq!(
            &a.data[..5],
            &[0.1159164, -0.5703212, -1.252424, -2.3885365, 0.069305882]
        );
        Ok(())
    }

    #[test]
    fn load_f32_multidim() -> Result<(), numpy::NpyError> {
        let a = numpy::load::<f32>(fixture("toy-f32/oracle/layer_0.npy"))?;
        assert_eq!(a.shape, [29, 64]);
        assert_eq!(a.data.len(), 29 * 64);
        assert_eq!(
            &row(&a, 0)[..5],
            &[1.7447004, -9.6088686, 10.651889, 1.9956603, 0.11691344]
        );
        assert_eq!(
            &row(&a, 1)[..5],
            &[-5.0171552, -4.059957, 6.0001159, 4.5852547, -0.58217168]
        );
        Ok(())
    }

    #[test]
    fn load_fixture_reads_npy() {
        let a = load_fixture::<f32>("toy-f32/oracle/layer_0.npy");
        assert_eq!(a.shape, [29, 64]);
    }

    #[test]
    #[should_panic(expected = "load_fixture(\"nope/missing.npy\") failed")]
    fn load_fixture_names_missing_path() {
        load_fixture::<f32>("nope/missing.npy");
    }

    #[test]
    fn row_borrows_correct_slice() {
        let a = Array {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        assert_eq!(row(&a, 0), &[1.0, 2.0, 3.0]);
        assert_eq!(row(&a, 1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "row 2 out of range for shape [2, 3]")]
    fn row_rejects_out_of_range() {
        let a = Array {
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        row(&a, 2);
    }

    #[test]
    #[should_panic(expected = "row() needs a 2-D array, got shape [6]")]
    fn row_rejects_wrong_rank() {
        let a = Array {
            shape: vec![6],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        row(&a, 0);
    }

    #[test]
    fn assert_close_accepts_exact_and_near() {
        assert_close(&[1.0, -2.0], &[1.0, -2.0], DEFAULT_TOL);
        assert_close(&[1.0, -2.0], &[1.000_001, -2.000_001], DEFAULT_TOL);
    }

    #[test]
    fn assert_close_accepts_empty() {
        assert_close(&[], &[], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "assert_close: length mismatch")]
    fn assert_close_rejects_length_mismatch() {
        assert_close(&[1.0, 2.0], &[1.0], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "worst at index 2")]
    fn assert_close_reports_worst_index() {
        // index 1 is off by 1e-4, index 2 by 1e-2 — the larger must win.
        assert_close(&[1.0, 1.000_1, 1.01], &[1.0, 1.0, 1.0], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "2 of 3 elements exceed tol")]
    fn assert_close_counts_failures() {
        assert_close(&[1.0, 5.0, 9.0], &[1.0, 1.0, 1.0], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "worst at index 0")]
    fn assert_close_ranks_nan_worst() {
        // A huge finite delta at index 1 must still lose to the NaN at 0.
        assert_close(&[f32::NAN, 1e30], &[1.0, 1.0], DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "failing indices: [0, 1, 2, 3, 4], and 3 more")]
    fn assert_close_truncates_index_list() {
        assert_close(&[9.0; 8], &[1.0; 8], DEFAULT_TOL);
    }

    #[test]
    fn assert_array_close_accepts_matching() {
        let a = Array {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let b = Array {
            shape: vec![2, 2],
            data: vec![1.000_001, 2.0, 3.0, 4.0],
        };
        assert_array_close(&a, &b, DEFAULT_TOL);
    }

    #[test]
    #[should_panic(expected = "assert_array_close: shape mismatch")]
    fn assert_array_close_rejects_shape_mismatch() {
        let a = Array {
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let b = Array {
            shape: vec![4],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_array_close(&a, &b, DEFAULT_TOL);
    }
}
