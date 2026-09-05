pub mod numpy;

use std::path::{Path, PathBuf};

pub fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(rel)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row<'a>(a: &'a numpy::Array<f32>, r: usize) -> &'a [f32] {
        let cols = a.shape[1];
        &a.data[r * cols..(r + 1) * cols]
    }

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
}
