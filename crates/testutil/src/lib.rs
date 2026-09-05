mod numpy;

use std::path::{Path, PathBuf};

pub fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(rel)
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

use super::*;

    #[test]
    fn laod_u32_flat() -> Result<(), numpy::NpyError> {
        let a = numpy::load::<u32>(fixture("toy-f32/oracle/token_ids.npy"))?;
        assert_eq!(a.shape, [29]);
        assert_eq!(a.data.len(), 29);
        Ok(())
    }

    #[test]
    fn load_f32_flat() -> Result<(), numpy::NpyError> {
        let a = numpy::load::<f32>(fixture("toy-f32/oracle/logits_last.npy"))?;
        assert_eq!(a.shape, [1024]);
        assert_eq!(a.data.len(), 1024);
        Ok(())
    }

    #[test]
    fn load_f32_multidim() -> Result<(), numpy::NpyError> {
        let a = numpy::load::<f32>(fixture("toy-f32/oracle/layer_0.npy"))?;
        assert_eq!(a.shape, [29, 64]);
        assert_eq!(a.data.len(), 29*64);
        Ok(())
    }
}
