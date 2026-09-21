use memmap2::Mmap;
use serde::Deserialize;
use std::error::Error;
use std::{
    fmt::{self},
    vec, write,
};
use thiserror::Error;

#[derive(Debug)]
pub enum Storage {
    Heap(Vec<f32>),
    Mmap(Mmap),
}

impl Storage {
    fn size_bytes(&self) -> usize {
        match self {
            Storage::Heap(v) => v.len() * size_of::<f32>(),
            Storage::Mmap(m) => m.len(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    F32,
}

impl Dtype {
    fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 => size_of::<f32>(),
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dtype::F32 => write!(f, "f32"),
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    dtype: Dtype,
    storage: Storage,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, dtype: Dtype, storage: Storage) -> Result<Self, Box<dyn Error>> {
        let shape_size: usize = shape.iter().product();
        let shape_size = shape_size * dtype.size_bytes();
        if shape_size != storage.size_bytes() {
            return Err(TensorError::SizeMismatch {
                shape_size,
                data_size: storage.size_bytes(),
            }
            .into());
        }
        match storage {
            Storage::Heap(_) => {
                if dtype != Dtype::F32 {
                    Err(TensorError::IncompatibleStorage(String::from(
                        "Cannot use heap with dtype other than f32",
                    )))
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }?;
        Ok(Tensor {
            shape,
            dtype,
            storage,
        })
    }

    pub fn zeros_f32(shape: Vec<usize>) -> Self {
        let heap_size = shape.iter().product();
        Tensor::new(shape, Dtype::F32, Storage::Heap(vec![0.0; heap_size])).unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn as_f32(&self) -> Result<&[f32], Box<dyn Error>> {
        self.check_dtype(Dtype::F32)?;
        match &self.storage {
            Storage::Heap(v) => Ok(v.as_slice()),
            Storage::Mmap(m) => cast_f32(m),
        }
    }

    pub fn as_mut_f32(&mut self) -> Result<&mut [f32], Box<dyn Error>> {
        self.check_dtype(Dtype::F32)?;
        match &mut self.storage {
            Storage::Heap(v) => Ok(v.as_mut_slice()),
            Storage::Mmap(_) => Err(TensorError::ReadOnly.into()),
        }
    }

    pub fn dim1(&self) -> Result<usize, Box<dyn Error>> {
        if self.shape.len() != 1 {
            return Err(TensorError::BadRank {
                expected: 1,
                actual: self.shape.len(),
            }
            .into());
        }
        Ok(self.shape[0])
    }

    pub fn dim2(&self) -> Result<(usize, usize), Box<dyn Error>> {
        if self.shape.len() != 2 {
            return Err(TensorError::BadRank {
                expected: 2,
                actual: self.shape.len(),
            }
            .into());
        }
        Ok((self.shape[0], self.shape[1]))
    }

    fn check_dtype(&self, dtype: Dtype) -> Result<(), Box<dyn Error>> {
        if self.dtype != dtype {
            return Err(TensorError::DtypeMismatch {
                expected: dtype,
                actual: self.dtype,
            }
            .into());
        }
        Ok(())
    }
}

fn cast_f32(bytes: &[u8]) -> Result<&[f32], Box<dyn Error>> {
    let ptr = bytes.as_ptr().cast::<f32>();
    if !ptr.is_aligned() || bytes.len() % size_of::<f32>() != 0 {
        return Err(TensorError::BadLayout(Dtype::F32).into());
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, bytes.len() / size_of::<f32>()) })
}

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("Tensor shape needs {shape_size} elements but got {data_size} elements from data")]
    SizeMismatch { shape_size: usize, data_size: usize },
    #[error("Out of bounds access to tensor")]
    OutOfBounds,
    #[error("Tried to fetch dtype {expected} but got {actual}")]
    DtypeMismatch { expected: Dtype, actual: Dtype },
    #[error("{0}")]
    IncompatibleStorage(String),
    #[error("Attempting to access write view of read-only memory")]
    ReadOnly,
    #[error("Expected rank of {expected} but got {actual}")]
    BadRank { expected: usize, actual: usize },
    #[error("Cannot cast a set of bytes to a {0} equivalent")]
    BadLayout(Dtype),
}

#[cfg(test)]
mod tests {
    use super::*;
    use testutil::{fixture, mmap_f32, mmap_f32_misaligned, mmap_file};

    fn heap(data: &[f32]) -> Storage {
        Storage::Heap(data.to_vec())
    }

    fn mapped(shape: Vec<usize>, data: &[f32]) -> Tensor {
        Tensor::new(shape, Dtype::F32, Storage::Mmap(mmap_f32(data))).unwrap()
    }

    #[track_caller]
    fn tensor_err<T: std::fmt::Debug>(result: Result<T, Box<dyn Error>>) -> TensorError {
        *result.unwrap_err().downcast().expect("not a TensorError")
    }

    // Construction

    #[test]
    fn rank_zero_tensor_holds_one_element() {
        let t = Tensor::new(vec![], Dtype::F32, heap(&[1.5])).unwrap();
        assert!(t.shape().is_empty());
        assert_eq!(t.as_f32().unwrap(), &[1.5]);
    }

    #[test]
    fn shape_is_reported_as_given() {
        assert_eq!(Tensor::zeros_f32(vec![2, 3]).shape(), &[2, 3]);
        assert_eq!(Tensor::zeros_f32(vec![2, 3, 4]).shape(), &[2, 3, 4]);
    }

    #[test]
    fn zero_sized_dimension_holds_no_data() {
        let t = Tensor::new(vec![0, 1, 2], Dtype::F32, heap(&[])).unwrap();
        assert!(t.as_f32().unwrap().is_empty());
    }

    // Both sizes are in bytes.
    #[test]
    fn new_rejects_data_that_does_not_fit_the_shape() {
        let err = tensor_err(Tensor::new(vec![1, 3], Dtype::F32, heap(&[1.0, 1.5])));
        assert!(
            matches!(
                err,
                TensorError::SizeMismatch {
                    shape_size: 12,
                    data_size: 8
                }
            ),
            "{err:?}"
        );

        let err = tensor_err(Tensor::new(vec![2], Dtype::F32, heap(&[1.0, 1.5, 2.0])));
        assert!(
            matches!(
                err,
                TensorError::SizeMismatch {
                    shape_size: 8,
                    data_size: 12
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn zeros_f32_is_all_zero_and_sized_by_the_shape() {
        let t = Tensor::zeros_f32(vec![2, 3, 4]);
        assert_eq!(t.as_f32().unwrap(), &[0.0; 24]);
    }

    #[test]
    fn dtype_display_uses_spec_names() {
        assert_eq!(Dtype::F32.to_string(), "f32");
    }

    // Views

    #[test]
    fn as_f32_exposes_the_data_in_order() {
        let t = Tensor::new(vec![2, 2], Dtype::F32, heap(&[1.0, 2.0, 3.0, 4.0])).unwrap();
        assert_eq!(t.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn as_mut_f32_writes_are_visible_through_as_f32() {
        let mut t = Tensor::zeros_f32(vec![2, 2]);
        t.as_mut_f32().unwrap()[3] = 7.5;
        assert_eq!(t.as_f32().unwrap(), &[0.0, 0.0, 0.0, 7.5]);
    }

    // Rank

    #[test]
    fn dim1_and_dim2_return_the_dimensions() {
        assert_eq!(Tensor::zeros_f32(vec![5]).dim1().unwrap(), 5);
        assert_eq!(Tensor::zeros_f32(vec![2, 3]).dim2().unwrap(), (2, 3));
        assert_eq!(Tensor::zeros_f32(vec![0, 3]).dim2().unwrap(), (0, 3));
    }

    #[test]
    fn dim1_rejects_other_ranks() {
        for shape in [vec![], vec![2, 3], vec![2, 3, 4]] {
            let rank = shape.len();
            let err = tensor_err(Tensor::zeros_f32(shape).dim1());
            assert!(
                matches!(err, TensorError::BadRank { expected: 1, actual } if actual == rank),
                "{err:?}"
            );
        }
    }

    #[test]
    fn dim2_rejects_other_ranks() {
        for shape in [vec![], vec![6], vec![2, 3, 4]] {
            let rank = shape.len();
            let err = tensor_err(Tensor::zeros_f32(shape).dim2());
            assert!(
                matches!(err, TensorError::BadRank { expected: 2, actual } if actual == rank),
                "{err:?}"
            );
        }
    }

    // Mapped storage
    //
    // `DtypeMismatch` and `IncompatibleStorage` need a second dtype to be
    // reachable. `BadLayout` by length is unreachable: `new` already rejects
    // storage whose byte count is not the shape's.

    #[test]
    fn mapped_tensor_exposes_the_file_contents() {
        let values = [1.5, -2.0, 0.0, 3.25, 1e-10, f32::MAX];
        assert_eq!(mapped(vec![2, 3], &values).as_f32().unwrap(), &values);
    }

    // Zero-copy: the view starts at the mapping's own address.
    #[test]
    fn mapped_view_points_into_the_mapping() {
        let map = mmap_f32(&[1.0, 2.0, 3.0, 4.0]);
        let addr = map.as_ptr() as usize;
        let t = Tensor::new(vec![4], Dtype::F32, Storage::Mmap(map)).unwrap();
        assert_eq!(t.as_f32().unwrap().as_ptr() as usize, addr);
    }

    // NaN payloads, -0.0, a subnormal and +inf survive only if nothing converts the bytes.
    #[test]
    fn mapped_view_preserves_exact_bit_patterns() {
        let bits = [
            0x7FC0_1234_u32,
            0xFFF0_0001,
            0x8000_0000,
            0x0000_0001,
            0x7F80_0000,
        ];
        let values: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
        let t = mapped(vec![5], &values);
        let seen: Vec<u32> = t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(seen, bits);
    }

    #[test]
    fn new_rejects_mapped_storage_of_the_wrong_size() {
        let err = tensor_err(Tensor::new(
            vec![3],
            Dtype::F32,
            Storage::Mmap(mmap_f32(&[1.0, 2.0])),
        ));
        assert!(
            matches!(
                err,
                TensorError::SizeMismatch {
                    shape_size: 12,
                    data_size: 8
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn as_mut_f32_refuses_mapped_storage() {
        let mut t = mapped(vec![2], &[1.0, 2.0]);
        assert!(matches!(tensor_err(t.as_mut_f32()), TensorError::ReadOnly));
        assert_eq!(t.as_f32().unwrap(), &[1.0, 2.0]); // still readable
    }

    // Construction succeeds because the byte count is right; the view is where alignment matters.
    #[test]
    fn as_f32_rejects_a_misaligned_mapping() {
        let map = mmap_f32_misaligned(&[1.0, 2.0, 3.0]);
        let t = Tensor::new(vec![3], Dtype::F32, Storage::Mmap(map)).unwrap();
        let err = tensor_err(t.as_f32());
        assert!(matches!(err, TensorError::BadLayout(Dtype::F32)), "{err:?}");
    }

    // A real toy weight, mapped whole, against an independent decode of the same bytes.
    #[test]
    fn mapped_fixture_weight_matches_an_independent_decode() {
        let path = fixture("toy-f32/model/weights/layers.0.attn.k_proj.bin");
        let expected: Vec<f32> = std::fs::read(&path)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let t = Tensor::new(vec![32, 64], Dtype::F32, Storage::Mmap(mmap_file(&path))).unwrap();
        assert_eq!(t.as_f32().unwrap(), expected);
    }
}
