use crate::TensorError::OutOfBounds;
use memmap2::Mmap;
use serde::Deserialize;
use std::{
    fmt::{self},
    vec, write,
};

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
pub struct Tensor<const R: usize> {
    shape: [usize; R],
    dtype: Dtype,
    storage: Storage,
}
pub type Vector = Tensor<1>;
pub type Matrix = Tensor<2>;

impl<const R: usize> Tensor<R> {
    pub fn new(shape: [usize; R], dtype: Dtype, storage: Storage) -> Result<Self, TensorError> {
        let shape_size: usize = shape.iter().product();
        let shape_size = shape_size * dtype.size_bytes();
        if shape_size != storage.size_bytes() {
            return Err(TensorError::SizeMismatch(shape_size, storage.size_bytes()));
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

    pub fn zeros_f32(shape: [usize; R]) -> Self {
        Tensor::new(
            shape,
            Dtype::F32,
            Storage::Heap(vec![0.0; shape.iter().product()]),
        )
        .unwrap()
    }

    pub fn shape(&self) -> &[usize; R] {
        &self.shape
    }

    pub fn as_f32(&self) -> Result<&[f32], TensorError> {
        self.check_dtype(Dtype::F32)?;
        match &self.storage {
            Storage::Heap(v) => Ok(v.as_slice()),
            Storage::Mmap(m) => cast_f32(m),
        }
    }

    pub fn as_mut_f32(&mut self) -> Result<&mut [f32], TensorError> {
        self.check_dtype(Dtype::F32)?;
        match &mut self.storage {
            Storage::Heap(v) => Ok(v.as_mut_slice()),
            Storage::Mmap(_) => Err(TensorError::ReadOnly),
        }
    }

    fn get_index(&self, coords: &[usize; R]) -> usize {
        let mut idx = 0usize;
        let mut idx_offset = 1usize;
        for (i, x) in coords.iter().enumerate().rev() {
            idx += x * idx_offset;
            idx_offset *= self.shape[i];
        }
        idx
    }

    fn check_bounds(&self, coords: &[usize; R]) -> Result<(), TensorError> {
        for (i, x) in coords.iter().enumerate() {
            if *x >= self.shape[i] {
                return Err(OutOfBounds());
            }
        }
        Ok(())
    }

    fn check_dtype(&self, dtype: Dtype) -> Result<(), TensorError> {
        if self.dtype != dtype {
            return Err(TensorError::DtypeMismatch(dtype, self.dtype));
        }
        Ok(())
    }
}

impl Matrix {
    pub fn num_rows(&self) -> usize {
        self.shape[0]
    }

    pub fn num_cols(&self) -> usize {
        self.shape[1]
    }

    pub fn row_f32(&self, m: usize) -> Result<&[f32], TensorError> {
        self.check_bounds(&[m, 0])?;
        let i = self.get_index(&[m, 0]);
        let j = self.get_index(&[m, self.num_cols()]);
        let data = self.as_f32()?;
        if j > data.len() {
            return Err(OutOfBounds());
        }
        Ok(&data[i..j])
    }

    pub fn row_f32_mut(&mut self, m: usize) -> Result<&mut [f32], TensorError> {
        self.check_bounds(&[m, 0])?;
        let i = self.get_index(&[m, 0]);
        let j = self.get_index(&[m, self.num_cols()]);
        let data = self.as_mut_f32()?;
        if j > data.len() {
            return Err(OutOfBounds());
        }
        Ok(&mut data[i..j])
    }
}

fn cast_f32(bytes: &[u8]) -> Result<&[f32], TensorError> {
    let ptr = bytes.as_ptr().cast::<f32>();
    if !ptr.is_aligned() || bytes.len() % size_of::<f32>() != 0 {
        return Err(TensorError::BadLayout(Dtype::F32));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, bytes.len() / size_of::<f32>()) })
}

#[derive(Debug)]
pub enum TensorError {
    SizeMismatch(usize, usize),
    OutOfBounds(),
    DtypeMismatch(Dtype, Dtype),
    IncompatibleStorage(String),
    ReadOnly,
    BadLayout(Dtype),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::SizeMismatch(shape_size, data_size) => write!(
                f,
                "Tensor shape needs {shape_size} elements but got {data_size} elements from data"
            ),
            TensorError::OutOfBounds() => write!(f, "Out of bounds access to tensor"),
            TensorError::DtypeMismatch(expected, actual) => {
                write!(f, "Tried to fetch dtype {expected} but got {actual}")
            }
            TensorError::IncompatibleStorage(m) => write!(f, "{m}"),
            TensorError::ReadOnly => {
                write!(f, "Attempting to access write view of read-only memory")
            }
            TensorError::BadLayout(dtype) => {
                write!(f, "Cannot cast a set of bytes to a {dtype} equivalent")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use testutil::{fixture, mmap_f32, mmap_f32_misaligned, mmap_file};

    fn heap(data: &[f32]) -> Storage {
        Storage::Heap(data.to_vec())
    }

    /// A [3, 4] matrix where each value encodes its position as
    /// `row * 10 + col`, so a wrong offset shows up as a recognizably wrong
    /// number rather than just "not equal".
    fn m_3x4() -> Matrix {
        let data: Vec<f32> = (0..3)
            .flat_map(|r| (0..4).map(move |c| (r * 10 + c) as f32))
            .collect();
        Matrix::new([3, 4], Dtype::F32, Storage::Heap(data)).unwrap()
    }

    // Construction.
    //
    // `DtypeMismatch` and `IncompatibleStorage` cannot be reached yet: both
    // need a second dtype to exist. Add their tests alongside that dtype.
    // `BadLayout` by length cannot be reached either: `new` already rejects
    // any storage whose byte count is not the shape's, so only misalignment
    // gets that far.

    /// A rank-0 tensor is a scalar. The empty product is 1, so it holds
    /// exactly one element.
    #[test]
    fn rank_zero_tensor_holds_one_element() {
        let t = Tensor::<0>::new([], Dtype::F32, heap(&[1.5])).unwrap();
        assert_eq!(t.shape().len(), 0);
        assert_eq!(t.as_f32().unwrap(), &[1.5]);
    }

    #[test]
    fn shape_is_reported_as_given() {
        let t = Matrix::zeros_f32([2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.num_rows(), 2);
        assert_eq!(t.num_cols(), 3);
    }

    /// A zero anywhere in the shape means no data at all, and that is valid.
    /// Embedding lookup with an empty prompt produces exactly this.
    #[test]
    fn zero_sized_dimension_holds_no_data() {
        let t = Tensor::<3>::new([0, 1, 2], Dtype::F32, heap(&[])).unwrap();
        assert_eq!(t.shape(), &[0, 1, 2]);
        assert!(t.as_f32().unwrap().is_empty());
    }

    #[test]
    fn new_accepts_data_matching_the_shape() {
        assert!(Matrix::new([1, 2], Dtype::F32, heap(&[1.0, 1.5])).is_ok());
    }

    /// The payload is (expected, actual) in bytes. Pinning both values also
    /// pins their order, which a bare `is_err()` would not.
    #[test]
    fn new_rejects_too_little_data() {
        let err = Matrix::new([1, 3], Dtype::F32, heap(&[1.0, 1.5])).unwrap_err();
        assert!(
            matches!(err, TensorError::SizeMismatch(12, 8)),
            "got {err:?}"
        );
    }

    #[test]
    fn new_rejects_too_much_data() {
        let err = Matrix::new([1, 2], Dtype::F32, heap(&[1.0, 1.5, 2.0])).unwrap_err();
        assert!(
            matches!(err, TensorError::SizeMismatch(8, 12)),
            "got {err:?}"
        );
    }

    #[test]
    fn zeros_f32_is_all_zero_and_sized_by_the_shape() {
        let t = Tensor::<3>::zeros_f32([2, 3, 4]);
        let data = t.as_f32().unwrap();
        assert_eq!(data.len(), 24);
        assert!(data.iter().all(|x| *x == 0.0));
    }

    /// `Display` is what ends up in error messages and CLI output, so it
    /// should use the same names as the OLM spec's dtype enum.
    #[test]
    fn dtype_display_uses_spec_names() {
        assert_eq!(Dtype::F32.to_string(), "f32");
    }

    // Whole-tensor views.

    #[test]
    fn as_f32_exposes_data_in_row_major_order() {
        let t = m_3x4();
        assert_eq!(
            t.as_f32().unwrap(),
            &[
                0.0, 1.0, 2.0, 3.0, // row 0
                10.0, 11.0, 12.0, 13.0, // row 1
                20.0, 21.0, 22.0, 23.0, // row 2
            ]
        );
    }

    #[test]
    fn as_mut_f32_writes_are_visible_through_as_f32() {
        let mut t = Matrix::zeros_f32([2, 2]);
        t.as_mut_f32().unwrap()[3] = 7.5;
        assert_eq!(t.as_f32().unwrap(), &[0.0, 0.0, 0.0, 7.5]);
    }

    // Row views.

    /// Every row, including the last. The last row's exclusive end index
    /// equals the data length, which is where an off-by-one in the bounds
    /// check shows up.
    #[test]
    fn row_f32_returns_each_row() {
        let t = m_3x4();
        assert_eq!(t.row_f32(0).unwrap(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(t.row_f32(1).unwrap(), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(t.row_f32(2).unwrap(), &[20.0, 21.0, 22.0, 23.0]);
    }

    /// The decode shape: one row, so the first row is also the last.
    #[test]
    fn row_f32_on_single_row_matrix() {
        let t = Matrix::new([1, 3], Dtype::F32, heap(&[1.0, 2.0, 3.0])).unwrap();
        assert_eq!(t.row_f32(0).unwrap(), &[1.0, 2.0, 3.0]);
    }

    /// With one column each row is a single element, so row `m` spans
    /// `m..m + 1`. Catches stride mistakes that a square matrix hides.
    #[test]
    fn row_f32_on_single_column_matrix() {
        let t = Matrix::new([3, 1], Dtype::F32, heap(&[5.0, 6.0, 7.0])).unwrap();
        assert_eq!(t.row_f32(0).unwrap(), &[5.0]);
        assert_eq!(t.row_f32(1).unwrap(), &[6.0]);
        assert_eq!(t.row_f32(2).unwrap(), &[7.0]);
    }

    #[test]
    fn row_f32_rejects_rows_past_the_end() {
        let t = m_3x4();
        for m in [3, 4, usize::MAX / 8] {
            let err = t.row_f32(m).unwrap_err();
            assert!(
                matches!(err, TensorError::OutOfBounds()),
                "row {m}: got {err:?}"
            );
        }
    }

    #[test]
    fn row_f32_rejects_any_row_of_a_matrix_with_no_rows() {
        let t = Matrix::zeros_f32([0, 3]);
        assert!(matches!(t.row_f32(0), Err(TensorError::OutOfBounds())));
    }

    /// Writing through a row view must land in that row only. Uses the last
    /// row for the same reason as `row_f32_returns_each_row`.
    #[test]
    fn row_f32_mut_writes_only_the_requested_row() {
        let mut t = m_3x4();
        t.row_f32_mut(2)
            .unwrap()
            .copy_from_slice(&[-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(
            t.as_f32().unwrap(),
            &[
                0.0, 1.0, 2.0, 3.0, // row 0 untouched
                10.0, 11.0, 12.0, 13.0, // row 1 untouched
                -1.0, -2.0, -3.0, -4.0, // row 2 replaced
            ]
        );
    }

    #[test]
    fn row_f32_mut_returns_each_row() {
        let mut t = m_3x4();
        assert_eq!(t.row_f32_mut(0).unwrap(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(t.row_f32_mut(1).unwrap(), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(t.row_f32_mut(2).unwrap(), &[20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn row_f32_mut_rejects_rows_past_the_end() {
        let mut t = m_3x4();
        let err = t.row_f32_mut(3).unwrap_err();
        assert!(matches!(err, TensorError::OutOfBounds()), "got {err:?}");
    }

    // Mapped storage. The helpers map throwaway temp files; see testutil::mmap.

    /// Same position-encoding matrix as `m_3x4`, but backed by a mapping.
    fn mapped_3x4() -> Matrix {
        let data = m_3x4().as_f32().unwrap().to_vec();
        Matrix::new([3, 4], Dtype::F32, Storage::Mmap(mmap_f32(&data))).unwrap()
    }

    #[test]
    fn mapped_tensor_exposes_the_file_contents() {
        let values = [1.5, -2.0, 0.0, 3.25, 1e-10, f32::MAX];
        let t = Vector::new([6], Dtype::F32, Storage::Mmap(mmap_f32(&values))).unwrap();
        assert_eq!(t.shape(), &[6]);
        assert_eq!(t.as_f32().unwrap(), &values);
    }

    /// The point of mapped storage: the view is the mapping, not a copy of
    /// it. The slice must start at the mapping's own address.
    #[test]
    fn mapped_view_points_into_the_mapping() {
        let map = mmap_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mapping_addr = map.as_ptr() as usize;
        let t = Vector::new([4], Dtype::F32, Storage::Mmap(map)).unwrap();
        assert_eq!(t.as_f32().unwrap().as_ptr() as usize, mapping_addr);
    }

    /// A reinterpreting view must not normalize anything. NaN payloads, the
    /// sign of zero and subnormals all survive only if no float arithmetic
    /// or conversion touches the bytes.
    #[test]
    fn mapped_view_preserves_exact_bit_patterns() {
        let bits = [
            0x7FC0_1234_u32, // quiet NaN with a payload
            0xFFF0_0001,     // negative NaN, different payload
            0x8000_0000,     // -0.0
            0x0000_0001,     // smallest subnormal
            0x7F80_0000,     // +inf
        ];
        let values: Vec<f32> = bits.iter().map(|b| f32::from_bits(*b)).collect();
        let t = Vector::new([5], Dtype::F32, Storage::Mmap(mmap_f32(&values))).unwrap();
        let seen: Vec<u32> = t.as_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        assert_eq!(seen, bits);
    }

    #[test]
    fn new_rejects_mapped_storage_of_the_wrong_size() {
        let err = Vector::new([3], Dtype::F32, Storage::Mmap(mmap_f32(&[1.0, 2.0]))).unwrap_err();
        assert!(
            matches!(err, TensorError::SizeMismatch(12, 8)),
            "got {err:?}"
        );
    }

    /// Row views go through the same cast, so they work on mappings too,
    /// last row included.
    #[test]
    fn row_f32_returns_each_row_of_a_mapped_matrix() {
        let t = mapped_3x4();
        assert_eq!(t.row_f32(0).unwrap(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(t.row_f32(1).unwrap(), &[10.0, 11.0, 12.0, 13.0]);
        assert_eq!(t.row_f32(2).unwrap(), &[20.0, 21.0, 22.0, 23.0]);
        assert!(matches!(t.row_f32(3), Err(TensorError::OutOfBounds())));
    }

    /// Weights are mapped without write permission. A mutable view must be
    /// refused up front; handing one out would fault on the first write.
    #[test]
    fn as_mut_f32_refuses_mapped_storage() {
        let mut t = mapped_3x4();
        let err = t.as_mut_f32().unwrap_err();
        assert!(matches!(err, TensorError::ReadOnly), "got {err:?}");
    }

    #[test]
    fn row_f32_mut_refuses_mapped_storage() {
        let mut t = mapped_3x4();
        let err = t.row_f32_mut(0).unwrap_err();
        assert!(matches!(err, TensorError::ReadOnly), "got {err:?}");
    }

    /// A refused mutable view must not poison the tensor for reading.
    #[test]
    fn mapped_tensor_is_still_readable_after_a_refused_write() {
        let mut t = mapped_3x4();
        assert!(t.as_mut_f32().is_err());
        assert_eq!(t.row_f32(1).unwrap(), &[10.0, 11.0, 12.0, 13.0]);
    }

    /// An `&[f32]` to a misaligned address is undefined behavior in Rust even
    /// on CPUs that tolerate unaligned loads, so the cast must refuse rather
    /// than reinterpret. Construction still succeeds, because the byte count
    /// is right; the view is where alignment matters.
    #[test]
    fn as_f32_rejects_a_misaligned_mapping() {
        let map = mmap_f32_misaligned(&[1.0, 2.0, 3.0]);
        let t = Vector::new([3], Dtype::F32, Storage::Mmap(map)).unwrap();
        let err = t.as_f32().unwrap_err();
        assert!(
            matches!(err, TensorError::BadLayout(Dtype::F32)),
            "got {err:?}"
        );
    }

    #[test]
    fn row_f32_rejects_a_misaligned_mapping() {
        let map = mmap_f32_misaligned(&[1.0, 2.0, 3.0, 4.0]);
        let t = Matrix::new([2, 2], Dtype::F32, Storage::Mmap(map)).unwrap();
        let err = t.row_f32(0).unwrap_err();
        assert!(
            matches!(err, TensorError::BadLayout(Dtype::F32)),
            "got {err:?}"
        );
    }

    /// The real thing: a weight file from the toy model, mapped whole, read
    /// row by row, and compared against an independent decode of the same
    /// bytes. `k_proj` is [32, 64] f32 with random values, so a wrong stride
    /// or offset cannot match by accident.
    #[test]
    fn mapped_fixture_weight_matches_an_independent_decode() {
        let path = fixture("toy-f32/model/weights/layers.0.attn.k_proj.bin");
        let expected: Vec<f32> = std::fs::read(&path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(expected.len(), 32 * 64);

        let t = Matrix::new([32, 64], Dtype::F32, Storage::Mmap(mmap_file(&path))).unwrap();
        for m in 0..32 {
            assert_eq!(
                t.row_f32(m).unwrap(),
                &expected[m * 64..(m + 1) * 64],
                "row {m}"
            );
        }
    }

    // Private index math. Row accessors only exercise it at rank 2.

    /// Row-major means the last coordinate moves fastest: the stride of each
    /// dimension is the product of the dimensions after it.
    #[test]
    fn get_index_is_row_major() {
        let t = Tensor::<3>::zeros_f32([2, 3, 4]);
        assert_eq!(t.get_index(&[0, 0, 0]), 0);
        assert_eq!(t.get_index(&[0, 0, 1]), 1);
        assert_eq!(t.get_index(&[0, 1, 0]), 4);
        assert_eq!(t.get_index(&[1, 0, 0]), 12);
        assert_eq!(t.get_index(&[1, 2, 3]), 23);
    }

    /// A coordinate equal to its dimension is already out of range, in any
    /// position, even when the other coordinates are fine.
    #[test]
    fn check_bounds_rejects_a_coordinate_at_or_past_its_dimension() {
        let t = Tensor::<3>::zeros_f32([2, 3, 4]);
        assert!(t.check_bounds(&[1, 2, 3]).is_ok());
        for coords in [[2, 0, 0], [0, 3, 0], [0, 0, 4]] {
            assert!(
                matches!(t.check_bounds(&coords), Err(TensorError::OutOfBounds())),
                "{coords:?} should be out of bounds"
            );
        }
    }
}
