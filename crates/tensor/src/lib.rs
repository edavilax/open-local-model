use std::{fmt, vec, write};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<const R: usize> {
    shape: [usize; R],
    data: Vec<f32>,
}
pub type Vector = Tensor<1>;
pub type Matrix = Tensor<2>;

impl<const R: usize> Tensor<R> {
    pub fn new(shape: [usize; R], data: Vec<f32>) -> Result<Self, TensorError> {
        let shape_size: usize = shape.iter().product();
        if shape_size != data.len() {
            return Err(TensorError::SizeMismatch(shape_size, data.len()));
        }
        Ok(Tensor { shape, data })
    }

    pub fn zeros(shape: [usize; R]) -> Self {
        Tensor::new(shape, vec![0.0; shape.iter().product()]).unwrap()
    }

    pub fn shape(&self) -> &[usize; R] {
        &self.shape
    }

    pub fn get(&self, coords: &[usize; R]) -> Result<f32, TensorError> {
        if !self.bounds_check(coords) {
            return Err(TensorError::OutOfBounds());
        }
        let idx = self.get_index(coords);
        Ok(self.data[idx])
    }

    pub fn set(&mut self, coords: &[usize; R], y: f32) -> Result<(), TensorError> {
        if !self.bounds_check(coords) {
            return Err(TensorError::OutOfBounds());
        }
        let idx = self.get_index(coords);
        self.data[idx] = y;
        return Ok(());
    }

    pub fn data_iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    pub fn mut_data_iter(&mut self) -> std::slice::IterMut<'_, f32> {
        self.data.iter_mut()
    }

    fn bounds_check(&self, coords: &[usize; R]) -> bool {
        for (i, x) in coords.iter().enumerate() {
            if *x >= self.shape[i] {
                return false;
            }
        }
        true
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
}

impl Matrix {
    pub fn num_rows(&self) -> usize {
        self.shape[0]
    }

    pub fn num_cols(&self) -> usize {
        self.shape[1]
    }

    // TODO: Return out-of-bounds with error.
    pub fn row(&self, m: usize) -> &[f32] {
        let i = self.get_index(&[m, 0]);
        let j = self.get_index(&[m, self.num_cols()]);
        &self.data[i..j]
    }

    // TODO: Return out-of-bounds with error.
    pub fn mut_row(&mut self, m: usize) -> &mut [f32] {
        let i = self.get_index(&[m, 0]);
        let j = self.get_index(&[m, self.num_cols()]);
        &mut self.data[i..j]
    }
}

#[derive(Debug)]
pub enum TensorError {
    SizeMismatch(usize, usize),
    OutOfBounds(),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::SizeMismatch(shape_size, data_size) => write!(
                f,
                "Tensor shape needs {shape_size} elements but got {data_size} elements from data"
            ),
            TensorError::OutOfBounds() => write!(f, "Out of bounds access to tensor"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_shape() {
        let t = Tensor::<0>::new([], vec![1.5]).unwrap();
        assert_eq!(0, t.shape().len());
        assert_eq!(1, t.shape().iter().product::<usize>());
    }

    #[test]
    fn nonempty_shape() {
        let t = Matrix::zeros([2, 3]);
        assert_eq!(2, t.shape().len());
        assert_eq!(6, t.shape().iter().product::<usize>());
        assert_eq!(2, t.shape()[0]);
        assert_eq!(3, t.shape()[1]);
    }

    #[test]
    fn leading_zeroes_shape() {
        let t = Tensor::<3>::new([0, 1, 2], vec![]).unwrap();
        assert_eq!(3, t.shape().len());
        assert_eq!(0, t.shape().iter().product::<usize>());
        assert_eq!(&[0, 1, 2], t.shape());
    }

    #[test]
    fn valid_tensor() {
        assert!(Matrix::new([1, 2], vec![1.0, 1.5]).is_ok());
    }

    #[test]
    fn invalid_tensor() {
        assert!(Matrix::new([1, 3], vec![1.0, 1.5]).is_err());
    }
}
