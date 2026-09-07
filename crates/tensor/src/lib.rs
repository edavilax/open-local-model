#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    pub fn ndims(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elems(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn get_dim(&self, i: usize) -> Option<usize> {
        self.dims.get(i).copied()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Shape,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn is_valid(&self) -> bool {
        return self.shape.num_elems() == self.data.len();
    }
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use super::*;

    #[test]
    fn empty_shape() {
        let shape = Shape::new(&[]);
        assert_eq!(0, shape.ndims());
        assert_eq!(1, shape.num_elems());
        assert!(shape.get_dim(0).is_none());
    }

    #[test]
    fn nonempty_shape() {
        let shape = Shape::new(&[2, 3]);
        assert_eq!(2, shape.ndims());
        assert_eq!(6, shape.num_elems());
        assert_eq!(2, shape.get_dim(0).unwrap());
        assert_eq!(3, shape.get_dim(1).unwrap());
        assert!(shape.get_dim(2).is_none());
    }

    #[test]
    fn leading_zeroes_shape() {
        let shape = Shape::new(&[0, 1, 2]);
        assert_eq!(3, shape.ndims());
        assert_eq!(0, shape.num_elems());
        assert_eq!(0, shape.get_dim(0).unwrap());
        assert_eq!(1, shape.get_dim(1).unwrap());
        assert_eq!(2, shape.get_dim(2).unwrap());
        assert!(shape.get_dim(3).is_none());
    }

    #[test]
    fn valid_tensor() {
        let tensor = Tensor {
            shape: Shape::new(&[1, 2]),
            data: [1.0, 1.5].to_vec(),
        };
        assert!(tensor.is_valid());
    }

    #[test]
    fn invalid_tensor() {
        let tensor = Tensor {
            shape: Shape::new(&[1, 3]),
            data: [1.0, 1.5].to_vec(),
        };
        assert!(!tensor.is_valid());
    }
}
