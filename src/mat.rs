use std::ops::{Add, Mul};


pub struct Dim { rows: usize, cols: usize }
pub fn dim(rows: usize, cols: usize) -> Dim { Dim { rows, cols } }

pub trait MatrixValue:
      Add<Self, Output = Self>
    + Mul<Self, Output = Self> 
    + Copy + Sized {}

impl<T> MatrixValue for T
where T: Add<Self, Output = Self> + Mul<Self, Output = Self> + Copy + Sized {}

pub struct Matrix<T> {
    dim: Dim,
    data: Vec<T>,
}

impl<T: MatrixValue> Matrix<T> {
    fn apply_mut() {

    }
}
