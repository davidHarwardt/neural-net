use std::ops::{Mul, Add, Sub, Index, IndexMut};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Matrix<T> {
    dim: Dim,
    data: Vec<T>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vector<T> {
    elements: Vec<T>,
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output { &self.elements[index] }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { &mut self.elements[index] }
}

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let elements = iter.into_iter().collect();
        Vector { elements }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim { rows: usize, cols: usize }
impl Dim {
    fn transposed(self) -> Self {
        let rows = self.cols;
        let cols = self.rows;
        Self { rows, cols }
    }
}

impl<T> Matrix<T> {
    pub fn row(&self, row: usize) -> impl Iterator<Item = &T> {
        (0..self.dim.cols).map(move |i| &self.data[i + row * self.dim.cols])
    }

    pub fn col(&self, col: usize) -> impl Iterator<Item = &T> {
        (0..self.dim.rows).map(move |i| &self.data[col + i * self.dim.cols])
    }

    pub fn cols(&self) -> usize { self.dim.cols }
    pub fn rows(&self) -> usize { self.dim.rows }

    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, T> { self.data.iter() }

    pub fn inner_iter<'a, RI: IntoIterator<Item = R>, R>(&'a self, func: impl FnOnce(std::slice::Iter<'a, T>) -> RI) -> Matrix<R>
    where T: 'a {
        Matrix::from_iter(self.dim.rows, self.dim.cols, func(self.iter()))
    }

    pub fn from_iter(rows: usize, cols: usize, it: impl IntoIterator<Item = T>) -> Self {
        let data = it.into_iter().collect();
        let dim = Dim { rows, cols };
        Matrix { dim, data }
    }

    pub fn vector(data: Vec<T>) -> Self {
        let dim = Dim { rows: data.len(), cols: 1 };
        Self { data, dim }
    }
}

impl<T: Copy> Matrix<T> {
    pub fn apply_fn(&self, mut func: impl FnMut(T) -> T) -> Self {
        self.inner_iter(|it| it.map(|v| func(*v)))
    }
}

impl<T> Index<usize> for Matrix<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[(index * self.dim.rows)..((index + 1) * (self.dim.rows))]
    }
}

impl<T> IndexMut<usize> for Matrix<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[(index * self.dim.rows)..((index + 1) * (self.dim.rows))]
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.dim.rows + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0 * self.dim.rows + index.1]
    }
}

impl<T: Clone> Matrix<T> {
    pub fn transposed(&self) -> Self {
        let data = (0..self.dim.cols)
            .flat_map(|col| { self.col(col).cloned() })
        .collect();
        let dim = self.dim.transposed();
        Self { data, dim }
    }
}

impl<T, I: IntoIterator<Item = T>> FromIterator<I> for Matrix<T> {
    fn from_iter<It: IntoIterator<Item = I>>(iter: It) -> Self {
        let mut last_len = None;
        let mut data = Vec::new();
        let mut rows = 0;
        for it in iter.into_iter() {
            let mut len = 0;
            for v in it.into_iter() {
                len += 1;
                data.push(v);
            }
            last_len.map(|v| assert_eq!(v, len, "all rows must have equal length"));
            last_len = Some(len);
            rows = last_len.unwrap();
        }
        let cols = data.len() / rows;

        Matrix { data, dim: Dim { cols, rows } }
    }
}

impl<T, const COL: usize, const ROW: usize> From<[[T; COL]; ROW]> for Matrix<T> {
    fn from(value: [[T; COL]; ROW]) -> Self {
        let data = value.into_iter().flat_map(|v| v.into_iter()).collect();
        Self { data, dim: Dim { cols: COL, rows: ROW } }
    }
}

fn dot<'a, T: Mul<Output = T> + Add<Output = T> + 'a + Copy>(
    a: impl Iterator<Item = &'a T>,
    b: impl Iterator<Item = &'a T>
) -> <T as Add>::Output {
    a.zip(b).map(|(a, b)| { *a * *b }).reduce(|acc, v| acc + v).unwrap()
}

impl<T: Add<Output = T> + Mul<Output = T> + Copy> Mul<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(
            self.dim.cols,
            rhs.dim.rows,
            "number of cols of left matrix must equal to the number of rows of the right matrix"
        );

        let res_cols = rhs.dim.cols;
        let res_rows = self.dim.rows;
        let data = (0..res_rows).flat_map(|row| {
            (0..res_cols).zip(std::iter::repeat(row)).map(|(col, row)| {
                dot(self.row(row), rhs.col(col))
            })
        }).collect();
        
        Matrix {
            data,
            dim: Dim { rows: res_rows, cols: res_cols },
        }
    }
}

impl<T: Add<Output = T> + Mul<Output = T> + Copy> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: Matrix<T>) -> Self::Output { &self * &rhs }
}

impl<T: Add<Output = T> + Copy> Add<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T::Output>;
    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.dim, rhs.dim, "the matrices must have equal size to be added");
        self.inner_iter(|it| it.zip(rhs.iter()).map(|(a, b)| *a + *b))
    }
}

impl<T: Add<Output = T> + Copy> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn add(self, rhs: Matrix<T>) -> Self::Output { &self + &rhs }
}

impl<T: Sub<Output = T> + Copy> Sub<&Matrix<T>> for &Matrix<T> {
    type Output = Matrix<T::Output>;
    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.dim, rhs.dim, "the matrices must have equal size to be added");
        self.inner_iter(|it| it.zip(rhs.iter()).map(|(a, b)| *a - *b))
    }
}

impl<T: Sub<Output = T> + Copy> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn sub(self, rhs: Matrix<T>) -> Self::Output { &self - &rhs }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn matrix_mult() {
        let mat_1 = Matrix::from([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        let mat_2 = Matrix::from([
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ]);

        let mat_3 = Matrix::from([
            [1],
            [2],
            [3],
        ]);

        let mat_4 = Matrix::from([
            [1, 2, 3],
        ]);

        assert_eq!(
            &mat_1 * &mat_2,
            Matrix::from([
                [300, 360, 420],
                [660, 810, 960],
                [1020, 1260, 1500],
            ]),
        );

        assert_eq!(
            &mat_1 * &mat_3,
            Matrix::from([
                [14],
                [32],
                [50],
           ]),
        );

        assert_eq!(
            &mat_4 * &mat_3,
            Matrix::from([
                [14],
            ]),
        );
    }

    #[test]
    fn matrix_vector() {
        let vec_1 = Matrix::vector(vec![1, 2, 3]);
        let mat_1 = Matrix::from([
            [1],
            [2],
            [3],
        ]);
        assert_eq!(vec_1, mat_1);
    }

    #[test]
    fn matrix_index() {
        let mat_1 = Matrix::from([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        assert_eq!(mat_1[0], [1, 2, 3]);
        assert_eq!(mat_1[1], [4, 5, 6]);
        assert_eq!(mat_1[2], [7, 8, 9]);
        assert_eq!(mat_1[0][0], 1);
        assert_eq!(mat_1[0][1], 2);
        assert_eq!(mat_1[0][2], 3);

        assert_eq!(mat_1[(0, 0)], 1);
        assert_eq!(mat_1[(1, 1)], 5);
    }

    #[test]
    #[should_panic]
    fn matrix_mult_correct_dim() {
        let mat_1 = Matrix::from([
            [1],
            [2],
            [3],
            [4],
        ]);

        let mat_2 = Matrix::from([
            [1, 2, 3],
        ]);

        let _ = &mat_2 * &mat_1;
    }

    #[test]
    fn matrix_transpose() {
        let mat_1 = Matrix::from([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        assert_eq!(
            mat_1.transposed(),
            Matrix::from([
                [1, 4, 7],
                [2, 5, 8],
                [3, 6, 9],
            ]),
        );
    }

    #[test]
    fn matrix_iter() {
        let mat_1 = Matrix::from([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        assert_eq!(mat_1.col(0).cloned().collect::<Vec<_>>(), vec![1, 4, 7]);
        assert_eq!(mat_1.col(1).cloned().collect::<Vec<_>>(), vec![2, 5, 8]);
        assert_eq!(mat_1.col(2).cloned().collect::<Vec<_>>(), vec![3, 6, 9]);

        assert_eq!(mat_1.row(0).cloned().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(mat_1.row(1).cloned().collect::<Vec<_>>(), vec![4, 5, 6]);
        assert_eq!(mat_1.row(2).cloned().collect::<Vec<_>>(), vec![7, 8, 9]);
    }

    #[test]
    fn matrix_add() {
        let mat_1: Matrix<_> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ].into();

        let mat_2: Matrix<_> = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ].into();

        assert_eq!(
            &mat_1 + &mat_2,
            [
                [11, 22, 33],
                [44, 55, 66],
                [77, 88, 99],
            ].into(),
        );
    }

    #[test]
    fn matrix_sub() {
        let mat_1: Matrix<_> = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ].into();

        let mat_2: Matrix<_> = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
        ].into();

        assert_eq!(
            &mat_2 - &mat_1,
            [
                [9, 18, 27],
                [36, 45, 54],
                [63, 72, 81],
            ].into(),
        );
    }

    #[test]
    #[should_panic]
    fn matrix_add_sub_equal_dim() {
        let mat_1 = Matrix::from([
            [1],
            [2],
            [3],
        ]);
        let mat_2 = Matrix::from([
            [1, 2, 3],
        ]);

        let _ = &mat_1 + &mat_2;
        let _ = &mat_1 - &mat_2;
    }
}

