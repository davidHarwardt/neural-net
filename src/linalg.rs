use std::ops::{Mul, Add, Sub};


#[derive(Clone, Debug)]
pub struct Matrix<T> {
    dim: Dim,
    data: Vec<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim { rows: usize, cols: usize }

impl<T> Matrix<T> {
    // fn iter(&self) -> impl Iterator<Item = &T> {
    //     todo!()
    // }

    fn row(&self, row: usize) -> impl Iterator<Item = &T> {
        (0..self.dim.rows).map(move |i| &self.data[i + row * self.dim.cols])
    }

    fn col(&self, col: usize) -> impl Iterator<Item = &T> {
        (0..self.dim.cols).map(move |i| &self.data[col + i * col])
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

fn dot<'a, T: Mul + Add<Output = T> + 'a>(
    a: impl Iterator<Item = &'a T>,
    b: impl Iterator<Item = &'a T>
) -> <T as Add>::Output {
    a.zip(b).map(|(a, b)| a * b).reduce(Add::add).unwrap()
}

impl<T: Add<Output = T> + Mul> Mul<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        assert_eq!(
            self.dim.cols,
            rhs.dim.rows,
            "number of columns of left matrix must equal to the number of rows of the right matrix"
        );
        let res_rows = self.dim.rows;
        let res_cols = rhs.dim.cols;
        let data = (0..res_rows).flat_map(|row| {
            (0..res_cols).map(|col| {
                dot(self.row(row), rhs.col(col))
            })
        }).collect::<Vec<_>>();
        
        Matrix {
            data,
            dim: Dim { rows: res_rows, cols: res_cols },
        }
    }
}

impl<T: Add> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn add(self, rhs: Matrix<T>) -> Self::Output {
        assert_eq!(self.dim, rhs.dim, "the matrices must have equal size to be added");

        todo!()
    }
}

impl<T: Sub> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T::Output>;
    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        assert_eq!(self.dim, rhs.dim, "the matrices must have equal size to be added");

        todo!()
    }
}
