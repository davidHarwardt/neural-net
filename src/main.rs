mod linalg;

use std::ops::{Neg, Add, Div, Sub, Mul};

use linalg::Matrix;

pub trait ActivationFn<T> {
    fn calculate(&self, x: T) -> T;
    fn derivative(&self, x: T) -> T;
}

pub trait ConstE: Sized { const E: Self; }
impl ConstE for f32 { const E: Self = std::f32::consts::E; }
impl ConstE for f64 { const E: Self = std::f64::consts::E; }

pub trait PowF: Sized { fn powf(self, exp: Self) -> Self; }
impl PowF for f32 { fn powf(self, exp: Self) -> Self { self.powf(exp) } }
impl PowF for f64 { fn powf(self, exp: Self) -> Self { self.powf(exp) } }

trait One: Sized { const ONE: Self; }
impl One for f32 { const ONE: Self = 1.0; }
impl One for f64 { const ONE: Self = 1.0; }

pub struct Sigmoid;
impl<T> ActivationFn<T> for Sigmoid
where
    T: ConstE + PowF + One + Neg<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
{
    fn calculate(&self, x: T) -> T { T::ONE / (T::ONE + T::E.powf(-x)) }
    fn derivative(&self, x: T) -> T {
        let a = self.calculate(x);
        a * (T::ONE - a)
    }
}

pub trait CostFn<T> {
    fn calculate(&self, output: T, expected: T) -> T;
    fn derivative(&self, output: T, expected: T) -> T;
}

pub struct SquaredError;
impl<T> CostFn<T> for SquaredError
where
    T: Sub<Output = T> + Mul<Output = T> + Copy + One + Add<Output = T>,
{
    fn calculate(&self, output: T, expected: T) -> T { let v = output - expected; v * v }
    fn derivative(&self, output: T, expected: T) -> T { (T::ONE + T::ONE) * (output - expected) }
}

pub struct DataPoint<T> {
    value: Vec<T>,
    expected: Vec<T>,
}

fn main() {
    let m_1 = Matrix::vector(vec![1.0, 2.0, 3.0]);

    let m_2 = Matrix::from([
        [1.0, 2.0, 3.0],
    ]);

    println!("{:?}", m_2 * m_1);
}
