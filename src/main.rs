
use std::ops::{Neg, Add, Div, Sub, Mul};

use nalgebra::DMatrix;

trait NetworkValue: Mul<Self, Output = Self> + Add<Self, Output = Self> + Sized + Copy {}
impl<T> NetworkValue for T where T: Mul<Self, Output = Self> + Add<Self, Output = Self> + Sized + Copy {}

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

trait Zero: Sized { const ZERO: Self; }
impl Zero for f32 { const ZERO: Self = 0.0; }
impl Zero for f64 { const ZERO: Self = 0.0; }

pub struct Sigmoid;
impl<T> ActivationFn<T> for Sigmoid
where T: ConstE + PowF + One + Neg<Output = T> + Add<Output = T> + Div<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
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

struct Layer<T> {
    nodes_in: usize,
    nodes_out: usize,
    weights: DMatrix<T>,
    /* biases: Matrix<T>, */
}

fn mult_mat<T: Add<T, Output = T> + Mul<T, Output = T> + Copy>(input: &[T], mat: &DMatrix<T>) -> Vec<T> {
    (0..mat.rows()).map(|row|
        mat.row(row).zip(std::iter::once(row))
            .map(|(v, row)| input[row] * *v)
            .reduce(Add::add).unwrap()
    ).collect()
}

impl<T: NetworkValue> Layer<T> {
    fn calculate(&self, input: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.nodes_in, input.rows(), "input rows should match number of input nodes for layer");
        //           |_| 
        // |_ _ _| * |_| = |_|
        // |_ _ _|   |_|   |_|
        &self.weights * &input
    }
}


struct Network<T, A, C> {
    layers: Vec<Layer<T>>,
    activation_fn: A,
    cost_fn: C,
}
impl<T: NetworkValue, A: ActivationFn<T>, C: CostFn<T>> Network<T, A, C> {
    fn new(layers: &[usize], activation_fn: A, cost_fn: C) -> Self {
        todo!()
    }

    fn input_size(&self) -> usize { self.layers.first().expect("can not get input size of empty network").nodes_in }
    fn output_size(&self) -> usize { self.layers.last().expect("can not get output size of empty network").nodes_out }

    fn calculate_data_point(&self, p: &DataPoint<T>) {

    }

    fn calculate(&self, input: Matrix<T>) -> Matrix<T> {
        self.layers.iter().fold(input, |acc, v| {
            v.calculate(acc).apply_fn(|v| self.activation_fn.calculate(v))
        })
    }
}

fn main() {

}
