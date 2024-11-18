use ndarray::prelude::*;
use rand::Rng;

pub trait Layer {
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32>;
}

pub struct Layers {}
impl Layers {
    pub fn dense(in_size: usize, out_size: usize) -> Box<dyn Layer> {
        let mut rng = rand::thread_rng();
        let weights = Array2::from_shape_fn((out_size, in_size), |_| rng.gen_range(-1.0..1.0));
        let biases = Array1::from_shape_fn(out_size, |_| rng.gen_range(-1.0..1.0));
        Box::new(DenseLayer { weights, biases })
    }
    pub fn sigmoid() -> Box<dyn Layer> {
        Box::new(SigmoidLayer {})
    }
    pub fn relu() -> Box<dyn Layer> {
        Box::new(ReluLayer {})
    }
    pub fn output() -> Box<dyn Layer> {
        Box::new(OutputLayer {})
    }
}

pub struct DenseLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,
}
impl Layer for DenseLayer {
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        self.weights.dot(&input) + &self.biases
    }
}

pub struct SigmoidLayer {}
impl Layer for SigmoidLayer {
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
}

pub struct ReluLayer {}
impl Layer for ReluLayer {
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        input.mapv(|x| x.max(0.0))
    }
}

pub struct OutputLayer {}
impl Layer for OutputLayer {
    // Softmax to get output between 0 and 1
    fn forward(&mut self, input: Array1<f32>) -> Array1<f32> {
        let max = input.iter().fold(0.0_f32, |acc, &x| acc.max(x));
        let exp_sum: f32 = input.iter().map(|x| (x - max).exp()).sum();
        input.mapv(|x| ((x - max).exp() / exp_sum * 10000.0).round() / 10000.0)
    }
}
