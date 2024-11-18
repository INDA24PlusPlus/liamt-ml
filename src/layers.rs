use ndarray::prelude::*;
use rand::Rng;

pub trait Layer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32>;
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
}

pub struct DenseLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,
}
impl Layer for DenseLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        self.weights.dot(&input) + &self.biases
    }
}

pub struct SigmoidLayer {}

impl Layer for SigmoidLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        input.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
    }
}

pub struct ReluLayer {}

impl Layer for ReluLayer {
    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        input.iter().map(|x| x.max(0.0)).collect()
    }
}
