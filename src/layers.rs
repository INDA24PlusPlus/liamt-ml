use ndarray::prelude::*;
use rand::distributions::Uniform;
use rand::Rng;
use std::any::Any;

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32>;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub struct Layers {}
impl Layers {
    pub fn dense(in_size: usize, out_size: usize) -> Box<dyn Layer> {
        let mut rng = rand::thread_rng();
        let weights =
            Array2::from_shape_fn((in_size, out_size), |_| rng.sample(Uniform::new(-1.0, 1.0)));
        let biases = Array1::from_shape_fn(out_size, |_| rng.sample(Uniform::new(-1.0, 1.0)));

        Box::new(DenseLayer {
            weights,
            biases,
            last_input: None,
            dw: Array2::zeros((0, 0)),
            db: Array1::zeros(0),
        })
    }
    pub fn relu() -> Box<dyn Layer> {
        Box::new(ReluLayer { last_input: None })
    }
    pub fn softmax() -> Box<dyn Layer> {
        Box::new(SoftMaxLayer {})
    }
    pub fn dropout(rate: f32) -> Box<dyn Layer> {
        Box::new(DropoutLayer {
            rate,
            mask: Array2::zeros((0, 0)),
        })
    }
}

pub struct DenseLayer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    last_input: Option<Array2<f32>>,
    pub dw: Array2<f32>,
    pub db: Array1<f32>,
}
impl Layer for DenseLayer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());

        input.dot(&self.weights) + &self.biases
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        // grad of loss with respect to weights
        let grad_weights = self
            .last_input
            .as_ref()
            .unwrap()
            .t()
            .dot(&grad_output.clone());

        // grad of loss with respect to inputs
        let grad_input = grad_output.dot(&self.weights.t());
        // grad of loss with respect to biases
        let grad_biases = grad_output.sum_axis(Axis(0));

        self.dw = grad_weights;
        self.db = grad_biases;

        grad_input
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct ReluLayer {
    last_input: Option<Array2<f32>>,
}
impl Layer for ReluLayer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.last_input = Some(input.clone());
        input.mapv(|x| x.max(0.0))
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        grad_output
            * self
                .last_input
                .as_ref()
                .unwrap()
                .mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct SoftMaxLayer {}
impl Layer for SoftMaxLayer {
    // softmax to get output between 0 and 1
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let max_vals = input.map_axis(Axis(1), |row| row.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let exp_data = (input - &max_vals.insert_axis(Axis(1))).mapv(f32::exp);
        let sum_exp = exp_data.sum_axis(Axis(1)).insert_axis(Axis(1));
        &exp_data / &sum_exp
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        grad_output.clone()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct DropoutLayer {
    rate: f32,
    mask: Array2<f32>,
}
impl Layer for DropoutLayer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.mask = input.mapv(|_| {
            if rand::thread_rng().gen::<f32>() < self.rate {
                0.0
            } else {
                1.0
            }
        });

        input * &self.mask
    }
    fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        grad_output * &self.mask
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
