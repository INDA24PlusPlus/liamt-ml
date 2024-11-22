use ndarray::prelude::*;

use crate::layers::*;

pub trait Optimizer {
    fn step(&mut self, layers: &mut Vec<Box<dyn Layer>>, learning_rate: f32);
    fn clear(&mut self) {}
}

pub struct Optimizers {}
impl Optimizers {
    pub fn default() -> Box<dyn Optimizer> {
        Box::new(Default {})
    }
    pub fn adam(b1: f32, b2: f32, epsilon: f32) -> Box<dyn Optimizer> {
        Box::new(Adam {
            b1,
            b2,
            epsilon,
            t: 0.0,
            ms: Vec::new(),
            vs: Vec::new(),
        })
    }
}

pub struct Default {}

impl Optimizer for Default {
    fn step(&mut self, layers: &mut Vec<Box<dyn Layer>>, learning_rate: f32) {
        for layer in layers.iter_mut() {
            if let Some(layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {
                layer.weights = layer.weights.clone() - &layer.dw * learning_rate;
                layer.biases = layer.biases.clone() - &layer.db * learning_rate;
            }
        }
    }
    fn clear(&mut self) {}
}

pub struct Adam {
    b1: f32,
    b2: f32,
    epsilon: f32,
    t: f32,
    ms: Vec<Vec<Vec<f32>>>,
    vs: Vec<Vec<Vec<f32>>>,
}

impl Optimizer for Adam {
    fn step(&mut self, layers: &mut Vec<Box<dyn Layer>>, learning_rate: f32) {
        for (i, layer) in layers.iter_mut().enumerate() {
            if let Some(layer) = layer.as_any_mut().downcast_mut::<DenseLayer>() {}
        }
    }
    fn clear(&mut self) {
        self.t = 0.0;
        self.ms.clear();
        self.vs.clear();
    }
}
