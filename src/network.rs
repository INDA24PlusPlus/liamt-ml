use crate::layers::*;
use ndarray::prelude::*;
pub struct Network {
    layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Network {
        Network { layers: Vec::new() }
    }
    pub fn push_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }
    pub fn predict(&mut self, inputs: Array1<f32>) -> Array1<f32> {
        self.layers
            .iter_mut()
            .fold(inputs, |input, layer| layer.forward(input))
    }
    /* pub fn predict(&mut self, inputs: Array2<f32>, targets: Array1<f32>) {
        self.layers
            .iter()
            .fold(inputs, |input, layer| layer.forward(input));
    } */
}
