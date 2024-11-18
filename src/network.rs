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
    pub fn train(&mut self, inputs: Array3<f32>, targets: Array1<f32>) {
        for input in inputs.outer_iter() {
            let mut output = input.to_owned();
            for layer in &mut self.layers {
                output = layer.forward(output);
            }
        }
    }
}
