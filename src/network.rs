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

    fn mean_squared_error(&mut self, prediction: Array1<f32>, target: f32) -> f32 {
        let target = Array1::from_shape_fn(prediction.len(), |i| {
            if i == target as usize {
                1.0
            } else {
                0.0
            }
        });

        prediction
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / prediction.len() as f32
    }

    pub fn train(&mut self, inputs: Array2<f32>, targets: Array1<f32>) {
        for inp in inputs.axis_iter(Axis(0)) {
            let input = inp.to_owned();
            let target = targets.clone();
            let res = self
                .layers
                .iter_mut()
                .fold(input.clone(), |input, layer| layer.forward(input));

            let mut error = self.mean_squared_error(input, target[0]);
            for layer in self.layers.iter_mut().rev() {
                error = layer.backward(error);
            }
        }
    }
}
