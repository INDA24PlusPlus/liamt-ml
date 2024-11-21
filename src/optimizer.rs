use ndarray::prelude::*;

pub trait Optimizer {
    fn step(&mut self, gradiant: &Array2<f32>, learning_rate: f32);
}

pub struct Optimizers {}
impl Optimizers {
    pub fn adam() -> Box<dyn Optimizer> {
        Box::new(Adam {})
    }
}

pub struct Adam {}

impl Optimizer for Adam {
    fn step(&mut self, gradiant: &Array2<f32>, learning_rate: f32) {
        //todo
    }
}
