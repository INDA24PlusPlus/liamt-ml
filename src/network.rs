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

    fn forward(&mut self, input: Array2<f32>) -> Array2<f32> {
        self.layers
            .iter_mut()
            .fold(input, |input, layer| layer.forward(input))
    }

    fn backward(&mut self, error: Array2<f32>, learning_rate: f32) {
        let mut grad_output = error.clone();
        for layer in self.layers.iter_mut().rev() {
            grad_output = layer.backward(grad_output, learning_rate);
        }
    }

    pub fn test(&mut self, inputs: Array2<f32>, targets: Array2<f32>) {
        let mut correct_preds = 0;
        let batch_iter = self.batch_iterator(inputs.clone(), targets.clone(), 4);
        for (input, target) in batch_iter.iter() {
            let forward = self.forward(input.clone());

            let correct = self.correct_predictions(forward.clone(), target.clone());

            correct_preds += correct;
        }

        let accuracy = correct_preds as f32 / inputs.len_of(Axis(0)) as f32;
        println!(
            "Correct: {}/{} Acc: {:.2}%",
            correct_preds,
            inputs.len_of(Axis(0)),
            accuracy * 100.0
        );
    }

    fn cross_entropy_error(&mut self, prediction: Array2<f32>, target: Array2<f32>) -> f32 {
        let epsilon = 1e-10; // avoid log(0)
        let size = prediction.dim().0;

        let err: f32 = (0..size)
            .map(|i| {
                let log_predicted = prediction.row(i).mapv(|x| (x.max(epsilon)).ln());
                -target.row(i).dot(&log_predicted)
            })
            .sum();

        err / size as f32
    }

    fn correct_prediction(&mut self, prediction: Array1<f32>, target: Array1<f32>) -> bool {
        let pred = prediction
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;
        let targ = target
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .unwrap()
            .0;

        pred == targ
    }

    fn correct_predictions(&mut self, predictions: Array2<f32>, targets: Array2<f32>) -> usize {
        let mut correct = 0;
        for (pred, targ) in predictions.outer_iter().zip(targets.outer_iter()) {
            if self.correct_prediction(pred.to_owned(), targ.to_owned()) {
                correct += 1;
            }
        }
        correct
    }

    fn batch_iterator(
        &mut self,
        inputs: Array2<f32>,
        targets: Array2<f32>,
        batch_size: usize,
    ) -> Vec<(Array2<f32>, Array2<f32>)> {
        let mut start = 0;
        let mut end = batch_size;
        let mut res_vec: Vec<(Array2<f32>, Array2<f32>)> = Vec::new();
        while end <= inputs.len_of(Axis(0)) {
            let input = inputs.slice(s![start..end, ..]).to_owned();
            let target = targets.slice(s![start..end, ..]).to_owned();
            res_vec.push((input, target));
            start = end;
            end += batch_size;
        }
        res_vec
    }

    pub fn train(
        &mut self,
        inputs: Array2<f32>,
        targets: Array2<f32>,
        batch_size: usize,
        epochs: usize,
        learning_rate: f32,
    ) {
        for e in 0..epochs {
            let mut correct_preds = 0;
            let mut error = 0.0;
            let batch_iter = self.batch_iterator(inputs.clone(), targets.clone(), batch_size);

            for (input, target) in batch_iter.iter() {
                let prediction = self.forward(input.clone());
                error = self.cross_entropy_error(prediction.clone(), target.clone());
                correct_preds += self.correct_predictions(prediction.clone(), target.clone());
                self.backward(prediction - target, learning_rate);
            }

            let acc = correct_preds as f32 / inputs.len_of(Axis(0)) as f32;
            println!("Epoch: {} Error: {} Acc: {:.2}%", e, error, acc * 100.0);
        }
    }

    pub fn nice(
        &mut self,
        inputs: Array2<f32>,
        targets: Array2<f32>,
        inputs_val: Array2<f32>,
        targets_val: Array2<f32>,
        batch_size: usize,
        epochs: usize,
        learning_rate: f32,
    ) {
        for e in 0..epochs {
            let mut correct_preds = 0;
            let mut error = 0.0;
            let batch_iter = self.batch_iterator(inputs.clone(), targets.clone(), batch_size);

            for (input, target) in batch_iter.iter() {
                let prediction = self.forward(input.clone());
                error = self.cross_entropy_error(prediction.clone(), target.clone());
                correct_preds += self.correct_predictions(prediction.clone(), target.clone());
                self.backward(prediction - target, learning_rate);
            }

            let acc = correct_preds as f32 / inputs.len_of(Axis(0)) as f32;
            println!(
                "Epoch: {} Error: {} Acc(Train): {:.2}%",
                e,
                error,
                acc * 100.0
            );

            self.test(inputs_val.clone(), targets_val.clone());

            println!("-----------");
        }
    }
}
