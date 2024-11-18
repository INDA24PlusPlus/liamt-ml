pub mod activation;
pub mod layers;
pub mod mnist;
pub mod network;

use layers::*;
use mnist::*;
use ndarray::prelude::*;
use network::*;

fn main() {
    let data: MNIST = MNIST::init(10, 10, 10);

    let mut network = Network::new();
    network.push_layer(Layers::dense(784, 128));
    network.push_layer(Layers::relu());
    network.push_layer(Layers::dense(128, 10));
    network.push_layer(Layers::relu());

    network.train(data.train_data, data.train_labels);
}
