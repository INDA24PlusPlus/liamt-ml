pub mod layers;
pub mod mnist;
pub mod network;

use layers::*;
use mnist::*;
use network::*;

fn main() {
    let data: MNIST = MNIST::init(60000, 10000);

    let mut network = Network::new();
    network.push_layer(Layers::dense(28 * 28, 10));
    network.push_layer(Layers::relu());
    network.push_layer(Layers::dense(10, 10));
    network.push_layer(Layers::softmax());

    //network.train(data.train_data, data.train_labels, 16, 50, 0.002);
    network.train(data.train_data, data.train_labels, 32, 10, 0.002);
    println!("-----------");
    network.test(data.test_data, data.test_labels);
}
