pub mod layers;
pub mod mnist;
pub mod network;

use layers::*;
use mnist::*;
use network::*;

fn main() {
    let data: MNIST = MNIST::init(60000, 10000);

    let mut network = Network::new();
    network.push_layer(Layers::dense(28 * 28, 128));
    network.push_layer(Layers::relu());
    network.push_layer(Layers::dropout(0.2));
    network.push_layer(Layers::dense(128, 64));
    network.push_layer(Layers::relu());
    network.push_layer(Layers::dropout(0.2));
    network.push_layer(Layers::dense(64, 10));
    network.push_layer(Layers::softmax());

    network.train(&data.train_data, &data.train_labels, 32, 500, 0.001);
    println!("-----------");
    network.test(&data.test_data, &data.test_labels);

    /* network.nice(
        data.train_data,
        data.train_labels,
        data.test_data,
        data.test_labels,
        32,
        500,
        0.001,
    ); */
}
