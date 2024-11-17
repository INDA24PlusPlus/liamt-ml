pub mod mnist;
use mnist::*;

fn main() {
    let data: MNIST = MNIST::init(10, 10, 10);
}
