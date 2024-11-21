use mnist::*;
use ndarray::prelude::*;

pub struct MNIST {
    pub train_data: Array2<f32>,
    pub train_labels: Array2<f32>,
    pub test_data: Array2<f32>,
    pub test_labels: Array2<f32>,
}

impl MNIST {
    fn convert_images(images: Vec<u8>, len: usize) -> Array2<f32> {
        Array2::from_shape_vec((len, 784), images)
            .unwrap()
            .map(|x| *x as f32 / 256.0)
    }

    fn convert_labels(labels: Vec<u8>, len: usize) -> Array2<f32> {
        let mut res = Array2::zeros((len, 10));
        for i in 0..len {
            res[[i, labels[i] as usize]] = 1.0;
        }

        res
    }

    pub fn init(trn_len: u32, tst_len: u32) -> Self {
        let Mnist {
            trn_img,
            trn_lbl,
            val_img: _,
            val_lbl: _,
            tst_img,
            tst_lbl,
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(trn_len)
            .validation_set_length(0)
            .test_set_length(tst_len)
            .finalize();

        MNIST {
            train_data: MNIST::convert_images(trn_img, trn_len as usize),
            train_labels: MNIST::convert_labels(trn_lbl, trn_len as usize),
            test_data: MNIST::convert_images(tst_img, tst_len as usize),
            test_labels: MNIST::convert_labels(tst_lbl, tst_len as usize),
        }
    }
}
