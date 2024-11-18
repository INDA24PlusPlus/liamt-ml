use mnist::*;
use ndarray::prelude::*;

pub struct MNIST {
    pub train_data: Array3<f32>,
    pub train_labels: Array1<f32>,
    pub validation_data: Array3<f32>,
    pub validation_labels: Array1<f32>,
    pub test_data: Array3<f32>,
    pub test_labels: Array1<f32>,
}

impl MNIST {
    fn convert_images(images: Vec<u8>, len: usize) -> Array3<f32> {
        Array3::from_shape_vec((len, 28, 28), images)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f32 / 256.0)
    }

    fn convert_labels(labels: Vec<u8>, len: usize) -> Array1<f32> {
        Array1::from_shape_vec(len, labels)
            .expect("Error converting training labels to Array2 struct")
            .map(|x| *x as f32)
    }

    pub fn init(trn_len: u32, val_len: u32, tst_len: u32) -> Self {
        let Mnist {
            trn_img,
            trn_lbl,
            val_img,
            val_lbl,
            tst_img,
            tst_lbl,
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(trn_len)
            .validation_set_length(val_len)
            .test_set_length(tst_len)
            .finalize();

        MNIST {
            train_data: MNIST::convert_images(trn_img, trn_len as usize),
            train_labels: MNIST::convert_labels(trn_lbl, trn_len as usize),
            validation_data: MNIST::convert_images(val_img, val_len as usize),
            validation_labels: MNIST::convert_labels(val_lbl, val_len as usize),
            test_data: MNIST::convert_images(tst_img, tst_len as usize),
            test_labels: MNIST::convert_labels(tst_lbl, tst_len as usize),
        }
    }
}
