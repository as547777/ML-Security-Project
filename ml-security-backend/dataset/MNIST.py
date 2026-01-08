from interfaces.AbstractDataset import AbstractDataset
from utils.art_dataset_loader import load_art_dataset

class MNIST(AbstractDataset):
    __desc__ = {
        "display_name": "MNIST",
        "description":
            "The MNIST dataset contains 70,000 images of handwritten digits (0–9). Each image is 28×28 grayscale.",
        "type": "Image",
        "trainCount": 60000,
        "testCount": 10000,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    def load(self):
        (x_train, y_train), (x_test, y_test) = load_art_dataset("mnist")

        w_res = 28
        h_res = 28
        color_channels = 1
        classes = 10

        return (x_train, y_train), (x_test, y_test), (w_res, h_res), color_channels, classes