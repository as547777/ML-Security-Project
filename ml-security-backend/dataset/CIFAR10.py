from interfaces.AbstractDataset import AbstractDataset
from utils.art_dataset_loader import load_art_dataset

class CIFAR10(AbstractDataset):
    def load(self):
        (x_train, y_train), (x_test, y_test) = load_art_dataset("cifar10")

        w_res = 32
        h_res = 32
        color_channels = 3

        return (x_train, y_train), (x_test, y_test), (w_res, h_res), color_channels