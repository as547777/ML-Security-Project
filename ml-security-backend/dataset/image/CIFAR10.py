from interfaces.AbstractDataset import AbstractDataset
from utils.art_dataset_loader import load_art_dataset

class CIFAR10(AbstractDataset):
    __desc__ = {
        "display_name": "CIFAR-10",
        "description":
            "CIFAR-10 consists of 60,000 32×32 color images in 10 classes, with 6,000 images per class.",
        "type": "Image",
        "trainCount": 50000,
        "testCount": 10000,
        "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    }

    def load(self):
        (x_train, y_train), (x_test, y_test) = load_art_dataset("cifar10")

        w_res = 32
        h_res = 32
        color_channels = 3
        classes = 10

        return (x_train, y_train), (x_test, y_test), (w_res, h_res), color_channels, classes