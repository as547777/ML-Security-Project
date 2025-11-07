import numpy as np
import torch

from interfaces.AbstractDataset import AbstractDataset
from art.utils import load_dataset

class MNIST(AbstractDataset):
    def load(self):
        (x_train, y_train), (x_test, y_test), min, max = load_dataset("mnist")

        if len(y_train.shape) > 1:
            y_train = np.argmax(y_train, axis=1)
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)

        # convert to torch tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # (N, H, W, C) -> (N, C, H, W)
        x_train = x_train.permute(0, 3, 1, 2)
        x_test = x_test.permute(0, 3, 1, 2)

        return (x_train, y_train), (x_test, y_test)