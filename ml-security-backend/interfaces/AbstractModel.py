from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def init(self, w_res, h_res, color_channels, classes):
        pass

    @abstractmethod
    def train(self, data_train, lr, momentum, epochs):
        pass

    @abstractmethod
    def predict(self, data_test):
        pass