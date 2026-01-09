from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def init(self, init_params):
        pass

    @abstractmethod
    def train(self, data_train, lr, momentum, epochs):
        pass

    @abstractmethod
    def predict(self, data_test):
        pass