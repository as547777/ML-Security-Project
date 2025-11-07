from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def train(self, data_train, lr, momentum, epochs):
        pass

    @abstractmethod
    def predict(self, data_test):
        pass