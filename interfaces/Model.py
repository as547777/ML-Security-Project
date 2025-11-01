from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def run(self, X,y,lr, epochs):
        pass