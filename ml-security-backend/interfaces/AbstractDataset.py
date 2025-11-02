from abc import ABC, abstractmethod

class AbstractDataset(ABC):
    @abstractmethod
    def load(self):
        pass
