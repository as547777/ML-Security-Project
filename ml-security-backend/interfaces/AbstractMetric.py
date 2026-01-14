from abc import ABC,abstractmethod

class AbstractMetric(ABC):
    @abstractmethod
    def compute(self, context):
        pass