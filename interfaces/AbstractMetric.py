from abc import ABC,abstractmethod

class AbstractMetric(ABC):
    @abstractmethod
    def compute_results(self, results):
        pass