from abc import ABC, abstractmethod

class AbstractDefense(ABC):
    @abstractmethod
    def execute(self,context):
        pass