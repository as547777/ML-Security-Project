from abc import ABC, abstractmethod

class AbstractAttack(ABC):
    @abstractmethod
    def execute(self, model, data):
        pass