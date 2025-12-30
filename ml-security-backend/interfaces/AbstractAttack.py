from abc import ABC, abstractmethod

class AbstractAttack(ABC):
    @abstractmethod
    def execute(self, model, data, params):
        pass

    @abstractmethod
    def prepare_for_attack_success_rate(self, data_test):
        """
        Prepares specific test data for Attack Success Rate (ASR) evaluation.
        
        Args:
            data_test: Tuple of (x_test, y_test).
        """
        pass

    def apply_trigger(self, tensor):
        """
        Applies the attack-specific trigger to a single tensor.
        Mainly used for visualization purposes.

        Args:
            tensor: The input tensor to be modified.
        """
        raise NotImplementedError("Each attack must implement apply_trigger for visualization.")