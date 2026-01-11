from interfaces.AbstractAttack import AbstractAttack

class NoAttack(AbstractAttack):
    __desc__ = {
        "display_name": "None",
        "description": "No data poisoning will occur, plain data will be sent forward.",
        "type": None,
        "time": None,
        "params": {}
    }

    def __init__(self):
        self.source_label = 0
        self.target_label = 0
        pass

    def apply_trigger(self, tensor):
        return tensor

    def prepare_for_attack_success_rate(self, data_test):
        x_test, y_test = data_test
        return x_test, y_test

    def execute(self, model, data, params):
        x_train, y_train, x_test, y_test = data
        return x_train, y_train, x_test, y_test