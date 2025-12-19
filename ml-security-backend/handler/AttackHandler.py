from interfaces import AbstractAttack

class AttackHandler:
    def __init__(self,attack:AbstractAttack):
        self.attack=attack
    
    def handle(self, context):
        x_train = context["x_train"]
        y_train = context["y_train"]
        x_test = context["x_test"]
        y_test = context["y_test"]
        params = context["attack_params"]

        context["attack_instance"]=self.attack

        (x_poisoned_train, y_poisoned_train, x_test_asr, y_test_asr) = self.attack.execute(None, (x_train, y_train, x_test, y_test), params)

        context["x_train"] = x_poisoned_train
        context["y_train"] = y_poisoned_train

        context["x_test_asr"] = x_test_asr
        context["y_test_asr"] = y_test_asr