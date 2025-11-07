from interfaces import AbstractAttack

class AttackHandler:
    def __init__(self,attack:AbstractAttack):
        self.attack=attack
    
    def handle(self, context):
        model = context["model"]
        x_train = context["x_train"]
        y_train = context["y_train"]

        context["attack_result"] = self.attack.execute(model, (x_train, y_train))
