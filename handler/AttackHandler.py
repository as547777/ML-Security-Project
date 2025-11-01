from interfaces import AbstractAttack


class AttackHandler:
    def __init__(self,attack:AbstractAttack):
        self.attack=attack
    
    def handle(self, context):
        model = context["model"]
        context["attack_result"] = self.attack.execute(model, (context["X"], context["y"]))
