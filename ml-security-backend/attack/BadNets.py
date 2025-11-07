from interfaces.AbstractAttack import AbstractAttack

class BadNets(AbstractAttack):
    def execute(self, model, data):
        return "Hello from concrete class"