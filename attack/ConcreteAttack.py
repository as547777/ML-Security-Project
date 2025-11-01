from interfaces.AbstractAttack import AbstractAttack

class ConcreteAttack(AbstractAttack):

    def execute(self, model, data):
        return "Hello from concrete class"