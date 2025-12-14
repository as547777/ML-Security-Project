from interfaces import AbstractDeffense

class DefenseHandler:
    def __init__(self, defense:AbstractDeffense):
        self.defense=defense

    def handle(self, context):
        self.defense.execute(context)
        