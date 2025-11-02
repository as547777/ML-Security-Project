from interfaces import AbstractDeffense

class DefenseHandler:
    def __init__(self, defense:AbstractDeffense):
        self.defense=defense

    def handle(self, context):
        model=context["model"]
        X=context["X"]
        y=context["y"]

        context["defense_result"]=self.defense.execute(model, X, y)
        