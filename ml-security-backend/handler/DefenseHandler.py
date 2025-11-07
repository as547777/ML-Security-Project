from interfaces import AbstractDeffense

class DefenseHandler:
    def __init__(self, defense:AbstractDeffense):
        self.defense=defense

    def handle(self, context):
        model=context["model"]
        x_train = context["x_train"]
        y_train = context["y_train"]

        context["defense_result"]=self.defense.execute(model, (x_train, y_train))
        