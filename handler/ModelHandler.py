from interfaces import Model


class ModelHandler:
    def __init__(self, model:Model):
        self.model=model
    
    def handle(self, context):
        X=context["X"]
        y=context["y"]
        lr=context["learning_rate"]
        epochs=context["epochs"]

        self.model.train(X,y,lr,epochs)
        context["model"]=self.model
        