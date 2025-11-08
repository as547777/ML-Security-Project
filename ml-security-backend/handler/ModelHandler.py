from interfaces.AbstractModel import AbstractModel

class ModelHandler:
    def __init__(self, model:AbstractModel):
        self.model=model

    def handle(self, context):
        x_train = context["x_train"]
        y_train = context["y_train"]
        lr = context["learning_rate"]
        momentum = context["momentum"]
        epochs = context["epochs"]

        self.model.train((x_train, y_train), lr, momentum, epochs)
        context["model"]=self.model

        x_test = context["x_test"]
        y_test = context["y_test"]
        _, acc = self.model.predict((x_test, y_test))
        context["acc"] = acc

        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        _, acc_asr = self.model.predict((x_test_asr, y_test_asr))
        context["acc_asr"] = acc_asr