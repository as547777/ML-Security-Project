from interfaces.AbstractModel import AbstractModel

class ModelHandler:
    def __init__(self, model_cls: type[AbstractModel]):
        self.model_cls = model_cls

    def handle(self, context):

        model=self.model_cls()

        w_res = context['w_res']
        h_res = context['h_res']
        color_channels = context['color_channels']
        classes = context['classes']
        init_params = {
            "w_res": w_res,
            "h_res": h_res,
            "color_channels": color_channels,
            "classes": classes
        }

        model.init(init_params)

        x_train = context["x_train"]
        y_train = context["y_train"]
        lr = context["learning_rate"]
        momentum = context["momentum"]
        epochs = context["epochs"]

        model.train((x_train, y_train), lr, momentum, epochs)

        x_test = context["x_test"]
        y_test = context["y_test"]
        _, acc = model.predict((x_test, y_test))
        context["acc"] = acc

        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        _, acc_asr = model.predict((x_test_asr, y_test_asr))
        context["acc_asr"] = acc_asr
        context["model"]=model