from interfaces import AbstractDataset

class DatasetHandler:
    def __init__(self, dataset:AbstractDataset):
        self.dataset=dataset
    
    def handle(self, context):
        (x_train, y_train), (x_test, y_test), (w_res, h_res), color_channels = self.dataset.load()
        context["x_train"] = x_train
        context["y_train"] = y_train
        context["x_test"] = x_test
        context["y_test"] = y_test
        context["w_res"] = w_res
        context["h_res"] = h_res
        context["color_channels"] = color_channels