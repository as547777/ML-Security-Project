from interfaces import AbstractMetric

class MetricsHandler:
    # def __init__(self,metrics:AbstractMetric):
    #     self.metrics=metrics
    
    def handle(self, context):
        model = context["model"]

        # metrics = self.metrics.compute_results(model, (x_test, y_test))
        # context["metrics"] = metrics

        x_test = context["x_test"]
        y_test = context["y_test"]
        _, acc = model.predict((x_test, y_test))
        context["acc"] = acc

        x_test_asr = context["x_test_asr"]
        y_test_asr = context["y_test_asr"]
        _, acc_asr = model.predict((x_test_asr, y_test_asr))
        context["acc_asr"] = acc_asr
        