from interfaces import AbstractMetric

class MetricsHandler:
    def __init__(self,metrics:AbstractMetric):
        self.metrics=metrics
    
    def handle(self, context):
        model=context["model"]
        x_test = context["x_test"]
        y_test = context["y_test"]

        metrics = self.metrics.compute_results(model, (x_test, y_test))
        context["metrics"] = metrics
        