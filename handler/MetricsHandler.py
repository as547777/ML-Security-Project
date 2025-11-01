from interfaces import AbstractMetric

class MetricsHandler:
    def __init__(self,metrics):
        self.metrics=metrics
    
    def handle(self, context):
        model=context["model"]
        X=context["X"]
        y=context["y"]

        metrics = self.metrics.compute_results(model, X,y)
        context["metrics"] = metrics
        