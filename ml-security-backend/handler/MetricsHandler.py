from interfaces.AbstractMetric import AbstractMetric

class MetricsHandler:
    def __init__(self,metrics:list[AbstractMetric]=[]):
         self.metrics=metrics
    
    def handle(self, context):
         for metric in self.metrics:
              metric.compute(context)
    
    def register(self,metric):
         self.metrics.append(metric)