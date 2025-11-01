from interfaces import AbstractDataset

class DatasetHandler:
    def __init__(self, dataset:AbstractDataset):
        self.dataset=dataset
    
    def handle(self, context):
        X,y = self.dataset.load()
        context["X"]=X
        context["y"]=y
        
        