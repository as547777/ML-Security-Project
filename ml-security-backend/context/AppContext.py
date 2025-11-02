from context import registry

class AppContext:
    def __init__(self):
        self.registry=registry

    def resolve_attack(self,name):
        return self.registry.ATTACK_MAP[name]()
    
    def resolve_defense(self, name):
        return self.registry.DEFENSE_MAP[name]()
    
    def resolve_dataset(self,name):
        return self.registry.DATASET_MAP[name]()
    
    def resolve_model(self, name):
        return self.registry.MODEL_MAP[name]()
    
    def resolve_metric(self, name):
        return self.registry.METRIC_MAP[name]()
    
    