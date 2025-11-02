import importlib
import pkgutil
import inspect
from interfaces.AbstractAttack import AbstractAttack
from interfaces.AbstractDataset import AbstractDataset
from interfaces.AbstractDeffense import AbstractDefense
from interfaces.AbstractMetric import AbstractMetric
from interfaces.Model import Model
from context import registry

def load_plugins(package):
    for _, module_name,_ in pkgutil.iter_modules(package.__path__):
        full_module_name=f"{package.__name__}.{module_name}"
        module = importlib.import_module(full_module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, AbstractAttack):
                registry.register_attack(module_name,obj)
            elif issubclass(obj,AbstractDataset):
                registry.register_dataset(module_name,obj)
            elif issubclass(obj,AbstractDefense):
                registry.register_defense(module_name,obj)
            elif issubclass(obj, Model):
                registry.register_model(module_name,obj)
            elif issubclass(obj, AbstractMetric):
                registry.register_metric(module_name, obj)
                