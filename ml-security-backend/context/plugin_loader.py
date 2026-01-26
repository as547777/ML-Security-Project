import importlib
import pkgutil
import inspect
from interfaces.AbstractAttack import AbstractAttack
from interfaces.AbstractDataset import AbstractDataset
from interfaces.AbstractDefense import AbstractDefense
from interfaces.AbstractMetric import AbstractMetric
from interfaces.AbstractModel import AbstractModel
from context import registry

def load_plugins(package):
    package_prefix=package.__name__ + "."
    for module_info in pkgutil.walk_packages(package.__path__, package_prefix):
        full_module_name=module_info.name
        module = importlib.import_module(full_module_name)

        for name, obj in inspect.getmembers(module, inspect.isclass):

            if inspect.isabstract(obj):
                continue

            if issubclass(obj, AbstractAttack):
                registry.register_attack(name,obj)
            elif issubclass(obj,AbstractDataset):
                registry.register_dataset(name,obj)
            elif issubclass(obj,AbstractDefense):
                registry.register_defense(name,obj)
            elif issubclass(obj, AbstractModel):
                registry.register_model(name,obj)
            elif issubclass(obj, AbstractMetric):
                registry.register_metric(name, obj)   
                