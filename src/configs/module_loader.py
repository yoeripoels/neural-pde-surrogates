import importlib
from argparse import Namespace


class ModuleClass(object):
    __all__ = list(set(vars().keys()))

    def __init__(self, module):
        for name in dir(module):
            item = getattr(module, name)
            setattr(self, name, item)

    def __getattribute__(self, name):
        """Method to ensure we deepcopy a dict/namespace if accessed
        """
        item = object.__getattribute__(self, name)
        if isinstance(item, dict):
            return dict(item)
        elif isinstance(item, Namespace):
            out = Namespace()
            for k, v in vars(item).items():
                vars(out)[k] = v
            return out
        else:
            return item


def load_module_safe(module_name, **kwargs):
    abstractModuleSpec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(abstractModuleSpec)
    for key, value in kwargs.items():
        setattr(module, key, value)
    module.__spec__.loader.exec_module(module)
    module = ModuleClass(module)
    return module
