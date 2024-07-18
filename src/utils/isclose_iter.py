import math
import numpy as np
import torch


def isclose_iter(x, y):
    if isinstance(x, dict) and isinstance(y, dict):
        if not x.keys() == y.keys():
            return False
        return all([isclose_iter(x[k], y[k]) for k in x.keys()])
    elif isinstance(x, list) and isinstance(y, list):
        if not len(x) == len(y):
            return False
        return all([isclose_iter(x[i], y[i]) for i in range(len(x))])
    if isinstance(x, float) and isinstance(y, float):
        return math.isclose(x, y)
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.isclose(x, y)
    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return torch.isclose(x, y)
    else:
        return x == y
