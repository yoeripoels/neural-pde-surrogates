import torch
import os
import random
import numpy as np


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # manual_seed_all if multi GPU
    torch.backends.cudnn.deterministic = False  # true causes dilated convs to be 10x+ slower (cudnn bug?)
    torch.backends.cudnn.benchmark = False
