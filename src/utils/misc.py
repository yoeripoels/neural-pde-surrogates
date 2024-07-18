import os
from typing import Union, Dict, List
import torch
import numpy as np
from torch.utils.data import Dataset
from data.base import DatasetInterface
from common.interfaces import D
import random
import string
import time


class Logger(object):
    def __init__(self, default_stdout, write_log=True, filename='log.txt'):
        self.terminal = default_stdout
        self.write_log = write_log
        if self.write_log:
            self.log = open(filename, 'a')

    def write(self, message):
        if self.write_log:
            self.log.write(message)
            self.log.flush()
        self.terminal.write(message)
        self.terminal.flush()

    def flush(self):
        if self.write_log:
            self.log.flush()
        self.terminal.flush()

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


def check_directory() -> None:
    """
    Check if log directory exists within experiments
    """
    if not os.path.exists(f"experiments"):
        os.mkdir(f"experiments")
    if not os.path.exists(f"experiments/log"):
        os.mkdir(f"experiments/log")
    if not os.path.exists(f"models/output"):
        os.mkdir(f"models/output")


def to_float(x: Union[float, torch.Tensor]) -> float:
    if isinstance(x, float):
        return x
    else:
        return x.item()


def to_floatdict(x: Dict[str, Union[float, torch.Tensor]]) -> Dict[str, float]:
    return {k: to_float(v) for k, v in x.items()}


def to_floatlist(x: List[Union[float, torch.Tensor]]) -> List[float]:
    return [to_float(v) for v in x]


def dict_str(x: dict, prefix: str = '', mapping: str = ': ', postfix: str = '', subdir_prefix: str = "  ") -> str:
    return '\n'.join([f'{prefix}{k}{mapping}{v}{postfix}' if not isinstance(v, dict) else
                      f'{prefix}{k}{mapping}\n{dict_str(v, prefix=subdir_prefix + prefix, mapping=mapping, postfix=postfix)}{postfix}'
                      for (k, v) in x.items()])


def get_graph_from_batch(u, pos, batch, idx):
    u, pos, batch = [x.cpu().numpy() for x in [u, pos, batch]]
    return u[batch == idx], pos[batch == idx]


def grid_graph_to_array(u, pos, batch, dxs):
    """
    Convert a graph to a grid, for plotting it as image.
    Note that we assume the graph (pos) to represent a grid, where dxs specifies the dx between nodes in
    the specified grid dimension.
    Args:
        u: graph node values
        pos: graph node positions
        batch: graph batch information
        dxs: delta-x of graph node positions
    Returns: np ndarray of graph converted to grids
    """
    batch_size = torch.max(batch) + 1
    simulations = []
    for i in range(batch_size):
        u_i, pos_i = get_graph_from_batch(u, pos, batch, i)
        pos_i_grid = pos_i.copy()
        min_x = pos_i[:, 1:].min(axis=0)
        for i in range(pos_i.shape[1] - 1):
            pos_i_grid[:, i + 1] -= min_x
            pos_i_grid[:, i + 1] /= dxs[i]
        pos_i_int = np.rint(pos_i_grid).astype(np.int64)
        assert np.sum(pos_i_grid[:, 1:]) == np.sum(pos_i_int[:, 1:])
        pos_i = pos_i_int
        num_c, num_t = u_i.shape[1:]

        dims = pos_i[:, 1:].max(axis=0)

        dims = [int(x+1) for x in dims]
        img = np.zeros((num_t, *dims, num_c))
        for t in range(num_t):
            indexing = [t]
            for j in range(len(dims)):
                dim_idx = j+1
                indexing.append(pos_i[:, dim_idx])
            indexing = tuple(indexing)
            img[indexing] = u_i[:, :, t]
        simulations.append(img)
    simulations = np.array(simulations)
    return np.moveaxis(simulations, -1, 1)


class DatasetToInterface(DatasetInterface):
    def __init__(self, dataset: Dataset, interface: D, set_as: str = 'test'):
        if set_as not in ['train', 'valid', 'test', 'all']:
            raise ValueError('"set_as" should be either "train", "valid", "test" or "all"')
        if interface not in D:
            raise ValueError('"interface" should be defined in common.interfaces.D')
        self.dataset = dataset
        self.set_as = set_as
        self.interface = interface

    def __repr__(self):
        return f'dataset_to_interface-{self.set_as}-D{self.interface}-{self.dataset}'

    @property
    def data_interface(self):
        return self.interface

    @property
    def train(self):
        if self.set_as in ['train', 'all']:
            return self.dataset
        else:
            return None

    @property
    def valid(self):
        if self.set_as in ['valid', 'all']:
            return self.dataset
        else:
            return None

    @property
    def test(self):
        if self.set_as in ['test', 'all']:
            return self.dataset
        else:
            return None


def get_batch_size(batch):
    elements_in_batch = -1
    for x in batch:
        if isinstance(x, torch.Tensor):
            if elements_in_batch != -1:  # make sure we match
                assert (elements_in_batch == x.shape[0])
            else:
                elements_in_batch = x.shape[0]
        elif isinstance(x, list):
            if elements_in_batch != -1:  # make sure we match
                assert (elements_in_batch == len(x))
            else:
                elements_in_batch = len(x)
    if elements_in_batch == -1:
        raise ValueError("Could not determine elements_in_batch from batch of data!")
    return elements_in_batch


def random_timestr(N=10):
    curr_time = str(round(time.time() * 1000))
    return curr_time + ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=N))


def default(value, d):
    return d if value is None else value
