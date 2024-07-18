from data.base import DatasetInterface
from data.memmap_dataset import MemMapDataset
import os
from common.interfaces import D
from data import transforms
import torch
import numpy as np
from utils.load_yaml import load_yaml
from pdes import PDE2D


class PDE2DDataset(DatasetInterface):
    data_interface = D.sim2d
    def __init__(
        self,
        base_path: str,
        experiment: str,
        data_format: str,
        data_file: str,
        conditioning: str = None,
        t_conditioning: str = None,
        spatial_conditioning: str = None,
        c_filter: list = None,
        split_file: str = None,
        split_val: float = .05,
        split_test: float = .05,
        name: str = "PDE2D",
        preprocess: bool = False,
        preprocess_path: str = None,
    ):
        data_path = os.path.join(base_path, f"{experiment}")
        self.experiment = experiment

        if c_filter is not None:
            c_filter = np.array(c_filter)
            data_transform = lambda u: u[c_filter]
        else:
            data_transform = None

        self.dataset = MemMapDataset(
            data_path, data_file, data_format=data_format,
            conditioning=conditioning, t_conditioning=t_conditioning,
            spatial_conditioning=spatial_conditioning,
            data_transform=data_transform, grid_transform=None,
            preprocess=preprocess, preprocess_path=preprocess_path,
            conditioning_transform=None, t_conditioning_transform=None)

        # split dataset into train <-> valid <-> test
        if split_file is not None:
            if not split_file.lower().endswith(".yaml"):
                split_file = split_file + ".yaml"
            split = load_yaml(os.path.join(data_path, split_file))
            train_idx = np.array(split["train"])
            valid_idx = np.array(split["valid"])
            test_idx = np.array(split["test"])
        else:
            idx = np.arange(len(self.dataset))
            n_val = int(split_val * len(self.dataset))
            n_test = int(split_test * len(self.dataset))
            train_idx = idx[:-(n_val + n_test)]
            valid_idx = idx[-(n_val + n_test):-n_test]
            test_idx = idx[-n_test:]
            print(f"Warning: No data split provided. Using {(1 - split_val - split_test) * 100:.1f}%:"
                  f"{split_val * 100:.1f}%:{split_test * 100:.1f}% train:valid:test ([0:{train_idx.shape[0]}], "
                  f"[{train_idx.shape[0]}:{train_idx.shape[0] + valid_idx.shape[0]}], "
                  f"[{train_idx.shape[0] + valid_idx.shape[0]}, {train_idx.shape[0] + valid_idx.shape[0] + test_idx.shape[0]}]) ")

        self.train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
        self.valid_dataset = torch.utils.data.Subset(self.dataset, valid_idx)
        self.test_dataset = torch.utils.data.Subset(self.dataset, test_idx)

        # set settings
        nt_in = int(self.dataset.tmax / self.dataset.dt) + 1
        nt_out = nt_in
        tmin, tmax = transforms.get_t_downsample(self.dataset.tmin, self.dataset.tmax,
                                                     nt_in, ratio_nt=1)
        x = self.dataset.x
        nx1, nx2 = x.shape[:2]
        # self.x = (x1, x2, 2) for 2 dims, (x1, x2, x3, 3) for 3 dims etc.
        L1 = x[-1, 0, 0] - x[0, 0, 0]
        L2 = x[0, -1, 1] - x[0, 0, 1]

        _, _, _, cond, t_cond, spatial_cond = self.dataset[0]
        n_cond_static = cond.shape[0] if conditioning is not None else 0
        n_cond_dynamic = t_cond.shape[0] if t_conditioning is not None else 0
        n_cond_spatial = spatial_cond.shape[0] if spatial_conditioning is not None else 0

        self._pde = PDE2D(tmin=tmin, tmax=tmax, nt=nt_out, L1=L1, L2=L2, nx1=nx1, nx2=nx2, x=x, name=name,
                          n_cond_static=n_cond_static, n_cond_dynamic=n_cond_dynamic, n_cond_spatial=n_cond_spatial)

    @property
    def pde(self):
        return self._pde

    def __repr__(self):
        return f"{self.pde}_{self.experiment}"

    @property
    def train(self):
        return self.train_dataset

    @property
    def valid(self):
        return self.valid_dataset

    @property
    def test(self):
        return self.test_dataset


