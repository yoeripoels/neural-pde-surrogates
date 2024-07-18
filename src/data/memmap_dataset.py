import os
from typing import List, Tuple, Union

import numpy as np
import torch
from numpy.lib.format import open_memmap
from mmap_ninja.ragged import RaggedMmap
from torch.utils.data import Dataset

from utils.load_yaml import load_yaml
from utils import misc as util
from utils.load_memmap import load_data
import weakref
import shutil


def precompute_and_save_memmap(memmap_in: np.memmap,
                               filename: Union[str, os.PathLike],
                               transform: callable,
                               dtype: torch.dtype) -> np.memmap:
    N = memmap_in.shape[0]
    element_shape = transform(torch.tensor(memmap_in[0], dtype=dtype)).shape
    memmap_out = open_memmap(filename, mode="w+", dtype=memmap_in.dtype, shape=(N, *element_shape))
    for i in range(N):
        memmap_out[i] = transform(torch.tensor(memmap_in[i], dtype=dtype)).numpy()
    return open_memmap(filename, mode="r")  # return read-only version


def precompute_and_save_raggedmemmap(raggedmemmap_in: RaggedMmap,
                                     dirname: Union[str, os.PathLike],
                                     transform: callable,
                                     dtype: torch.dtype,
                                     batch_size: int = 128) -> RaggedMmap:
    N = len(raggedmemmap_in)
    RaggedMmap.from_generator(out_dir=dirname,
                              sample_generator=(transform(torch.tensor(raggedmemmap_in[i], dtype=dtype)).numpy()
                                                for i in range(N)),
                              batch_size=batch_size,
                              verbose=False)
    return RaggedMmap(dirname)


def precompute_and_save(data_format: str,
                        data_in: Union[np.memmap, RaggedMmap],
                        preprocess_dir: Union[str, os.PathLike],
                        save_name: Union[str, os.PathLike],
                        transform: callable,
                        dtype: torch.dtype,
                        batch_size: int = 128,
                        return_path=True) -> Union[Union[np.memmap, RaggedMmap],
                                                   Tuple[Union[np.memmap, RaggedMmap], Union[str, os.PathLike]]]:
    if data_format == "memmap":
        save_name = os.path.join(preprocess_dir, f"{save_name}.npy")
        data_out = precompute_and_save_memmap(data_in, save_name, transform, dtype)
    elif data_format == "raggedmemmap":
        save_name = os.path.join(preprocess_dir, save_name)
        data_out = precompute_and_save_raggedmemmap(data_in, save_name, transform, dtype, batch_size)
    else:
        raise ValueError(f"data format {data_format} not supported")

    if return_path:
        return data_out, save_name
    else:
        return data_out


def get_x_dim(x, dim=0):
    if len(x.shape) == 1:
        return x
    n_dim = len(x.shape) - 1
    slicer = []
    for i in range(n_dim):
        if i == dim:
            slicer.append(slice(None))
        else:
            slicer.append(0)
    slicer.append(dim)
    return x[slicer]


class MemMapDataset(Dataset):
    """Load samples of an PDE Dataset, get items according to PDE"""

    def __init__(
        self,
        path: str,
        data_file: str,
        baseline_file: str = None,
        conditioning: str = None,
        t_conditioning: str = None,
        spatial_conditioning: str = None,
        data_transform: callable = None,
        grid_transform: callable = None,
        baseline_transform: callable = None,
        conditioning_transform: callable = None,
        t_conditioning_transform: callable = None,
        spatial_conditioning_transform: callable = None,
        data_format: str = 'memmap',
        raggedmemmap_batch_size: int = 128,
        dtype: torch.dtype = torch.float32,
        preprocess: bool = False,
        preprocess_path: str = None,
        load_all: bool = False,
    ) -> None:
        """Initialize the dataset object
        Args:
            path: path to dataset
            pde: PDE object corresponding to the dataset

            data_file: str of object containing the dataset
            baseline_file: str of object containing the baseline dataset (optional; None = do not use)
            conditioning: str of object containing the static conditioning dataset (optional; None = do not use)
            t_conditioning: str of object containing the time-varying conditioning dataset (optional; None = do not use)

            data_transform: function to apply to the dataset, element-by-element
            grid_transform: function to apply to the grid
            baseline_transform: function to apply to the baseline data, element-by-element
            conditioning_transform: function to apply to the static conditioning data, element-by-element
            t_conditioning_transform: function to apply to the time conditioning data, element-by-element

            data_format: format of datasets, "memmap" [fixed size] or "raggedmemmap" [variable size]
            raggedmemmap_batch_size: when preprocessing raggedmemmap, this is the batch_size for saving them

            dtype: data type / precision

            preprocess: whether to preprocess: If True, all transforms are done up front and saved. The interpolated
                        data is removed on object deletion/python interpreter close
            preprocess_path: (temp) directory to save the preprocessed data in
            load_all: whether to load all data into memory
        Returns:
            None
        """
        super().__init__()
        self.dtype = dtype
        assert data_format in ['memmap', 'raggedmemmap'], \
            "data format must be memmap (numpy) or raggedmemmap (numpy+mmap_ninja)"
        self.data_format = data_format
        self.return_baseline = baseline_file is not None
        self.return_conditioning = conditioning is not None
        self.return_t_conditioning = t_conditioning is not None
        self.return_spatial_conditioning = spatial_conditioning is not None

        # save transforms
        self.data_transform = data_transform
        self.grid_transform = grid_transform
        self.baseline_transform = baseline_transform if self.return_baseline else None
        self.conditioning_transform = conditioning_transform if self.return_conditioning else None
        self.t_conditioning_transform = t_conditioning_transform if self.return_t_conditioning else None
        self.spatial_conditioning_transform = spatial_conditioning_transform if self.return_spatial_conditioning else None

        # check if we want to preprocess
        self.preprocess = preprocess
        if all(v is None for v in [self.data_transform, self.baseline_transform,
                                   self.conditioning_transform, self.t_conditioning_transform]):
            if self.preprocess:
                print("Overriding preprocess to False, since no transforms were specified")
                self.preprocess = False

        if self.preprocess:
            if preprocess_path is not None:
                self.preprocess_dir = preprocess_path
            else:
                self.preprocess_dir = os.path.join(path, "tmp")
            os.makedirs(self.preprocess_dir, exist_ok=True)
        else:
            self.preprocess_dir = None

        self.data = {"data": load_data(self.data_format, path, data_file)}
        if self.return_baseline:
            self.data["baseline"] = load_data(self.data_format, path, baseline_file)
        if self.return_conditioning:
            self.data["conditioning"] = load_data(self.data_format, path, conditioning)
        if self.return_t_conditioning:
            self.data["t_conditioning"] = load_data(self.data_format, path, t_conditioning)
        if self.return_spatial_conditioning:
            self.data["spatial_conditioning"] = load_data(self.data_format, path, spatial_conditioning)

        # process config
        self.config = load_yaml(os.path.join(path, data_file + ".yaml"))
        if "x" in self.config:  # if we have 1D grid, simply load x
            self.x = torch.tensor(self.config["x"], dtype=self.dtype)
            self.x_all = [self.x]
        else:  # otherwise, scan for x1, x2, etc.
            x_keys = [x for x in self.config if x.startswith("x")]  # start with x
            x_keys = [int(x[1:]) for x in x_keys if str.isdigit(x[1:])]  # get all that are ints
            if set(range(1, len(x_keys)+1)) != set(x_keys):
                raise ValueError(f"Found grid keys {['x'+str(x) for x in x_keys]}, "
                                 f"expected keys {['x'+str(x) for x in range(1, len(x_keys)+1)]}")
            if len(x_keys) == 0:
                raise ValueError(f"Could not find a grid in {data_file}.yaml")
            x_keys = ['x'+str(x) for x in x_keys]
            x_keys.sort()
            self.x_all = [torch.tensor(self.config[x], dtype=self.dtype) for x in x_keys]
            if len(self.x_all) == 1:
                self.x = self.x_all[0]
            else:
                # self.x = (x1, x2, 2) for 2 dims, (x1, x2, x3, 3) for 3 dims etc.
                self.x = torch.stack(torch.meshgrid(*self.x_all, indexing="ij"))
                self.x = torch.movedim(self.x, 0, -1)
        self.tmin = self.config["tmin"]
        self.tmax = self.config["tmax"]
        self.dt = self.config["dt"]

        # handle preprocessing
        if self.grid_transform is not None:  # grid is constant -> apply transform in init
            self.x = self.grid_transform(self.x)

        if self.preprocess:
            postfix = util.random_timestr()
            print(f"Preprocessing dataset '{path}'")
            self.preprocess_output = {}
            for data_name, return_data, transform in \
                    [("data",          True,                       self.data_transform),
                     ("baseline",       self.return_baseline,       self.baseline_transform),
                     ("conditioning",   self.return_conditioning,   self.conditioning_transform),
                     ("t_conditioning", self.return_t_conditioning, self.t_conditioning_transform),
                     ("spatial_conditioning", self.return_spatial_conditioning, self.spatial_conditioning_transform)]:
                if return_data is None:  # skip data we do not handle at all
                    continue
                if transform is None:  # print that we skip transform
                    print(f"No transform specified for {data_name}, skipping.")
                    continue
                # otherwise, handle the transform
                self.data[data_name], path = precompute_and_save(
                    data_format=self.data_format, data_in=self.data[data_name], preprocess_dir=self.preprocess_dir,
                    save_name=f"{data_name}_{postfix}", transform=transform, dtype=self.dtype,
                    batch_size=raggedmemmap_batch_size)
                print(f"Preprocessed {data_name}, saved to {path}")
                self.preprocess_output[data_name] = path
            # register finalizer
            self._finalizer = weakref.finalize(self, self._delete_preprocess_files)

        if load_all:
            data = {k: v[:] for k, v in self.data if
                    isinstance(v, torch.Tensor) or isinstance(v, np.memmap) or isinstance(v, np.ndarray)}
            self.data = data

    def cleanup(self):
        self._finalizer()

    def _delete_preprocess_files(self):
        # clean up preprocessed datasets at the end
        for data_name in ["data", "baseline", "conditioning", "t_conditioning"]:
            if data_name in self.preprocess_output:
                if self.data_format == "memmap":
                    self.data[data_name]._mmap.close()
                    os.remove(self.preprocess_output[data_name])
                elif self.data_format == "raggedmemmap":
                    del self.data[data_name]
                    shutil.rmtree(self.preprocess_output[data_name])

    def __len__(self):
        if self.data_format == "memmap":
            return self.data["data"].shape[0]
        elif self.data_format == "raggedmemmap":
            return len(self.data["data"])

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data item
        Args:
            idx (int): data index
        Returns:
            torch.Tensor: baseline trajectory (optional, torch.empty(0) otherwise)
            torch.Tensor: trajectory used for training
            torch.Tensor: spatial coordinates
            torch.Tensor: simulation-wide conditioning
            torch.Tensor: time-varying conditioning
            torch.Tensor: spatial_conditioning (constant throughout simulation)
        """
        # load all variables
        u = torch.tensor(self.data["data"][idx], dtype=self.dtype)
        x = self.x
        if self.return_baseline:
            u_base = torch.tensor(self.data["baseline"][idx], dtype=self.dtype)
        else:
            u_base = torch.empty(0)
        if self.return_conditioning:
            conditioning = torch.tensor(self.data["conditioning"][idx], dtype=self.dtype)
        else:
            conditioning = torch.empty(0)
        if self.return_t_conditioning:
            t_conditioning = torch.tensor(self.data["t_conditioning"][idx], dtype=self.dtype)
        else:
            t_conditioning = torch.empty(0)
        if self.return_spatial_conditioning:
            spatial_conditioning = torch.tensor(self.data['spatial_conditioning'][idx], dtype=self.dtype)
        else:
            spatial_conditioning = torch.empty(0)
        # apply dynamic transforms if applicable
        if not self.preprocess:
            if self.data_transform is not None:
                u = self.data_transform(u)
            if self.baseline_transform is not None:
                u_base = self.baseline_transform(u_base)
            if self.conditioning_transform is not None:
                conditioning = self.conditioning_transform(conditioning)
            if self.t_conditioning_transform is not None:
                t_conditioning = self.t_conditioning_transform(t_conditioning)
            if self.spatial_conditioning_transform is not None:
                spatial_conditioning = self.spatial_conditioning_transform(spatial_conditioning)
        return u_base, u, x, conditioning, t_conditioning, spatial_conditioning

