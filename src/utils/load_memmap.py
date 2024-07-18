from typing import  Union
import os
from numpy.lib.format import open_memmap
from mmap_ninja.ragged import RaggedMmap
import numpy as np


def load_data(data_format: str, data_dir: Union[str, os.PathLike], load_name: Union[str, os.PathLike]
              ) -> Union[np.memmap, RaggedMmap]:
    if data_format == "memmap":
        return open_memmap(os.path.join(data_dir, load_name + ".npy"), mode="r")
    elif data_format == "raggedmemmap":
        return RaggedMmap(os.path.join(data_dir, load_name))
    else:
        raise ValueError(f"data format {data_format} not supported")
