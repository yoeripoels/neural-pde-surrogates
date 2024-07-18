"""
Helper function to slice/index a batch
"""
import torch
import numpy as np
from collections.abc import Iterable

def handle_item(item, idx, slice_item=None):
    if isinstance(item, torch.Tensor):
        return item[idx]
    elif isinstance(item, np.ndarray):
        return item[idx]
    elif isinstance(item, list):
        return [item[i] for i in idx]
    elif isinstance(item, dict):
        if slice_item is None:
            slice_item = {k: True for k in item}
        assert isinstance(slice_item, dict)
        assert set(slice_item.keys()) == set(item.keys())
        return {k: handle_item(v, idx) if slice_item[k] else v
                    for k, v in item.items()}
    elif item is None:
        return None
    else:
        raise ValueError(f"Unexpected item type: '{type(item)}'")


def index_batch(batch, s=None, e=None, idx=None, slice_item=None):
    if s is not None and e is not None and idx is None:
        idx = torch.arange(s, e)
    elif idx is not None and (s is not None or e is not None):
        raise ValueError("Both and index and a start/end provided!")
    elif idx is None and (s is None or e is None):
        raise ValueError("No index provided, and no start/end provided!")
    if not isinstance(batch, Iterable) or isinstance(batch, dict):
        return handle_item(batch, idx, slice_item=slice_item)
    else:
        if slice_item is None:
            slice_item = [True for _ in batch]
        assert isinstance(slice_item, Iterable)
        assert len(slice_item) == len(batch)
        batch_out = []
        for item, do_slice in zip(batch, slice_item):
            if do_slice:
                batch_out.append(handle_item(item, idx))
            else:
                batch_out.append(item)
        return batch_out
