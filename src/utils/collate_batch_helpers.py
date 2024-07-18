import torch
import numpy as np
from collections import defaultdict


def collate_batch_sim(t_dim=1, mode='min', tw=25):
    """
    Combine items with a time dimension s.t. all are of equal size
    Either by minimum (mode='min') -> sample range for longer simulations
        or by maximum (mode='max') -> pad ends with 0s
    """
    def collate_fn(batch_list):
        shapes = [x.shape for x in batch_list]
        if mode == 'min':
            shape_t = min([x[t_dim] for x in shapes])
        elif mode == 'max':
            shape_t = max([x[t_dim] for x in shapes])
        else:
            raise ValueError("Combining mode must be 'min' or 'max'")
        # round up to nearest tw window
        if mode == 'max':
            to_add = tw - (shape_t % tw) if shape_t % tw != 0 else 0
            shape_t += to_add
        elif mode == 'min':
            to_remove = (shape_t % tw) if shape_t % tw != 0 else 0
            shape_t -= to_remove
        shape = torch.tensor(shapes[0])
        shape[t_dim] = shape_t
        batch_size = len(batch_list)
        batch_data = torch.zeros((batch_size, *shape), device=batch_list[0].device)
        for i in range(batch_size):
            t_len = shapes[i][t_dim]
            if mode == 'max':
                indexing = [i]
                for _ in range(t_dim):
                    indexing.append(slice(None, None, None))
                indexing.append(slice(None, t_len))

                batch_data[indexing] = batch_list[i]
            elif mode == 'min':
                start_idx = np.random.randint(0, t_len - shape_t+1)
                indexing = []
                for _ in range(t_dim):
                    indexing.append(slice(None, None, None))
                indexing.append(slice(start_idx, start_idx+shape_t))
                batch_data[i] = batch_list[i][indexing]

        return batch_data
    return collate_fn


def collate_data(t_dim=1, mode='min', return_lengths=True, tw=25):
    """
    Collate all data, padding simulations of unequal length
    If return_lengths=True, we additionally return the lengths of the individual simulations
    (useful info when padding, i.e. when mode='max')
    """

    def collate_memmap_var_t(batch):
        collate_u = collate_batch_sim(t_dim=t_dim, mode=mode, tw=tw)
        assert len(batch[0]) == 5  # 5 items per batch
        data_u_base = [x[0] for x in batch]
        data_u_super = [x[1] for x in batch]
        data_x = [x[2] for x in batch]
        data_variables = [x[3] for x in batch]
        data_conditioning = [x[4] for x in batch]
        if sum([torch.numel(x) for x in data_u_base]) > 0:
            data_u_base = collate_u(data_u_base)
        else:
            data_u_base = torch.stack(data_u_base, dim=0)
        if sum([torch.numel(x) for x in data_conditioning]) > 0:
            data_conditioning = collate_u(data_conditioning)
        else:
            data_conditioning = torch.stack(data_conditioning, dim=0)

        data_lengths = [x.shape[t_dim] for x in data_u_super]
        data_u_super = collate_u(data_u_super)
        data_x = torch.stack(data_x, dim=0)
        data_variables = torch.stack(data_variables, dim=0)

        if return_lengths:
            return data_u_base, data_u_super, data_x, data_variables, data_conditioning, data_lengths
        else:
            return data_u_base, data_u_super, data_x, data_variables, data_conditioning
    return collate_memmap_var_t


def create_data_mask(data, t_lengths, t_dim=1):
    """
    Create mask according to lengths of specified t_dim, where mask=1 if it exists and mask=0 if not
    """
    mask = torch.zeros_like(data, device=data.device, dtype=data.dtype)
    batch_size = data.shape[0]
    for i in range(batch_size):
        indexing = [i]  # create our indexing according to variable t_dim, where dim 0 is ignored / reserved for batch
        for _ in range(t_dim):
            indexing.append(slice(None, None, None))
        indexing.append(slice(0, t_lengths[i]))
        for _ in range(t_dim+2, len(data.shape)):
            indexing.append(slice(None, None, None))
        mask[indexing] = 1
    return mask