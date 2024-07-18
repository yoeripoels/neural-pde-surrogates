import torch
import torch.nn.functional as F
import numpy as np
import math

def downsample_1d_average_periodic_conv(u_super, ratio_nt, ratio_nx, has_batch=False, smooth=True):
    if not has_batch:
        u_super = u_super[:: ratio_nt][None, None, ...]  # add batch + dummy dim
    else:
        batch_size = u_super.shape[0]
        u_super = u_super[:, :: ratio_nt][:, None, ...]  # only add batch
    if smooth == True:
        left = u_super[..., -3:-1]
        right = u_super[..., 1:3]
        u_super_padded = torch.tensor(np.concatenate((left, u_super, right), -1))
        weights = torch.FloatTensor([[[[0.2] * 5]]])
        u_super = F.conv2d(
            u_super_padded, weights, stride=(1, ratio_nx)
        ).squeeze()
    else:
        u_super = u_super[..., :: ratio_nx].squeeze()
    if has_batch and batch_size == 1:  # if batch size was 1, it got squeezed away!
        u_super = u_super.unsqueeze(0)
    return u_super

def downsample_2d_average_periodic_conv(u_super, ratio_nt, ratio_nx):
    u_super = u_super[:: ratio_nt]
    left_x = u_super[:, -3:-1]
    right_x = u_super[:, 1:3]
    # Pad u_super
    u_super = torch.cat((left_x, u_super, right_x), 1)
    left_y = u_super[:, :, -3:-1]
    right_y = u_super[:, :, 1:3]
    u_super = torch.cat((left_y, u_super, right_y), 2)

    weights = torch.full((1, 1, 5, 5), 1.0 / (5 * 5))
    u_super = F.conv2d(
        u_super[:, None], weights, stride=(ratio_nx, ratio_nx)
    ).squeeze()
    return u_super

def get_1d_downsample_matrix(nx_in, nx_out, dtype=np.float32):
    assert nx_in > nx_out, "nx_out >= nx_in, this is not downsampling!"
    C = np.zeros((nx_in, nx_out), dtype=dtype)
    C[0, 0] = C[-1, -1] = 1

    grid_a = np.linspace(0, nx_in-1, nx_in)
    grid_b = np.linspace(0, nx_in-1, nx_out)
    time_ratio = (nx_in-1)/(nx_out-1)
    for i in range(1, nx_out-1):
        j = math.floor((i) * time_ratio)
        if grid_b[i] == grid_a[j]:
            C[j, i] = 1
        else:
            dif_a = abs(grid_a[j] - grid_b[i])
            dif_b = abs(grid_a[j+1] - grid_b[i])
            total = dif_a + dif_b
            C[j, i] = (total-dif_a) / total
            C[j+1, i] = (total-dif_b) / total
    return C

def get_1d_averaging_matrix(nx, n_average, boundary='periodic', dtype=np.float32):
    assert n_average < nx, "Cannot smooth over more than the entire domain"
    assert n_average % 2 != 0, "Smoothing domain must be odd"
    C = np.zeros((nx, nx), dtype=dtype)
    for i in range(nx):
        s_i = i - n_average // 2
        e_i = i + n_average // 2 + 1
        if s_i < 0:
            if boundary == 'periodic':
                idx = [j if j >= 0 else j + nx for j in range(s_i, e_i)]
            elif boundary == 'fixed':
                idx = [j for j in range(s_i, e_i) if j >= 0]
        elif e_i >= nx:
            if boundary == 'periodic':
                idx = [j if j < nx else j - nx for j in range(s_i, e_i)]
            elif boundary == 'fixed':
                idx = [j for j in range(s_i, e_i) if j < nx]
        else:
            idx = range(s_i, e_i)
        for j in idx:
            C[j, i] = 1/len(idx)
    return C

def downsample_1d_average_periodic_mm(ratio_nt, nx_in, nx_out, n_average, boundary='periodic', dtype=np.float32, has_c=False):
    do_smooth = n_average > 1
    do_downsample = nx_in > nx_out
    if do_downsample:
        matrix_downsample = get_1d_downsample_matrix(nx_in, nx_out, dtype=dtype)
    else:
        matrix_downsample = None
    if do_smooth:
        matrix_smooth = get_1d_averaging_matrix(nx_in, n_average, boundary=boundary, dtype=dtype)
    else:
        matrix_smooth = None
    if do_downsample and do_smooth:
        matrix_transform = matrix_smooth @ matrix_downsample
    elif do_downsample:
        matrix_transform = matrix_downsample
    elif do_smooth:
        matrix_transform = matrix_smooth
    else:
        matrix_transform = None

    if matrix_transform is not None:
        matrix_transform = torch.from_numpy(matrix_transform)

    def inner_transform(u):
        if not has_c:
            if matrix_transform is not None:
                return torch.matmul(u[::ratio_nt], matrix_transform)
            else:
                return u[::ratio_nt]
        else:
            if matrix_transform is not None:
                return torch.matmul(u[:, ::ratio_nt], matrix_transform)
            else:
                return u[:, ::ratio_nt]
    return inner_transform

def downsample_1d_mm(nx_in, nx_out, dtype=np.float32):
    if nx_in > nx_out:
        matrix_downsample = get_1d_downsample_matrix(nx_in, nx_out, dtype=dtype)
        matrix_downsample = torch.from_numpy(matrix_downsample)
    else:
        matrix_downsample = None

    def inner_transform(x):
        if matrix_downsample is not None:
            return torch.matmul(x, matrix_downsample)
        else:
            return x
    return inner_transform

def get_t_downsample(tmin, tmax, nt_in, nt_out=None, ratio_nt=None):
    tdelta = tmax - tmin
    range_old = [tmin + (x / (nt_in - 1) * tdelta) for x in range(0, nt_in)]
    if nt_out is None and ratio_nt is None:
        raise ValueError("Either nt_out or ratio_nt must be specified")
    elif ratio_nt is None:
        ratio_nt = nt_in / nt_out
    if not isinstance(ratio_nt, int):
        assert ratio_nt.is_integer()
        ratio_nt = int(ratio_nt)
    range_new = range_old[::ratio_nt]
    tmin, tmax = range_new[0], range_new[-1]
    return tmin, tmax

def get_1d_interp_matrix(grid_in, grid_out, dtype=np.float32):
    """
    Returns the matrix to interpolate an array between two non-evenly spaced 1d grids
    Assume both input grids to be sorted
    """
    assert len(grid_in.shape) == len(grid_out.shape) == 1, "grid not 1d"
    assert np.all(grid_in[:-1] <= grid_in[1:]), "grid_in not sorted"
    assert np.all(grid_out[:-1] <= grid_out[1:]), "grid_out not sorted"
    nx_in, nx_out = grid_in.shape[0], grid_out.shape[0]

    C = np.zeros((nx_in, nx_out), dtype=dtype)
    for i in range(nx_out):  # loop over all elements in
        if grid_out[i] <= grid_in[0]:  # start of domain
            C[0, i] = 1
        elif grid_out[i] >= grid_in[-1]:  # hit end of domain
            C[-1, i] = 1
        else:  # in domain
            j = np.searchsorted(grid_in, grid_out[i])
            if grid_in[j] == grid_out[i]:  # found exact match
                C[j, i] = 1
            else:
                dif_a = abs(grid_in[j] - grid_out[i])
                dif_b = abs(grid_in[j - 1] - grid_out[i])
                total = dif_a + dif_b
                C[j, i] = (total - dif_a) / total
                C[j - 1, i] = (total - dif_b) / total
    return C

def get_1d_interp_matrix_to_even(grid_in, nx_out=None, dtype=np.float32):
    if nx_out is None:
        nx_out = grid_in.shape[0]
    grid_out = np.linspace(grid_in[0], grid_in[-1], nx_out)
    return get_1d_interp_matrix(grid_in, grid_out, dtype=dtype)
