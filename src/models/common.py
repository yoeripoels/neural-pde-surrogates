import torch
from torch import nn
import copy
import numpy as np
import torch.nn.functional as F


class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


def crop_Nd(num_spatial_dims, enc_ftrs: torch.Tensor, shape: torch.Tensor):
    if isinstance(shape, torch.Tensor) or isinstance(shape, np.ndarray):
        shape = shape.shape
    else:
        shape = shape
    s_des = shape[-num_spatial_dims:]
    s_current = enc_ftrs.shape[-num_spatial_dims:]
    # first, calculate preliminary paddings - may contain non-integers ending in .5):
    pad_temp = np.repeat(np.subtract(s_des, s_current) / 2, 2)
    # to break the .5 symmetry to round one padding up and one down, we add a small pos/neg number respectively
    # note this will not impact the case where pad_temp[i] is integer since it is still rounded to that integer
    breaking_arr = np.tile([1, -1], int(len(pad_temp) / 2)) / 1000
    pad = tuple(reversed(tuple(map(lambda p: int(round(p)), pad_temp + breaking_arr))))
    enc_ftrs = F.pad(enc_ftrs, pad)
    return enc_ftrs


def get_conv_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        conv = nn.Conv1d(**kwargs)
    elif spatial_dim == 2:
        conv = nn.Conv2d(**kwargs)
    elif spatial_dim == 3:
        conv = nn.Conv3d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=3d convs implemented so far, but found spatial dim {spatial_dim}!')

    return conv


def get_maxpool_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        pool = nn.MaxPool1d(**kwargs)
    elif spatial_dim == 2:
        pool = nn.MaxPool2d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=2d convs implemented so far, but found spatial dim {spatial_dim}!')

    return pool


def circular_pad_2d(x, pad):
    """
    Apply circular padding to the input tensor.

    Args:
    x (torch.Tensor): Input tensor.
    pad (int or tuple): Padding size. If an integer, the same padding is applied on all sides.
                        If a tuple, it represents (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
    torch.Tensor: Circular padded tensor.
    """
    if isinstance(pad, int):
        pad = (pad, pad, pad, pad)
    else:
        assert len(pad) == 4, "Pad must be an integer or a tuple of length 4."

    pad_left, pad_right, pad_top, pad_bottom = pad

    # Padding left and right
    left = x[..., -pad_left:]  # Take last 'pad_left' elements
    right = x[..., :pad_right]  # Take first 'pad_right' elements
    x = torch.cat([left, x, right], dim=-1)

    # Padding top and bottom
    top = x[..., -pad_top:, :]  # Take last 'pad_top' rows
    bottom = x[..., :pad_bottom, :]  # Take first 'pad_bottom' rows
    x = torch.cat([top, x, bottom], dim=-2)

    return x


class ConvTranspose2d_padded(nn.ConvTranspose2d):
    def __init__(self, pad, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = pad

    def forward(self, x):
        x = circular_pad_2d(x, pad=self.pad)
        return super().forward(x)


def get_upconv_with_right_spatial_dim(spatial_dim, in_channels, out_channels, **kwargs):
    if spatial_dim == 1:
        upconv = nn.ConvTranspose1d(in_channels, out_channels, **kwargs)
    elif spatial_dim == 2:
        if 'padding_mode' in kwargs and kwargs["padding_mode"] == 'circular':
            assert "kernel_size" in kwargs
            kernel_size = kwargs["kernel_size"]
            slice_size = (kernel_size - 1) // 2

            kwargs_duplicate = copy.deepcopy(kwargs)
            del kwargs_duplicate["padding_mode"]
            upconv = ConvTranspose2d_padded(slice_size, in_channels, out_channels, **kwargs_duplicate)
        else:
            upconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=2d convs implemented so far, but found spatial dim {spatial_dim}!')

    return upconv


class BCEncoder(nn.Module):
    def __init__(self, bc_encoder_in, bc_encoder_out, bc_encoder_hidden, bc_encoder_kernel, time_window, num_spatial_dims, activation, bc_encoder_n_hidden=1, **kwargs):
        super().__init__()

        bc_enc_kwargs_1 = {'in_channels': bc_encoder_in,
                           'out_channels': bc_encoder_hidden,
                           'kernel_size': bc_encoder_kernel,
                           'padding': 'same',
                           'padding_mode': 'zeros'}
        bc_enc_kwargs_2 = {'in_channels': bc_encoder_hidden,
                           'out_channels': bc_encoder_hidden,
                           'kernel_size': bc_encoder_kernel,
                           'padding': 'same',
                           'padding_mode': 'zeros'}
        bc_encoder_layers = [get_conv_with_right_spatial_dim(num_spatial_dims, **bc_enc_kwargs_1),
                            activation]
        for _ in range(bc_encoder_n_hidden):
            bc_encoder_layers += [get_conv_with_right_spatial_dim(num_spatial_dims, **bc_enc_kwargs_2),
                                  activation]

        bc_encoder_layers += [nn.Flatten(start_dim=1),
                              nn.Linear(time_window * bc_encoder_hidden, bc_encoder_out)]
        self.bc_encoder = nn.Sequential(*bc_encoder_layers)
        self.n_out = bc_encoder_out

    def forward(self, x: torch.Tensor):
        return self.bc_encoder(x)

