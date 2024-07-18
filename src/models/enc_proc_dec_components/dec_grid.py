import torch
from torch import nn
from pdes import PDE
from models.common import get_conv_with_right_spatial_dim, Swish
import math


def add_delta(delta, u, pde_dt, time_window, num_spatial_dims, delta_mode='per_step', delta_dt=True):
    if delta_dt is False:
        pde_dt = 1
    assert delta_mode in ['per_step',  'all', 'none']
    if delta_mode == 'per_step':
        # Helper function to update using dt + last timestep, adapted from formula 10 of https://arxiv.org/abs/2202.03376
        dt = (torch.ones(1, 1, time_window) * pde_dt).to(delta.device)  # construct dt tensor
        dt = torch.cumsum(dt, dim=2)
        for _ in range(num_spatial_dims):
            dt = torch.unsqueeze(dt[:], -1)

        # get domain at last timestep and broadcast to time_window
        u_last_full = u[:, :, [-1], ...].repeat(1, 1, time_window, *(1 for _ in range(num_spatial_dims)))

        # scale output according to dt and add to last timestep
        return u_last_full + dt * delta
    elif delta_mode == 'all':
        # get domain at last timestep and broadcast to time_window
        u_last_full = u[:, :, [-1], ...].repeat(1, 1, time_window, *(1 for _ in range(num_spatial_dims)))
        return u_last_full + pde_dt * delta
    elif delta_mode == 'none':
        return delta
    else:
        raise ValueError(f"Unrecognized dec_delta_mode {delta_mode}")


class LinearConv(nn.Module):
    """Simple convolution (with no activation)
    """
    def __init__(self, pde: PDE, num_c, num_spatial_dims, time_window, hidden_features, dec_kernel_size, dec_padding_mode, dec_delta_mode='per_step', dec_delta_dt=True, **kwargs):
        super().__init__()
        self.pde = pde
        self.num_spatial_dims = num_spatial_dims
        self.time_window = time_window
        self.dec_delta_mode = dec_delta_mode
        self.dec_delta_dt = dec_delta_dt

        self.decoder = get_conv_with_right_spatial_dim(num_spatial_dims,
                                                       in_channels=hidden_features,
                                                       out_channels=num_c * time_window,
                                                       kernel_size=dec_kernel_size,
                                                       padding="same",
                                                       padding_mode=dec_padding_mode)

    def forward(self, h: torch.Tensor, u: torch.Tensor, **kwargs):
        delta = self.decoder(h)  # get decoder output
        delta = delta.view(u.shape)  # [b, c * tw, *spatial_dims] -> [b, c, tw, *spatial_dims]
        return add_delta(delta, u, self.pde.dt, self.time_window, self.num_spatial_dims, delta_mode=self.dec_delta_mode, delta_dt=self.dec_delta_dt)


class TimeConv(nn.Module):
    """Nonlinear convolution over time window
    """
    def __init__(self, pde: PDE, num_c, num_spatial_dims, time_window, hidden_features, dec_delta_mode='per_step', dec_delta_dt=True, **kwargs):
        super().__init__()
        self.pde = pde
        self.num_spatial_dims = num_spatial_dims
        self.time_window = time_window
        self.num_c = num_c
        self.dec_delta_mode = dec_delta_mode
        self.dec_delta_dt = dec_delta_dt

        # formula to get kernel / stride for CNN decoder so it gets the desired output size
        var = time_window + 9
        stride = hidden_features // var
        assert stride > 0, "found stride 0 -- most likely, hidden_features is too small!"
        kernelsize = hidden_features - stride * var + 1
        self.decoder = nn.Sequential(nn.Conv1d(1, 8, kernelsize, stride=stride),
                                     Swish(),
                                     nn.Conv1d(8, num_c, 10, stride=1))

    def forward(self, h: torch.Tensor, u: torch.Tensor, **kwargs):
        # reshape output s.t. we can convolve over time window
        spatial_axes = [i for i in range(2, self.num_spatial_dims + 2)]
        h = torch.permute(h, dims=(0, *spatial_axes, 1))  # go from b, h, *spatial to b, *spatial, h
        batch, *spatial, hid = h.shape
        h = torch.flatten(h, 0, self.num_spatial_dims)  # now go to (b* \prod_i |spatial_i|), h

        # [batch*nx, h] -> 1DCNN([batch*nx, 1, h]) -> [batch*nx, t]
        delta = self.decoder(h[:, None])
        delta = delta.view(batch, *spatial, self.num_c, self.time_window)  # "unflatten"

        spatial_axes = [i for i in range(1, self.num_spatial_dims + 1)]
        delta = torch.permute(delta, (
            0, self.num_spatial_dims + 1, self.num_spatial_dims + 2, *spatial_axes))  # (b, c, t, *spatial)

        return add_delta(delta, u, self.pde.dt, self.time_window, self.num_spatial_dims, delta_mode=self.dec_delta_mode, delta_dt=self.dec_delta_dt)


class TimeConvDense(nn.Module):
    """Nonlinear convolution over time window
    """
    def __init__(self, pde: PDE, num_c, num_spatial_dims, time_window, hidden_features, activation, dec_delta_mode='per_step', dec_delta_dt=True, **kwargs):
        super().__init__()
        self.pde = pde
        self.num_spatial_dims = num_spatial_dims
        self.time_window = time_window
        self.num_c = num_c
        self.dec_delta_mode = dec_delta_mode
        self.dec_delta_dt = dec_delta_dt

        # reshape to preset dim
        decoder_input_dim = time_window * 3 * num_c
        self.pre_decoder = get_conv_with_right_spatial_dim(num_spatial_dims,
                                                       in_channels=hidden_features,
                                                       out_channels=decoder_input_dim,
                                                       kernel_size=1)

        # formula to get kernel / stride for CNN decoder so it gets the desired output size
        kernel_size_a = math.ceil(time_window / 2)
        kernel_size_b = math.ceil(time_window / 4) + 1
        if time_window % 4 == 0:
            kernel_size_b += 1

        self.decoder = nn.Sequential(nn.Conv1d(num_c, num_c*2, kernel_size_a, stride=2),
                                     activation,
                                     nn.Conv1d(num_c*2, num_c, kernel_size_b, stride=1))

    def forward(self, h: torch.Tensor, u: torch.Tensor, **kwargs):
        # go to output dim for decoder
        h = self.pre_decoder(h)

        # reshape output s.t. we can convolve over time window
        spatial_axes = [i for i in range(2, self.num_spatial_dims + 2)]
        h = torch.permute(h, dims=(0, *spatial_axes, 1))  # go from b, h, *spatial to b, *spatial, h
        batch, *spatial, hid = h.shape
        spatial_prod = math.prod(spatial)
        h = torch.flatten(h, 0, self.num_spatial_dims).view(batch * spatial_prod, self.num_c,
                                                       self.time_window * 3)

        # [batch*nx, h, num_c] -> [batch*nx, num_c, h]
        delta = self.decoder(h)
        delta = delta.view(batch, *spatial, self.num_c, self.time_window)  # "unflatten"

        spatial_axes = [i for i in range(1, self.num_spatial_dims + 1)]
        delta = torch.permute(delta, (
            0, self.num_spatial_dims + 1, self.num_spatial_dims + 2, *spatial_axes))  # (b, c, t, *spatial)

        return add_delta(delta, u, self.pde.dt, self.time_window, self.num_spatial_dims, delta_mode=self.dec_delta_mode, delta_dt=self.dec_delta_dt)


class TimeConvLinear(nn.Module):
    """Linear convolution over time window
    """
    def __init__(self, pde: PDE, num_c, num_spatial_dims, time_window, hidden_features, activation, dec_delta_mode='per_step', dec_delta_dt=True, **kwargs):
        super().__init__()
        self.pde = pde
        self.num_spatial_dims = num_spatial_dims
        self.time_window = time_window
        self.num_c = num_c
        self.dec_delta_mode = dec_delta_mode
        self.dec_delta_dt = dec_delta_dt

        # reshape to preset dim
        self.decoder_input_dim = time_window * 3 - 1 - math.ceil((time_window-1)/2)
        if time_window == 1:
            self.decoder_input_dim -= 1
        self.pre_decoder = get_conv_with_right_spatial_dim(num_spatial_dims,
                                                       in_channels=hidden_features,
                                                       out_channels=self.decoder_input_dim * num_c,
                                                       kernel_size=1)

        # formula to get kernel / stride for CNN decoder so it gets the desired output size
        kernel_size_a = math.ceil(time_window / 2)

        self.decoder = nn.Conv1d(num_c, num_c, kernel_size_a, stride=2)

    def forward(self, h: torch.Tensor, u: torch.Tensor, **kwargs):
        # go to output dim for decoder
        h = self.pre_decoder(h)

        # reshape output s.t. we can convolve over time window
        spatial_axes = [i for i in range(2, self.num_spatial_dims + 2)]
        h = torch.permute(h, dims=(0, *spatial_axes, 1))  # go from b, h, *spatial to b, *spatial, h
        batch, *spatial, hid = h.shape
        spatial_prod = math.prod(spatial)
        h = torch.flatten(h, 0, self.num_spatial_dims).view(batch * spatial_prod, self.num_c,
                                                       self.decoder_input_dim)

        # [batch*nx, h, num_c] -> [batch*nx, num_c, h]
        delta = self.decoder(h)
        delta = delta.view(batch, *spatial, self.num_c, self.time_window)  # "unflatten"

        spatial_axes = [i for i in range(1, self.num_spatial_dims + 1)]
        delta = torch.permute(delta, (
            0, self.num_spatial_dims + 1, self.num_spatial_dims + 2, *spatial_axes))  # (b, c, t, *spatial)

        return add_delta(delta, u, self.pde.dt, self.time_window, self.num_spatial_dims, delta_mode=self.dec_delta_mode, delta_dt=self.dec_delta_dt)
