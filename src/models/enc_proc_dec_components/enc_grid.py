import torch
from torch import nn
from pdes import PDE
from models.common import get_conv_with_right_spatial_dim, Swish


class LinearConv(nn.Module):
    """Simple convolution (with no activation)
    """
    def __init__(self, pde: PDE, num_c, num_spatial_dims, time_window, hidden_features, enc_kernel_size, enc_padding_mode, **kwargs):
        super().__init__()
        self.encoder = get_conv_with_right_spatial_dim(num_spatial_dims,
                                                       in_channels=num_c * time_window,
                                                       out_channels=hidden_features,
                                                       kernel_size=enc_kernel_size,
                                                       padding="same",
                                                       padding_mode=enc_padding_mode)

    def forward(self, u: torch.Tensor, **kwargs):
        h = torch.flatten(u, start_dim=1, end_dim=2)  # [b, c, tw, *spatial_dims] -> [b, c*tw, *spatial_dims]
        return self.encoder(h)  # encode to h


class ElementWise(nn.Module):
    """Point-wise encoder, adapted from https://arxiv.org/abs/2202.03376 for grids
    """
    def __init__(self, pde: PDE, num_c, num_spatial_dims, time_window, hidden_features, n_cond, activation=Swish(), **kwargs):
        super().__init__()
        num_channels = num_c * time_window  # flatten the temporal bundling + channels
        self.encoder = nn.Sequential(
            get_conv_with_right_spatial_dim(num_spatial_dims,
                                            in_channels=num_channels + num_spatial_dims + n_cond,
                                            out_channels=hidden_features, kernel_size=1),
            activation,
            get_conv_with_right_spatial_dim(num_spatial_dims,
                                            in_channels=hidden_features,
                                            out_channels=hidden_features, kernel_size=1),
            activation,
        )

    def forward(self, u: torch.Tensor, pos: torch.Tensor, variables_broadcast: torch.Tensor = None, **kwargs):
        h = torch.flatten(u, start_dim=1, end_dim=2)  # [b, c, tw, *spatial_dims] -> [b, c*tw, *spatial_dims]
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(-1)  # if no dimension added, add 1 to the end
        pos = torch.movedim(pos, -1, 1)  # move num_dim to channel dim; [b, *spatial_dims, nd] -> [b, nd, *spatial_dims]
        if variables_broadcast is not None:
            h = torch.cat([h, pos, variables_broadcast], dim=1)  # concat the broadcasted variables
        else:
            h = torch.cat([h, pos], dim=1)  # only concat position
        return self.encoder(h)  # encode to h
