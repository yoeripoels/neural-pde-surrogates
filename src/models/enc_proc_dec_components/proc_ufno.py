import torch
from torch import nn
from torch.nn import functional as F
from pdes import PDE
from models.common import get_conv_with_right_spatial_dim
from models.base import ModelInterface
from models.enc_proc_dec_components.proc_fno import FNO_Layer
from models.enc_proc_dec_components.proc_unet_modern import UNetModern
from common.interfaces import D, M
from typing import List, Optional, Tuple, Union

"""
Main implementation:
[1] Li, Zongyi, et al. "Fourier Neural Operator for Parametric Partial Differential Equations." 
International Conference on Learning Representations. 2020.

Conditioning:
[2] Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer." 
Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

Implemented in 1, 2 and 3 dimensions.
"""


class UFNO(nn.Module):
    model_interface = M.AR_TB
    data_interface = [D.sim1d, D.sim1d_var_t, D.sim2d]

    def __init__(self,
                 pde: PDE,
                 num_spatial_dims: int = 1,
                 n_cond: int = 0,
                 hidden_features: int = 128,
                 hidden_blocks: int = 4,
                 cond_mode: str = "concat",
                 padding_mode: str = "circular",

                 # FNO specific
                 fno_modes: int = 48,
                 fno_kernel_size: int = 1,
                 fno_conv_mode: str = "single",

                 # UNet specific
                 activation: nn.Module = nn.GELU(),
                 norm: bool = False,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 1, 1),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False),
                 mid_attn: bool = False,
                 n_blocks: int = 1,
                 use1x1: bool = True,
                 **kwargs):
        super().__init__()
        self.pde = pde
        self.num_spatial_dims = num_spatial_dims
        self.cond_mode = cond_mode
        self.activation = activation

        assert self.cond_mode in ["film", "concat", None], "Incorrect conditioning mode supplied"

        if self.cond_mode == "film":
            feature_transform = n_cond > 0
            feature_transform_dim = n_cond
            hidden_dim_in = hidden_features
        elif self.cond_mode == "concat":
            feature_transform = False
            feature_transform_dim = 0
            hidden_dim_in = hidden_features + n_cond
        else:
            feature_transform = False
            feature_transform_dim = 0
            hidden_dim_in = hidden_features

        self.fno_layers = nn.ModuleList([FNO_Layer(
            hidden_dim=hidden_dim_in,
            hidden_dim_out=hidden_features,
            num_spatial_dims=num_spatial_dims,
            modes=fno_modes,
            feature_transform=feature_transform,
            feature_transform_dim=feature_transform_dim,
            kernel_size=fno_kernel_size,
            conv_mode=fno_conv_mode,
            padding_mode=padding_mode if padding_mode != "ones" else "zeros",
            activation=None,
        ) for _ in range(hidden_blocks)])

        self.unet_layers = nn.ModuleList([UNetModern(
            pde=pde,
            num_spatial_dims=num_spatial_dims,
            n_cond=n_cond,
            hidden_features=hidden_features,
            cond_mode=cond_mode,
            activation=activation,
            norm=norm,
            ch_mults=ch_mults,
            is_attn=is_attn,
            mid_attn=mid_attn,
            n_blocks=n_blocks,
            use1x1=use1x1,
            padding_mode=padding_mode,
        ) for _ in range(hidden_blocks)])

    def __repr__(self):
        return f'U-FNO{self.num_spatial_dims}D'

    def forward(self, h: torch.Tensor, variables: torch.Tensor = None, variables_broadcast: torch.Tensor = None, pos=None):
        for i in range(len(self.fno_layers)):
            if self.cond_mode == "film":
                h_fno = self.fno_layers[i](h, p=variables)  # apply FNO
            elif self.cond_mode == "concat":
                if variables_broadcast is not None:
                    h_in = torch.cat([h, variables_broadcast], dim=1)  # concat channel dim with conditioning signal
                else:
                    h_in = h
                h_fno = self.fno_layers[i](h_in)  # apply FNO, no conditioning inside
            else:
                raise ValueError(f"Unknown cond_mode {self.cond_mode}")
            h_unet = self.unet_layers[i](h=h, variables_broadcast=variables_broadcast, pos=pos)
            h = self.activation(h_fno + h_unet)
        return h
