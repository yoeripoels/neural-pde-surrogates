import torch
from torch import nn
from pdes import PDE
from models.common import get_conv_with_right_spatial_dim
from common.interfaces import D, M

"""
[1] Stachenfeld, Kimberly et al. "Learned Simulators for Turbulence". 
International Conference on Learning Representations, 2022.

Implemented in 1, 2 and 3 dimensions.
"""


class DilatedResnet(nn.Module):
    model_interface = M.AR_TB
    data_interface = [D.sim1d, D.sim2d, D.sim1d_var_t]

    def __init__(self,
                 pde: PDE,
                 hidden_features: int = 128,
                 kernel_size: int = 3,  # note: for 2d assume square kernel
                 hidden_blocks: int = 4,
                 activation: nn.Module = nn.ReLU(),
                 padding_mode: str = 'zeros',
                 num_spatial_dims: int = 1,
                 n_cond: int = 0, **kwargs):

        super().__init__()
        dilation_rates = (1, 2, 4, 8, 4, 2, 1)
        self.num_spatial_dims = num_spatial_dims
        self.processor = nn.Sequential(*[
            DilatedResnetBlock(num_spatial_dims, hidden_features + n_cond,
                               kernel_size,
                               dilation_rates, activation, padding_mode,
                               hidden_features_out=hidden_features)
            for _ in range(hidden_blocks)
        ])

    def __repr__(self):
        return f"DRN{self.num_spatial_dims}D"

    def forward(self, h: torch.Tensor, variables_broadcast: torch.Tensor = None, pos=None):  # pos is ignored, not necessary for dilresnet
        for block in self.processor.children():  # each child module of the processor is one DilatedResnetBlock
            if variables_broadcast is not None:
                enc_input = torch.cat([h, variables_broadcast], dim=1)
            else:
                enc_input = h
            h = h + block(enc_input)  # residual connection, note: activation function modules included in these blocks!
        return h


class DilatedResnetBlock(nn.Module):
    def __init__(self, num_spatial_dims=1, hidden_features_in=48, kernel_size=3, dilation_rates=(1, 2, 4, 8, 4, 2, 1),
                 activation: nn.Module = nn.ReLU(), padding_mode: str = 'zeros', hidden_features_out=None):

        super().__init__()
        self.num_spatial_dims = num_spatial_dims
        self.hidden_features_in = hidden_features_in
        if hidden_features_out is not None:
            self.hidden_features_out = hidden_features_out
        else:
            self.hidden_features_out = hidden_features_in
        self.dilation_rates = dilation_rates
        self.activation = activation
        self.padding_mode = padding_mode

        layer_list = []

        for l, dilrate in enumerate(dilation_rates):
            convkwargs = {'in_channels': self.hidden_features_in if l == 0 else self.hidden_features_out,
                          'out_channels': self.hidden_features_out,
                          'kernel_size': kernel_size,
                          'padding': 'same',
                          'dilation': dilrate,
                          'padding_mode': self.padding_mode}
            conv = get_conv_with_right_spatial_dim(num_spatial_dims, **convkwargs)
            layer_list.append(conv)
            layer_list.append(self.activation)

        self.layers = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor):
        return self.layers(x)  # layers already contains activation modules.
    # should we also add skip connections within each block? Doesn't seem so from the paper (fig 1a) but not sure
