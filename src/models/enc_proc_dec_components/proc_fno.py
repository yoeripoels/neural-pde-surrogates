import torch
from torch import nn
from torch.nn import functional as F
from pdes import PDE
from models.common import get_conv_with_right_spatial_dim
from models.base import ModelInterface
from common.interfaces import D, M

"""
Main implementation:
[1] Li, Zongyi, et al. "Fourier Neural Operator for Parametric Partial Differential Equations." 
International Conference on Learning Representations. 2020.

Conditioning:
[2] Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer." 
Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

Implemented in 1, 2 and 3 dimensions.
"""


class FNO(nn.Module):
    model_interface = M.AR_TB
    data_interface = [D.sim1d, D.sim1d_var_t, D.sim2d]

    def __init__(self,
                 pde: PDE,
                 num_spatial_dims: int = 1,
                 n_cond: int = 0,
                 hidden_features: int = 128,
                 fno_modes: int = 48,
                 hidden_blocks: int = 4,
                 cond_mode: str = "concat",
                 fno_kernel_size: int = 1,
                 fno_conv_mode: str = "single",
                 padding_mode: str = "circular",
                 **kwargs):
        super().__init__()
        self.pde = pde
        self.num_spatial_dims = num_spatial_dims
        self.cond_mode = cond_mode

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
        ) for _ in range(hidden_blocks)])

    def __repr__(self):
        return f'FNO{self.num_spatial_dims}D'

    def forward(self, h: torch.Tensor, variables: torch.Tensor = None, variables_broadcast: torch.Tensor = None, pos=None):
        for i in range(len(self.fno_layers)):
            if self.cond_mode == "film":
                h = self.fno_layers[i](h, p=variables)  # apply FNO
            elif self.cond_mode == "concat":
                if variables_broadcast is not None:
                    h_in = torch.cat([h, variables_broadcast], dim=1)  # concat channel dim with conditioning signal
                else:
                    h_in = h
                h = self.fno_layers[i](h_in)  # apply FNO, no conditioning inside
        return h

## the below is built upon
# https://github.com/zongyi-li/fourier_neural_operator/blob/fd7b0ded5370861c43cc1e014ff70e0d047d6fe9/fourier_1d.py
class FNO_Layer(nn.Module):
    '''
    Settings:
    0 = default (paper)
    '''

    def __init__(self, hidden_dim, num_spatial_dims: int = 1,
                 kernel_size=1, modes=16, activation=nn.GELU, activation_params=None,
                 feature_transform=False, feature_transform_dim=6, transform_mode=0, hidden_dim_out=None,
                 conv_mode="single", padding_mode="circular"):
        super(FNO_Layer, self).__init__()
        self.num_spatial_dims = num_spatial_dims
        assert conv_mode in ["single", "double"]
        self.conv_mode = conv_mode

        if isinstance(modes, int):
            modes = tuple([modes for _ in range(num_spatial_dims)])  # tuple

        assert len(modes) == num_spatial_dims, 'modes should be int or tuple of ints with length equal to spatial dim!'
        self.modes = modes
        if hidden_dim_out is None:
            hidden_dim_out = hidden_dim
        self.conv = get_spectral_conv_with_right_spatial_dim(spatial_dim=num_spatial_dims,
                                                             in_channels=hidden_dim, out_channels=hidden_dim_out,
                                                             modes=modes, feature_transform=feature_transform,
                                                             feature_transform_dim=feature_transform_dim,
                                                             transform_mode=transform_mode)
        if conv_mode == "single":
            self.w = get_conv_with_right_spatial_dim(spatial_dim=num_spatial_dims, in_channels=hidden_dim,
                                                     out_channels=hidden_dim_out, kernel_size=kernel_size, padding='same',
                                                     padding_mode=padding_mode)
        elif conv_mode == "double":
            self.w = get_conv_with_right_spatial_dim(spatial_dim=num_spatial_dims, in_channels=hidden_dim,
                                                     out_channels=hidden_dim_out, kernel_size=1,
                                                     padding='same')
            self.w2 = get_conv_with_right_spatial_dim(spatial_dim=num_spatial_dims, in_channels=hidden_dim,
                                                     out_channels=hidden_dim_out, kernel_size=kernel_size,
                                                     padding='same', padding_mode=padding_mode)
        if activation is None:
            self.act = None
        else:
            if activation_params is None:
                activation_params = {}
            self.act = activation(**activation_params)


    def forward(self, x, p=None):
        spat_dim = x.shape[-self.num_spatial_dims:]
        for i, s in enumerate(spat_dim):
            if i == len(spat_dim) - 1:
                assert self.modes[i] <= s // 2 + 1, 'modes should be at most the spatial dim // 2 + 1 for the last spatial dimension!'
            else:
                assert self.modes[i] <= s, 'modes should be at most the spatial dim all but the last spatial dimensions!'

        # actually do fw pass:
        x1 = self.conv(x, p)
        x2 = self.w(x)

        if self.conv_mode == "single":
            x = x1 + x2
        elif self.conv_mode == "double":
            x3 = self.w2(x)
            x = x1 + x2 + x3
        else:
            raise ValueError(f"Unknown conv_mode {self.conv_mode}")

        if self.act is not None:
            x = self.act(x)
        return x


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes: tuple, feature_transform=False, feature_transform_dim=6,
                 transform_mode=1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        '''
        Transform modes: 
        0 = Affine transformation, modeled as "activation * (1 + FiLM)" (parametrize delta)
        1 = Affine transformation, modeled as "activation * FiLM (parametrize transformation)
        '''

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1,
        # where N is the number of points in our grid

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.complex64))

        # FiLM
        self.feature_transform = feature_transform
        self.feature_transform_dim = feature_transform_dim
        self.transform_mode = transform_mode
        if feature_transform:
            self.weights_feat = nn.Linear(feature_transform_dim, self.out_channels * self.modes1)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication, weights come in batch
    def compl_mul1d_batch(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,biox->box", input, weights)

    def forward(self, x, p=None):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        ft_size = x.size(-1) // 2 + 1
        out_ft = torch.zeros(batchsize, self.out_channels, ft_size, device=x.device, dtype=torch.complex64)

        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        if self.feature_transform:  # apply conditioning layer
            assert p is not None  # we must have supplied params
            FiLM_weights = self.weights_feat(p)
            FiLM_weights = FiLM_weights.view(batchsize, self.out_channels, self.modes1)
            FiLM = torch.ones(batchsize, self.out_channels, ft_size, device=x.device, dtype=torch.float)
            if self.transform_mode == 0:
                FiLM[:, :, :self.modes1] = FiLM[:, :, :self.modes1] + FiLM_weights
            elif self.transform_mode == 1:
                FiLM[:, :, :self.modes1] = FiLM_weights
            out_ft = out_ft * FiLM

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes: tuple, feature_transform=False, feature_transform_dim=6,
                 transform_mode=1):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes[1]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # FiLM
        self.feature_transform = feature_transform
        self.feature_transform_dim = feature_transform_dim
        self.transform_mode = transform_mode
        if feature_transform:
            self.weights_feat = nn.Linear(feature_transform_dim, self.out_channels * 2 * self.modes1 * self.modes2)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, p=None):
        batchsize = x.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        ft_size = (x.size(-2), x.size(-1) // 2 + 1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, *ft_size, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        if self.feature_transform:  # apply conditioning layer
            assert p is not None  # we must have supplied params
            FiLM_weights = self.weights_feat(p)
            FiLM_weights = FiLM_weights.view(batchsize, self.out_channels, 2 * self.modes1, self.modes2)
            FiLM = torch.ones(batchsize, self.out_channels, *ft_size, device=x.device, dtype=torch.float)
            if self.transform_mode == 0:
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM[:, :, :self.modes1, :self.modes2] \
                                                         + FiLM_weights[:, :, :self.modes1, :]
                FiLM[:, :, -self.modes1:, :self.modes2] = FiLM[:, :, -self.modes1:, :self.modes2] \
                                                          + FiLM_weights[:, :, -self.modes1:, :]
            elif self.transform_mode == 1:
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM_weights[:, :, :self.modes1, :]
                FiLM[:, :, -self.modes1:, :self.modes2] = FiLM_weights[:, :, -self.modes1:, :]
            out_ft = out_ft * FiLM

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes: tuple, feature_transform=False,
                 feature_transform_dim=6,
                 transform_mode=1):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes[0]  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes[1]
        self.modes3 = modes[2]

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

        # FiLM
        self.feature_transform = feature_transform
        self.feature_transform_dim = feature_transform_dim
        self.transform_mode = transform_mode
        if feature_transform:
            self.weights_feat = nn.Linear(feature_transform_dim,
                                          self.out_channels * 2 * self.modes1 * 2 * self.modes2 * self.modes3)

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x, p=None):
        batchsize = x.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        ft_size = (x.size(-3), x.size(-2), x.size(-1) // 2 + 1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, *ft_size, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        if self.feature_transform:  # apply conditioning layer
            assert p is not None  # we must have supplied params
            FiLM_weights = self.weights_feat(p)
            FiLM_weights = FiLM_weights.view(batchsize, self.out_channels, 2 * self.modes1, 2 * self.modes2,
                                             self.modes3)
            FiLM = torch.ones(batchsize, self.out_channels, *ft_size, device=x.device, dtype=torch.float)
            if self.transform_mode == 0:
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM[:, :, :self.modes1, :self.modes2, :self.modes3] \
                                                         + FiLM_weights[:, :, :self.modes1, :self.modes2, :]
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM[:, :, -self.modes1:, :self.modes2, :self.modes3] \
                                                         + FiLM_weights[:, :, -self.modes1:, :self.modes2, :]
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM[:, :, :self.modes1, -self.modes2:, :self.modes3] \
                                                         + FiLM_weights[:, :, :self.modes1, -self.modes2:, :]
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM[:, :, -self.modes1:, -self.modes2:, :self.modes3] \
                                                         + FiLM_weights[:, :, -self.modes1:, -self.modes2:, :]
            elif self.transform_mode == 1:
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM_weights[:, :, :self.modes1, :self.modes2, :]
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM_weights[:, :, -self.modes1:, :self.modes2, :]
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM_weights[:, :, :self.modes1, -self.modes2:, :]
                FiLM[:, :, :self.modes1, :self.modes2] = FiLM_weights[:, :, -self.modes1:, -self.modes2:, :]
            out_ft = out_ft * FiLM

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


def get_spectral_conv_with_right_spatial_dim(spatial_dim, **kwargs):
    if spatial_dim == 1:
        conv = SpectralConv1d(**kwargs)
    elif spatial_dim == 2:
        conv = SpectralConv2d(**kwargs)
    elif spatial_dim == 3:
        conv = SpectralConv3d(**kwargs)
    else:
        raise NotImplementedError(f'only 0<x<=3d convs implemented so far, but found spatial dim {spatial_dim}!')

    return conv
