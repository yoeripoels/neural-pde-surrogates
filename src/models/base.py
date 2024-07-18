import torch
from abc import abstractmethod, ABCMeta
from common.interfaces import M, D
from typing import List


class ModelInterface(torch.nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def model_interface(self) -> M:
        """The enum defining the model's interface
        """
        raise NotImplementedError("model_interface not set!")

    @property
    @abstractmethod
    def data_interface(self) -> List[D]:
        """The list of the data interfaces the model supports
        """
        return []

    # below, we add some utils that we found convenient to re-use across models for the datasets we worked with,
    # but these can freely be ignored
    def embed_conditioning_signal(self, cond: torch.Tensor = None, boundary_conditions: torch.Tensor = None,
                                  t_cond: torch.Tensor = None, spatial_cond: torch.Tensor = None, unsqueeze_dims: int = 0):
        """
        converts PDE equation coefficients, boundary_conditions, and misc conditioning (possibly varying over time) to
        a learned embedding that represents the conditioning signal
        Args:
            cond: dict with key, values of the PDE parameters
            boundary_conditions: tensor with the boundary condition values, or None
            t_cond: misc conditioning tensor, or None
        """
        # set to None if no elements
        if cond is not None and torch.numel(cond) == 0:
            cond = None
        if boundary_conditions is not None and torch.numel(boundary_conditions) == 0:
            boundary_conditions = None
        if t_cond is not None and torch.numel(t_cond) == 0:
            t_cond = None

        variables = []
        if cond is not None:
            for i in range(cond.shape[1]):  # channel dimension
                variables.append(cond[:, i])

        # if bc is scalar (for example in the case of a 1D grid), we expand it to be of grid shape
        bc = boundary_conditions

        # check which time-varying info we should encode
        if bc is not None and t_cond is not None:
            bc_in = torch.cat([bc, t_cond], dim=1)  # concat along channel dim
        elif bc is not None and t_cond is None:
            bc_in = bc  # only use BC data
        elif bc is None and t_cond is not None:
            bc_in = t_cond  # only use conditioning data
        else:
            bc_in = None  # neither

        if bc_in is not None and self.bc_encoder is not None:
            bc_variables = self.bc_encoder(bc_in)
            for i in range(bc_variables.shape[1]):
                variables.append(bc_variables[:, i])

        # stack variables or set to None
        if len(variables) == 0:
            variables = None
        else:
            variables = torch.stack(variables, dim=1)  # stack to [batchsize, num_var]
            for _ in range(unsqueeze_dims):
                variables = variables.unsqueeze(-1)

        return variables

