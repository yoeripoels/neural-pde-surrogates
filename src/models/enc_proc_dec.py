from typing import Union
import torch
from torch import nn
from pdes import PDE
from models.base import ModelInterface
from argparse import Namespace
import torch_geometric.data
import models
from utils import broadcast_to_grid
import warnings
from utils.attr import getattr_nested


def create_model(model: Union[nn.Module, dict, Namespace, str], pde: PDE, base_args: dict, extra_kwargs: dict = None):
    if isinstance(model, nn.Module):
        return model
    elif isinstance(model, dict) or isinstance(model, Namespace) or isinstance(model, str):
        if isinstance(model, str):
            model_class = model
            model_kwargs = dict(base_args)
        else:
            if isinstance(model, Namespace):
                model = vars(model)
            model_class = model.pop("object")
            # arguments specified in model dict take priority over arguments specified in base_args
            model_kwargs = dict(list(base_args.items()) + list(model.items()))
        if extra_kwargs is not None:
            model_kwargs = dict(list(model_kwargs.items()) + list(extra_kwargs.items()))
        # specify modules to check
        modules_to_check = [models.enc_proc_dec_components, models, models.common]
        for module in modules_to_check:
            if (model_init := getattr_nested(module, model_class)) is not False:
                return model_init(**model_kwargs, pde=pde)
        else:  # if we did not find it in any module, e.g., loop finished
            raise ValueError(f"Cannot find object {model_class} in any of "
                    f"{[m.__name__ if hasattr(m, '__name__') else m for m in modules_to_check]}]")
    else:
        raise ValueError("Model was not the correct type: Should be nn.Module / dict / argparse.Namespace")


class EncProcDec(ModelInterface):
    """
    Encode-Process-Decode with standardize input/output for all components.
    Easy to interchange components like this.
    """
    def __init__(self,
                 pde: PDE,
                 encoder: Union[nn.Module, dict, Namespace, str],
                 processor: Union[nn.Module, dict, Namespace, str],
                 decoder: Union[nn.Module, dict, Namespace, str],
                 bc_encoder: Union[nn.Module, dict, Namespace, str] = None,  # optional
                 num_c: int = 1,
                 num_spatial_dims: int = 1,
                 time_window: int = 25,
                 data_structure: str = "grid",
                 processor_residual: bool = False,  # whether we add a residual connection after the processor
                 **base_args,
                 ):

        super().__init__()

        # load some settings
        self.pde = pde
        self.num_c = num_c
        self.num_spatial_dims = num_spatial_dims
        self.time_window = time_window
        self.processor_residual = processor_residual

        # parse data structure
        data_structures = ["grid", "graph"]
        assert data_structure in data_structure, f"Property {data_structure} must be one of {data_structures}"
        self.data_structure = data_structure

        # also save them in base args, so we can easily pass them to children
        base_args["num_c"] = num_c
        base_args["num_spatial_dims"] = num_spatial_dims
        base_args["time_window"] = time_window

        # initialize models
        # first initialize BC encoder, as we might need its output dim
        if bc_encoder is not None:
            self.bc_encoder = create_model(bc_encoder, self.pde, base_args,
                                           extra_kwargs=dict(bc_encoder_in=self.pde.n_cond_dynamic))
            self.n_cond = self.pde.n_cond_static + self.pde.n_cond_spatial + self.bc_encoder.n_out
        else:
            self.bc_encoder = None
            self.n_cond = self.pde.n_cond_static + self.pde.n_cond_spatial

        base_args["n_cond"] = self.n_cond

        # handle encoder
        self.encoder = create_model(encoder, self.pde, base_args)

        # processor can be multiple --> chain them
        if isinstance(processor, list) or isinstance(processor, tuple):
            self.processor = nn.ModuleList([create_model(p, self.pde, base_args) for p in processor])
        else:
            self.processor = nn.ModuleList([create_model(processor, self.pde, base_args)])  # list of 1 element

        # handle decoder
        self.decoder = create_model(decoder, self.pde, base_args)

    def __repr__(self):
        return f'{self.encoder}-{self.processor}-{self.decoder}'

    # pass on the model and data interface from the processors
    @property
    def model_interface(self):
        mi = [p.model_interface for p in self.processor]
        assert mi.count(mi[0]) == len(mi), "Not all processors have the same model interface!"
        return mi[0]

    @property
    def data_interface(self):
        return set.intersection(*[set(p.data_interface) for p in self.processor])

    def forward(self, x: Union[torch.Tensor, torch_geometric.data.Data], cond: torch.Tensor = None, bc: torch.Tensor = None, pos: torch.Tensor = None,
                t_cond: torch.Tensor = None, spatial_cond: torch.Tensor = None):
        check_none = lambda x: None if (x is None or torch.numel(x) == 0) else x
        cond = check_none(cond)
        bc = check_none(bc)
        pos = check_none(pos)
        t_cond = check_none(t_cond)
        spatial_cond = check_none(spatial_cond)

        # embed conditioning signal, parse variables, pre-broadcast to domain
        if self.data_structure == "grid":
            u = x  # get actual input, shape [b, c, tw, *spatial_dims]
            variables = self.embed_conditioning_signal(cond, bc, t_cond)

            # broadcast to spatial dims
            if variables is not None:
                # broadcast variables beforehand; [b, c, tw, *spatial_dims] --> get spatial_dims
                variables_broadcast = broadcast_to_grid(variables, list(u.shape[3:]))
                variables_broadcast = torch.cat([variables_broadcast, spatial_cond], dim=1) if spatial_cond is not None else variables_broadcast
            else:
                variables_broadcast = spatial_cond
            kwargs = {}
        elif self.data_structure == "graph":
            warnings.warn('gnn data structure has been deprecated in favor of GNN wrapper!')
            u = x.x  # get actual input, shape [nx * b, c, tw]
            edge_index = x.edge_index
            batch = x.batch

            # set up variables
            variables = self.embed_conditioning_signal(cond, bc, t_cond)

            # broadcast variables to nodes
            variables_broadcast = variables[batch.long()] if variables is not None else None  # new but faster
            warnings.warn('Careful: we did not implement spatial conditioning support for data_structure == graph!')


            # gather & normalize spatial domain, save in pos
            pos = x.pos  # gather position info of each node
            n_dim = pos.shape[1] - 1  # -1 for time
            assert self.num_spatial_dims == n_dim, f"Supplied input ({n_dim}D) does not match " \
                                                   f"self.num_spatial_dims ({self.num_spatial_dims}D)"
            if n_dim == 1:
                pos_graph = pos[:, 1][:, None] / self.pde.L
            else:
                pos_graph = []
                for i in range(n_dim):
                    pos_graph.append(pos[:, i+1][:, None] / self.pde.L[i])
                pos_graph = torch.cat(pos_graph, dim=-1)
            pos = pos_graph

            # set up forward kwargs to pass along
            kwargs = dict(edge_index=edge_index, batch=batch)
        else:
            raise ValueError(f"No forward(..) implementation for data_structure '{self.data_structure}'")

        # start the actual forward call
        h = self.encoder(u=u, variables_broadcast=variables_broadcast, pos=pos, **kwargs)

        for i, p in enumerate(self.processor):
            h_next = p(h=h, variables_broadcast=variables_broadcast, pos=pos, **kwargs)
            if self.processor_residual and i > 0:  # only apply residual from proc->proc, on i=0 h is the enc output
                h = h_next + h
            else:
                h = h_next

        h = self.decoder(h=h, u=u, variables=variables, variables_broadcast=variables_broadcast, pos=pos, **kwargs)
        return h
