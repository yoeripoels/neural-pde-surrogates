import torch
from torch import nn
from typing import Tuple, Union

from torch_geometric.data import Data
try:
    from torch_cluster import radius_graph, knn_graph
except ImportError:
    print("'torch_cluster' not available, not importing 'radius_graph' and 'knn_graph'")
    def rt_error(text):
        raise RuntimeError(text)
    radius_graph = lambda: rt_error("Cannot use radius_graph, torch_cluster not available")
    knn_graph = lambda: rt_error("Cannot use knn_graph, torch_cluster not available")

from pdes import PDE


class DataCreator(nn.Module):
    def __init__(
        self,
        pde: PDE,
        neighbors: int = 2,
        time_window: int = 5,
        t_resolution: int = 250,
        x_resolution: int = 100,
    ) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            t_resolution (int): temporal resolution
            x_resolution (int): grid resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_res = t_resolution
        self.x_res = x_resolution

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    def create_data(
        self, datapoints: torch.Tensor, steps: list, mode="both"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        assert mode in ["data", "labels", "both"]
        if mode == "data" or mode == "both":
            data = []
        if mode == "labels" or mode == "both":
            labels = []

        for (dp, step) in zip(datapoints, steps):
            assert step-self.tw >= 0 and step + self.tw <= dp.shape[1], 'this step - time window combination is not valid'
            if mode == "data" or mode == "both":
                d = dp[:, step - self.tw: step]
                data.append(d.unsqueeze(dim=0))
            if mode == "labels" or mode == "both":
                l = dp[:, step: self.tw + step]
                labels.append(l.unsqueeze(dim=0))
        if mode == "data":
            return torch.cat(data, dim=0)
        elif mode == "labels":
            return torch.cat(labels, dim=0)
        elif mode == "both":
            return torch.cat(data, dim=0), torch.cat(labels, dim=0)

    def create_graph(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        steps: list,
    ):
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            conditioning (Tensor): Tensor of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # handle these on CPU
        data = data.to('cpu')
        labels = labels.to('cpu')
        x = x.to('cpu')

        nt = self.pde.nt
        x_0 = x[0]  # assume equal nx for all items in batch
        nx = x_0.shape[0] if len(x_0.shape) == 1 else x_0.flatten(0, -2).shape[0]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        u, x_pos, t_pos, y, batch = [], [], [], [], []
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            u.append(data_batch.flatten(2).permute(2, 0, 1))
            y.append(labels_batch.flatten(2).permute(2, 0, 1))
            x_flat = x[b] if len(x[b].shape) == 1 else x[b].flatten(0, -2)  # flatten grid to 1D if needed
            x_pos.append(x_flat)
            t_pos.append(torch.ones(nx) * t[step])
            batch.append(torch.ones(nx) * b)
        u = torch.cat(u)
        x_pos = torch.cat(x_pos)
        t_pos = torch.cat(t_pos)
        y = torch.cat(y)
        batch = torch.cat(batch)

        # Calculate the edge_index
        if f"{self.pde}" == "CE" or f"{self.pde}" == "burgers":
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        elif f"{self.pde}" == "WE":
            edge_index = knn_graph(x_pos, k=self.n, batch=batch.long(), loop=False)
        elif f"{self.pde}" == "NS":
            radius = self.n * (self.pde.dx1**2 + self.pde.dx2**2) ** 0.5
            edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        elif f"{self.pde}" == "DIV1D":
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            edge_index = radius_graph(x_pos, r=radius, batch=batch.long(), loop=False)
        else:
            raise ValueError(f"{self.pde} is not implemented")

        graph = Data(x=u, edge_index=edge_index)
        graph.y = y
        if x_pos.ndim == 1:
            x_pos = x_pos[:, None]
        graph.pos = torch.cat((t_pos[:, None], x_pos), 1)
        graph.batch = batch.long()

        # Equation specific parameters
        if conditioning is not None and torch.numel(conditioning) > 0:
            cond = []
            for i in batch.long():
                cond.append(conditioning[i])
            cond = torch.stack(cond, dim=0)
            graph.cond = cond

        return graph

    def create_next_graph(
        self, graph: Data, pred: torch.Tensor, labels: torch.Tensor, steps: list
    ) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        device = graph.x.device
        pred = pred.to(device)
        labels = labels.to(device)

        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 2)[:, :, self.tw:]
        nt = self.pde.nt
        nx = self.pde.x.shape[0] if len(self.pde.x.shape) == 1 else self.pde.x.flatten(0, -2).shape[0]
        t = torch.linspace(self.pde.tmin, self.pde.tmax, nt)
        # Update labels and input timesteps
        y = []
        t_pos = []
        for (labels_batch, step) in zip(labels, steps):
            y.append(labels_batch.flatten(2).permute(2, 0, 1))
            t_pos.append(torch.ones(nx, device=device) * t[step])
        graph.y = torch.cat(y).to(device)
        graph.pos[:, 0] = torch.cat(t_pos).to(device)

        return graph
