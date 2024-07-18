import torch
from pdes import PDE
from typing import List, Union
from common.interfaces import M
from pdes import PDE


def process_step(pde: PDE, model_interface: M, sim_prev: torch.Tensor, gtbc_prev: torch.Tensor, gtbc_next: torch.Tensor, device: str, get_bc=True, set_bc=True, set_min=True, process_settings: dict = None) -> (torch.Tensor, Union[torch.Tensor, None]):
    """
    Function to process the intermediate simulation output at each step (extract BCs from gt, set BCs, set min values)
    Args:
        pde_name: PDE we are working with (to identify what conditions to apply & extract info)
        model_interface: Model interface currently used
        sim_prev: Previous output step
        gtbc_prev: Previous ground-truth BC info
        gtbc_next: Next ground-truth BC info
        device: Device to place data on
        get_bc: Whether to extract BC info
        set_bc: Whether to set BCs in sim
        set_min: Whether to set min values in sim
        process_settings: Overrides for settings, e.g., set_bc or set_min can be forced off
    Returns: Tuple of:
        -simulation with adjusted values
        -bc information
    """
    if process_settings is not None:
        if "set_bc" in process_settings:
            set_bc = set_bc and process_settings["set_bc"]
        if "set_min" in process_settings:
            set_min = set_min and process_settings["set_min"]

    if f"{pde}" == "DIV1D" and (model_interface == M.AR_TB or model_interface == M.AR_TB_GNN):
        if set_bc and model_interface == M.AR_TB:
            sim_prev = set_bc_1d(0, 0, sim_prev, gtbc_prev, device)
        if set_min:
            mu_ne, sd_ne = pde.var_mean_sd[0]
            ne_0 = (0.1 - mu_ne) / sd_ne
            mu_te, sd_te = pde.var_mean_sd[2]
            te_0 = (0.1 - mu_te) / sd_te
            mu_nn, sd_nn = pde.var_mean_sd[3]
            nn_0 = (0.1 - mu_nn) / sd_nn
            min_dims = [0, 2, 3]
            min_values = [ne_0, te_0, nn_0]
            sim_prev = set_min_values(min_dims, min_values, sim_prev)
        if get_bc:
            if model_interface == M.AR_TB:
                bc = extract_bc_1d(0, 0, sim_prev=sim_prev, gtbc_prev=gtbc_prev, gtbc_next=gtbc_next, device=device, mode=pde.bc_mode)
            elif model_interface == model_interface == M.AR_TB_GNN:
                bc = extract_bc_1d_simple(0, 0, gtbc_prev=gtbc_prev, gtbc_next=gtbc_next, device=device, mode=pde.bc_mode)
        else:
            bc = None
        return sim_prev, bc
    else:
        return sim_prev, None


def set_bc_1d(bc_dim: int, bc_x: int, sim: torch.Tensor, gtbc: torch.Tensor, device: str) -> torch.Tensor:
    """
    For a simulation, fix the boundary values to the boundary conditions of the groundtruth simulation
    Args:
        bc_dim: which dimension to fix BC for
        bc_x: value of where to fix BCs (should be 0 or -1, i.e., start or end of domain)
        sim: Model output to correct
        gtbc: Groundtruth data to extract BC from
        device: Device to place output tensor on
    Returns:
        Simulation with corrected boundary values
    """
    bs, num_c, tw, nx = sim.shape
    assert bc_dim in range(0, num_c)
    assert bc_x in [0, -1]

    sim[:, bc_dim, :, bc_x] = gtbc[:, bc_dim, :, bc_x]
    return sim.to(device)


def set_min_values(dims: List[int], min_values: List[float], sim: torch.Tensor) -> torch.Tensor:
    """
    For a simulation, set the minimum value in the specified dimensions
    Args:
        dims: Which dimensions to fix
        min_values: Which values to set as minimum
        sim: Simulation to adjust
    Returns:
    """
    num_c = sim.shape[1]
    assert all([d in range(0, num_c) for d in dims])
    for i, d in enumerate(dims):
        sim[:, d, ...][sim[:, d, ...] < min_values[i]] = min_values[i]
    return sim


def extract_bc_1d_simple(bc_dim: int, bc_x: int, gtbc_prev: torch.Tensor, gtbc_next: torch.Tensor, device: str, mode='delta') -> torch.Tensor:
    """
    From a given simulation, extract the BCs
    Args:
        bc_dim: which dimension to extract BC info for
        bc_x: value of where to take BCs (should be 0 or -1, i.e., start or end of domain)
        gtbc_prev: Ground truth at previous timewindow
        gtbc_next: Ground truth at next timewindow
        device: Device to place output tensor on
        mode: How to process BCs
    Returns:
        Boundary condition information in shape [batch, num_c, time_window], where num_c depends on the selected mode
    """
    bs, num_c, tw, nx = gtbc_prev.shape
    assert bc_dim in range(0, num_c)
    assert bc_x in [0, -1]

    gtbc_prev, gtbc_next = [x.to(device) for x in [gtbc_prev, gtbc_next]]

    # get boundary information (all of shape (bs, tw))
    bc_prev = gtbc_prev[:, bc_dim, :, bc_x]
    bc_next = gtbc_next[:, bc_dim, :, bc_x]

    boundary_last = bc_prev[:, -1]
    boundary_last = boundary_last[:, None].repeat(1, tw)

    dif_new = bc_next - boundary_last
    dif_prev = bc_prev - boundary_last
    if mode == "delta":
        bc = torch.stack([dif_new, dif_prev], dim=1)
    elif mode == "all_fixed_bc":
        raise NotImplementedError("Not supported")
    elif mode == "all":
        raise NotImplementedError("Not supported")
    elif mode == "simple":
        bc = torch.stack([bc_prev, bc_next], dim=1)
    else:
        raise ValueError("Incorrect BC mode")
    return bc.to(device)


def extract_bc_1d(bc_dim: int, bc_x: int, sim_prev: torch.Tensor, gtbc_prev: torch.Tensor, gtbc_next: torch.Tensor, device: str, mode='delta') -> torch.Tensor:
    """
    From a given simulation, extract the BCs
    Args:
        bc_dim: which dimension to extract BC info for
        bc_x: value of where to take BCs (should be 0 or -1, i.e., start or end of domain)
        sim_prev: Model output at previous timewindow
        gtbc_prev: Ground truth at previous timewindow
        gtbc_next: Ground truth at next timewindow
        device: Device to place output tensor on
        mode: How to process BCs
    Returns:
        Boundary condition information in shape [batch, num_c, time_window], where num_c depends on the selected mode
    """
    bs, num_c, tw, nx = sim_prev.shape
    assert bc_dim in range(0, num_c)
    assert bc_x in [0, -1]

    sim_prev, gtbc_prev, gtbc_next = [x.to(device) for x in [sim_prev, gtbc_prev, gtbc_next]]

    # get boundary information (all of shape (bs, tw))
    boundary_prev = sim_prev[:, bc_dim, :, bc_x]
    bc_prev = gtbc_prev[:, bc_dim, :, bc_x]
    bc_next = gtbc_next[:, bc_dim, :, bc_x]

    boundary_last = boundary_prev[:, -1]
    boundary_last = boundary_last[:, None].repeat(1, tw)

    dif_new = bc_next - boundary_last
    dif_prev = bc_prev - boundary_last
    if mode == "delta":
        bc = torch.stack([dif_new, dif_prev], dim=1)
    elif mode == "all_fixed_bc":
        bc = torch.stack([boundary_prev, bc_next, dif_new], dim=1)
    elif mode == "all":
        bc = torch.stack([boundary_prev, bc_prev, bc_next, dif_new, dif_prev], dim=1)
    elif mode == "simple":
        bc = torch.stack([bc_prev, bc_next], dim=1)
    else:
        raise ValueError("Incorrect BC mode")
    return bc.to(device)

