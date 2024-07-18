import torch


def broadcast_to_grid(x: torch.Tensor, spatial_dims: list):
    """
    Args:
        x: tensor of shape [b, c]
        spatial_dims: *spatial_dims
    Returns: x broadcasted to spatial dims, of shape [b, c, *spatial_dims]
    """
    for _ in range(len(spatial_dims)):
        x = x.unsqueeze(-1)
    return x.repeat([1, 1] + spatial_dims)

