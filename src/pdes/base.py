import torch


class PDE:
    """Generic PDE template"""
    def __init__(self, tmin, tmax, nt, name, n_cond_static=0, n_cond_dynamic=0, n_cond_spatial=0, **kwargs):
        self.tmin = tmin
        self.tmax = tmax
        self.nt = nt
        self.name = name
        self.n_cond_static = n_cond_static
        self.n_cond_dynamic = n_cond_dynamic
        self.n_cond_spatial = n_cond_spatial
        for name, val in kwargs.items():
            setattr(self, name, val)

    def __repr__(self):
        return self.name


class PDE1D(PDE):
    def __init__(self, tmin, tmax, nt, L, nx, x, name, n_cond_static=0, n_cond_dynamic=0, **kwargs):
        super().__init__(tmin, tmax, nt, name, n_cond_static, n_cond_dynamic, **kwargs)
        self.dt = (self.tmax - self.tmin) / (self.nt-1)
        self.L = L
        self.nx = nx
        self.dx = L / (self.nx - 1)
        self.dxs = [self.dx]
        if x is None:
            x = torch.linspace(0, self.L, self.nx)
        self.x = x


class PDE2D(PDE):
    def __init__(self, tmin, tmax, nt, L1, L2, nx1, nx2, x, name, n_cond_static=0, n_cond_dynamic=0, n_cond_spatial=0, **kwargs):
        super().__init__(tmin, tmax, nt, name, n_cond_static, n_cond_dynamic, n_cond_spatial, **kwargs)
        self.L1 = L1
        self.L2 = L2
        self.L = [L1, L2]
        self.nx1 = nx1
        self.nx2 = nx2

        self.dt = self.tmax / (nt - 1)
        self.dx1 = self.L1 / (nx1 - 1)
        self.dx2 = self.L2 / (nx2 - 1)
        self.dxs = [self.dx1, self.dx2]

        if x is None:
            x_all = [torch.linspace(0, L1, nx1), torch.linspace(0, L2, nx2)]
            x = torch.stack(torch.meshgrid(*x_all, indexing="ij"))
            x = torch.movedim(x, 0, -1)
        self.x = x
