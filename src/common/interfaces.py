from enum import Enum


class D(Enum):  # data
    sim1d = 0  # 1d simulation, 1 element is of shape (c, t, x)
    sim2d = 1  # 2d simulation, 1 element is of shape (c, t, x, y)
    sim1d_var_t = 2  # 1d simulations -> (c, t, x), but t can vary


class M(Enum):
    AR_TB_GNN = 0  # autoregressive + temporal bundling + GNN
    AR_TB = 1  # autoregressive + temporal bundling
