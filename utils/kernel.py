from gpytorch.kernels.rbf_kernel import RBFKernel
from torch import Tensor
import torch
# from torchmetrics.functional.pairwise import pairwise_manhattan_distance 


def dist(x1, x2, x1_eq_x2=False):
    """L1 norm"""
    if x1.ndim <= 2:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(0)
    elif x1.ndim == 3:
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(1)
    else:
        print("ndim error")
    distance = torch.abs(x1 - x2).sum(dim=-1)
    return distance.clamp_min_(0)


def sq_dist(x1, x2, x1_eq_x2=False):
    """squared L1 norm"""
    distance = dist(x1, x2, x1_eq_x2)
    squared_distance = distance.pow(2)
    return squared_distance.clamp_min_(0)


class LaplacianKernel(RBFKernel):
    def __init__(self, lengthscale=1.0, ard_num_dims=None):
        super().__init__(ard_num_dims=ard_num_dims)
        self.lengthscale = lengthscale

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                res = torch.linalg.norm(x1 - x2, dim=-1)  # 2-norm by default
                return res.pow(2) if square_dist else res
        else:
            dist_func = sq_dist if square_dist else dist
            return dist_func(x1, x2, x1_eq_x2)

