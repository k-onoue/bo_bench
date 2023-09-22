import torch


def sphere_function(x):
    return torch.sum(x ** 2, 1, keepdim=True)

