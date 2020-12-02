from scipy.stats import truncnorm
import numpy as np
import torch


def truncnorm_like(x):
    size = [int(s) for s in x.shape]
    eps = truncnorm.rvs(-3.001, 3.001, size=size) / 3.
    return torch.from_numpy(eps)
