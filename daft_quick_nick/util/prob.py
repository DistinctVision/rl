import math

import random
import numpy as np
import torch


def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def distributions_kl_divergence(mean_a: torch.Tensor, std_a: torch.Tensor,
                                mean_b: torch.Tensor, std_b: torch.Tensor) -> torch.Tensor:
    return torch.log(std_b / std_a) + \
        (torch.square(std_a) + torch.square(mean_a - mean_b)) / (2 * torch.square(std_b)) - 0.5


def distribution_entropy(std: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log((2 * math.pi * math.e) * torch.square(std))
