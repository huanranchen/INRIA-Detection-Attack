import torch
import numpy as np


@torch.no_grad()
def mixup(x: torch.tensor,  cutmix_prob: int = 0.5, beta: int = 10) -> torch.tensor:
    if np.random.rand() > cutmix_prob:
        return x
    N, _, H, W = x.shape
    indices = torch.randperm(N, device=torch.device('cuda'))
    x1 = x[indices, :, :, :].clone()
    lam = np.random.beta(beta, beta)

    x = lam * x + (1 - lam) * x1
    return x
