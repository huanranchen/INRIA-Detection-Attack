import torch
import numpy as np


@torch.no_grad()
def cutmix(x: torch.tensor, y: torch.tensor, cutmix_prob: int = 0.1, beta: int = 0.3,
           num_classes: int = 60) -> torch.tensor:
    if np.random.rand() > cutmix_prob:
        return x, y
    N, _, H, W = x.shape
    indices = torch.randperm(N, device=torch.device('cuda'))
    label = torch.zeros((N, num_classes), device=torch.device('cuda'))
    x1 = x[indices, :, :, :]
    y1 = x.clone()[indices]
    lam = np.random.beta(beta, beta)

    rate = np.sqrt(1 - lam)
    cut_x, cut_y = (H * rate) // 2, (W * rate) // 2
    cx, cy = np.random.randint(cut_x, H - cut_x), np.random.randint(cut_y, W - cut_x)
    bx1, bx2 = cx - cut_x, cx + cut_x
    by1, by2 = cy - cut_y, cy + cut_y

    x[:, :, bx1:bx2, by1:by2] = x1[:, :, bx1:bx2, by1:by2].clone()
    label[torch.arange(N), y] = lam
    label[torch.arange(N), y1] = 1 - lam
    return x, label
