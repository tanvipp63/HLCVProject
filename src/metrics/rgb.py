import math

import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_metric


def mse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Mean Squared Error (lower is better).
    """

    return F.mse_loss(pred, target).item()


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio in dB (higher is better).
    """

    mse_value = mse(pred, target)

    if mse_value == 0:
        return float("inf")

    return 20 * math.log10(max_value) - 10 * math.log10(mse_value)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Structural Similarity Index (higher is better).

    Expects tensors of shape:
        (3, H, W) or (1, 3, H, W)
    """

    if pred.ndim == 4:
        pred = pred.squeeze(0)

    if target.ndim == 4:
        target = target.squeeze(0)

    pred = pred.detach().cpu().permute(1, 2, 0).numpy()
    target = target.detach().cpu().permute(1, 2, 0).numpy()

    return ssim_metric(
        pred,
        target,
        channel_axis=-1,
        data_range=1.0,
    )