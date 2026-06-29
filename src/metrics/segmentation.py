import torch


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """
    Computes pixel accuracy for semantic segmentation.

    Higher is better.
    """

    pred = pred.view(-1)
    target = target.view(-1)

    correct = (pred == target).sum().float()
    total = target.numel()

    return (correct / total).item()


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-8,
) -> float:
    """
    Computes mean Intersection over Union (mIoU)
    across all classes.

    Higher is better.
    """

    pred = pred.view(-1)
    target = target.view(-1)

    ious = []

    for cls in range(num_classes):

        pred_mask = pred == cls
        target_mask = target == cls

        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union == 0:
            continue

        ious.append((intersection / (union + eps)).item())

    if len(ious) == 0:
        return 0.0

    return sum(ious) / len(ious)