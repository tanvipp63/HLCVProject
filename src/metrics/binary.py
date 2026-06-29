import torch


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """
    Computes the F1 (Dice) score for binary predictions.

    Higher is better.
    """

    pred = (pred > threshold).float()
    target = (target > threshold).float()

    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    return f1.item()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> float:
    """
    Computes the Intersection over Union (Jaccard Index).

    Higher is better.
    """

    pred = (pred > threshold).float()
    target = (target > threshold).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = intersection / (union + eps)

    return iou.item()