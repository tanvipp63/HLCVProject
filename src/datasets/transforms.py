from torchvision import transforms


# Standard ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    """
    Construct a torchvision preprocessing pipeline.

    Args:
        image_size: Final square image size.
        mean: Normalization mean.
        std: Normalization standard deviation.

    Returns:
        torchvision.transforms.Compose
    """

    resize_size = int(image_size * 256 / 224)

    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_dino_transform(image_size: int = 224):
    """
    Image preprocessing for DINOv2.
    """

    # Placeholder.
    # Replace with DINO-specific normalization.
    return _build_transform(
        image_size=image_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def get_clip_transform(image_size: int = 224):
    """
    Image preprocessing for CLIP.
    """

    # Placeholder.
    # Replace with CLIP-specific normalization.
    return _build_transform(
        image_size=image_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def get_siglip_transform(image_size: int = 224):
    """
    Image preprocessing for SigLIP.
    """

    # Placeholder.
    # Replace with SigLIP-specific normalization.
    return _build_transform(
        image_size=image_size,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )