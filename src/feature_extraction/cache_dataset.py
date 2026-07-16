import torch
from torch.utils.data import Dataset

from .cache import FeatureCache


class CachedFeatureDataset(Dataset):
    """
    Dataset backed by cached feature batches.
    """

    def __init__(
        self,
        cache: FeatureCache,
        encoder_name: str,
        split: str,
        layer: int = -1,
    ):

        self.cache = cache
        self.encoder_name = encoder_name
        self.split = split
        self.layer = layer

        self.batch_paths = cache.list_batches(
            encoder_name,
            split,
            layer,
        )

    def __len__(self):

        return len(self.batch_paths)

    def __getitem__(self, idx):

        if idx >= len(self.batch_paths):
            raise IndexError

        return torch.load(self.batch_paths[idx])