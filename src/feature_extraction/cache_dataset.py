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

        return self.cache.load_batch(
            encoder_name=self.encoder_name,
            split=self.split,
            layer=self.layer,
            batch_idx=idx,
        )