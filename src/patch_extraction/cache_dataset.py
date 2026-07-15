from torch.utils.data import Dataset

from .targets import PatchTargets


class CachedTargetDataset(Dataset):
    """
    Dataset backed by cached target batches.
    """

    def __init__(
        self,
        target_generator: PatchTargets,
        split: str,
        target_type: str,
        encoder_name: str,
    ):

        self.target_generator = target_generator
        self.split = split
        self.target_type = target_type
        self.encoder_name = encoder_name

        self.batch_paths = target_generator.list_batches(
            split,
            target_type,
            encoder_name,
        )

    def __len__(self):

        return len(self.batch_paths)

    def __getitem__(self, idx):

        return self.target_generator.load_batch(
            split=self.split,
            target_type=self.target_type,
            encoder_name=self.encoder_name,
            batch_idx=idx,
        )