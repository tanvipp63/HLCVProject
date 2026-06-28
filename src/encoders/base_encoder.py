from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    """
    Abstract base class for Encoder Models.
    """

    @abstractmethod
    def load_model(self):
        """Load the pretrained encoder."""
        pass

    @abstractmethod
    def get_transform(self):
        """Return the image preprocessing pipeline."""
        pass

    @abstractmethod
    def extract_features(self, images):
        """Extract dense patch features from a batch of images."""
        pass