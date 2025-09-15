from abc import ABC, abstractmethod
import os
from pathlib import Path


class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders."""
    def __init__(self, data_path: Path, transform=None):
        self.data_path = data_path
        self.transform = transform

    def dataset_available(self):
        if self.dataset_name in os.listdir(self.data_path):
            return True
        return False

    def download_dataset(self):
        if not self.dataset_available():
            self._download_dataset()
        else:
            print(f"Dataset '{self.dataset_name}' already downloaded...")

    def get_dataset_as_tensor(self, data_type="full"):
        # Check if dataset is already downloaded and if not, download it
        self.download_dataset()

        # Return the actual dataset as a tensor
        return self._get_dataset_as_tensor(data_type=data_type)

    def get_dataset(self, data_type="full"):
        # Check if dataset is already downloaded and if not, download it
        self.download_dataset()

        # Return the actual dataset
        return self._get_dataset(data_type=data_type)

    @abstractmethod
    def _get_dataset_as_tensor(self, data_type="full"):
        pass

    @abstractmethod
    def _get_dataset(self, data_type="full"):
        pass

    @abstractmethod
    def _download_dataset(self):
        pass
