import pandas as pd
import numpy as np
import os
import pathlib
import torch
import sys

from amml_utils.registry import register_dataset
from amml_utils.utils import version_check

DATASET_NAME = "Simulated_Bloch"
DATASET_DESCRIPTION = "Simulated Bloch dataset using BART"
DATASET_VERSION = "v1.2"
DATASET_VERSION_FILE = "version.txt"


def standard_transform(trajectory):
    return trajectory


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset that can be used for loading image data from files.

    Implements in particular the `__len__` method and the `__getitem__` method.

    Parameters
    ----------
    data_path
        Path to the location where the dataset is stored.
        Important: This is assumed to be the path without the name of the dataset,
        for instance `/opt/project/data/
    subset
        Either "full", "train", "test" or "val".
    """

    def get_data_dirname(self):
        return DATASET_NAME

    def __init__(self, data_path, subset, transform=standard_transform):
        if subset == "full":
            folder_names = ["train", "test", "val"]
        else:
            folder_names = [subset]
        self.data_paths = []

        for name in folder_names:
            path = pathlib.Path(os.path.join(data_path, self.get_data_dirname(), name))
            for file_path in path.iterdir():
                if not file_path.as_uri().endswith(".parquet"):
                    continue
                self.data_paths.append(file_path)

        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    # Indices of the magnetization
    FEATURES = ['t', 'B_x', 'B_y', 'B_z', 'R1', 'R2', 'M0']
    LABELS = ['M_x', 'M_y', 'M_z']

    def __getitem__(self, idx):
        # Load the parquet file and convert data to numpy array
        data_df = pd.read_parquet(self.data_paths[idx]).astype(np.float64)

        # Apply transformation if specified
        if self.transform:
            data_df = self.transform(data_df)

        # Separate features and labels, and convert numpy arrays to PyTorch tensors
        return torch.from_numpy(data_df[self.FEATURES].values), torch.from_numpy(data_df[self.LABELS].values)


register_dataset(DATASET_NAME, CustomDataset, description=DATASET_DESCRIPTION,
                 version_check_function=lambda path: version_check(path, DATASET_NAME, DATASET_VERSION,
                                                                   DATASET_VERSION_FILE))
