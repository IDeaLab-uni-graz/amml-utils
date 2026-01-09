import pandas as pd
import numpy as np
import os
import pathlib
import torch
import sys

from amml_utils.registry import register_dataset

DATASET_NAME = "Simulated_Bloch"
DATASET_DESCRIPTION = "Simulated Bloch dataset using BART"
DATASET_VERSION = "v1.0"
DATASET_VERSION_FILE = "version.txt"


def version_check(data_path, dataset_name, dataset_version, dataset_version_file):
    path = os.path.join(data_path, dataset_name, dataset_version_file)
    if os.path.isfile(path):
        with open(path) as input_file:
            version, last_modified = [next(input_file).replace('\n', '') for _ in range(2)]
            return version == dataset_version, f"Expected: {dataset_version} - Got: {version} from {last_modified}"
    else:
        return False, f"Version file '{dataset_version_file}' not found! Expected: {dataset_version}"


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

    def __init__(self, data_path, subset, transform=standard_transform):
        if subset == "full":
            folder_names = ["train", "test", "val"]
        else:
            folder_names = [subset]
        self.data_paths = []

        for name in folder_names:
            path = pathlib.Path(os.path.join(data_path, DATASET_NAME, name))
            for file_path in path.iterdir():
                if not file_path.as_uri().endswith(".parquet"):
                    continue
                self.data_paths.append(file_path)

        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    # Indices of the magnetization
    FEATURES_LENGTH = 7
    LABELS_LENGTH = 3

    def __getitem__(self, idx):
        # Load the parquet file and convert data to numpy array
        data_df = pd.read_parquet(self.data_paths[idx]).astype(np.float32)
        datapoint = data_df.to_numpy()

        # Apply transformation if specified
        if self.transform:
            datapoint = self.transform(datapoint)

            # Separate features and labels
        features = datapoint[:, :self.FEATURES_LENGTH]
        labels = datapoint[:, self.FEATURES_LENGTH:]

        # Convert numpy arrays to PyTorch tensors
        return torch.from_numpy(features), torch.from_numpy(labels)


register_dataset(DATASET_NAME, CustomDataset, description=DATASET_DESCRIPTION,
                 version_check_function=lambda path: version_check(path, DATASET_NAME, DATASET_VERSION,
                                                                   DATASET_VERSION_FILE))
