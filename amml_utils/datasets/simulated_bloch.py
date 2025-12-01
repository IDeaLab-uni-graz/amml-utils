import pandas as pd
import numpy as np
import os
import pathlib
import torch

from amml_utils.registry import register_dataset

# Warning: As the sequence lengths vary, currently we do not support batch size > 1
# We will have to take care of it with padding (via a collate function for the DataLoader or similar)
# See:
# - https://www.codefull.net/2018/11/use-pytorchs-dataloader-with-variable-length-sequences-for-lstm-gru/
# - https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/13

DATASET_NAME = "Simulated_Bloch"

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
                print(file_path.as_uri())
                if not file_path.as_uri().endswith(".parquet"):
                    continue
                self.data_paths.append(file_path)

        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    # Indices of the magnetization
    LABEL_INDICES = range(4, 7)

    def __getitem__(self, idx):
        # Load the parquet file and convert data to numpy array
        data_df = pd.read_parquet(self.data_paths[idx]).astype(np.float32)
        datapoint = data_df.to_numpy()

        # Apply transformation if specified
        if self.transform:
            datapoint = self.transform(datapoint)

            # Separate features and labels
        features = np.delete(datapoint, self.LABEL_INDICES, axis=1)
        labels = datapoint[:, self.LABEL_INDICES]

        # Convert numpy arrays to PyTorch tensors
        return torch.from_numpy(features), torch.from_numpy(labels)


register_dataset(DATASET_NAME, None, CustomDataset)
