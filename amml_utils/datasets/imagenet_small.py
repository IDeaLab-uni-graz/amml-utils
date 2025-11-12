import imageio.v3 as imageio
import os
import pathlib
import torch
from torchvision.transforms import RandomCrop

from amml_utils.registry import register_dataset


DATASET_NAME = "IMAGENET_SMALL"

STANDARD_WIDTH = 256
STANDARD_HEIGHT = 256


def standard_transform(image):
    image = image.mean(-1, dtype=float)
    return RandomCrop((STANDARD_HEIGHT, STANDARD_WIDTH)).forward(image)


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
        self.image_paths = []

        for name in folder_names:
            path = pathlib.Path(os.path.join(data_path, DATASET_NAME, name))
            for impath in path.iterdir():
                self.image_paths.append(impath)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torch.from_numpy(imageio.imread(self.image_paths[idx]))
        if self.transform:
            image = self.transform(image)
        return image


register_dataset(DATASET_NAME, None, CustomDataset)
