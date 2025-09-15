from functools import partial
import imageio.v3 as imageio
import os
import pathlib
import torch

from amml_utils.registry import register_dataset
from amml_utils.utils import download_from_nextcloud


DATASET_NAME = "EXAMPLE"


def _read_image(impath):
    return torch.from_numpy(imageio.imread(impath))


def standard_transform(image):
    image = (image / 255.0).mean(-1, keepdims=True).permute(2, 0, 1)
    if image.shape[1] == 481:
        image = torch.rot90(image, 1, (-2, -1))
    return image


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset that can be used for loading image data from files.

    Implements in particular the `__len__` method and the `__getitem__` method.

    Parameters
    ----------
    base_path
        Base path to the location where the dataset is stored.
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
            path = pathlib.Path(os.path.join(data_path, DATASET_NAME, "small_BSDS500/data/images", name))
            for impath in path.iterdir():
                if not impath.as_uri().endswith("jpg"):
                    continue
                self.image_paths.append(impath)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = _read_image(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image


register_dataset(DATASET_NAME, partial(download_from_nextcloud, dataset_name=DATASET_NAME), CustomDataset)
