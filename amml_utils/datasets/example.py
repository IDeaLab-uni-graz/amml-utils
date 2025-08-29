import imageio.v3 as imageio
import os
import pathlib
import torch

from amml_utils.base_data_loader import BaseDataLoader
from amml_utils.registry import register_data_loader
from amml_utils.utils import download_from_nextcloud


DATASET_NAME = "EXAMPLE"


def _read_image(impath):
    image = torch.from_numpy(
        (imageio.imread(impath) / 255.0).mean(-1, keepdims=True)
    ).permute(2, 0, 1)
    if image.shape[1] == 481:
        image = torch.rot90(image, 1, (-2, -1))
    return image


def load_dataset_as_tensor(data_path, data_type):
    """Function to load a dataset as a tensor.

    Parameters
    ----------
    base_path
        Base path to the location where the dataset is stored.
        Important: This is assumed to be the path without the name of the dataset,
        for instance `/opt/project/data/
    data_type
        Either "full", "train", "test" or "val".

    Returns
    -------
    A torch tensor containing the images.
    """
    images = []
    if data_type == "full":
        folder_names = ["train", "test", "val"]
    else:
        folder_names = [data_type]
    for name in folder_names:
        path = pathlib.Path(os.path.join(data_path, DATASET_NAME, "small_BSDS500/data/images", name))
        for impath in path.iterdir():
            if not impath.as_uri().endswith("jpg"):
                continue
            image = _read_image(impath)
            images.append(image)

    return torch.stack(images)


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset that can be used for loading image data from files.

    Implements in particular the `__len__` method and the `__getitem__` method.

    Parameters
    ----------
    base_path
        Base path to the location where the dataset is stored.
        Important: This is assumed to be the path without the name of the dataset,
        for instance `/opt/project/data/
    data_type
        Either "full", "train", "test" or "val".
    """
    def __init__(self, data_path, data_type):
        if data_type == "full":
            folder_names = ["train", "test", "val"]
        else:
            folder_names = [data_type]
        self.image_paths = []

        for name in folder_names:
            path = pathlib.Path(os.path.join(data_path, DATASET_NAME, "small_BSDS500/data/images", name))
            for impath in path.iterdir():
                if not impath.as_uri().endswith("jpg"):
                    continue
                self.image_paths.append(impath)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return _read_image(self.image_paths[idx])[None, ...]


class ExampleDataLoader(BaseDataLoader):
    dataset_name = DATASET_NAME

    def __init__(self, data_path):
        super().__init__(data_path)

    def _get_dataset_as_tensor(self, data_type="full"):
        return load_dataset_as_tensor(self.data_path, data_type)

    def _get_dataset(self, data_type="full"):
        return CustomDataset(self.data_path, data_type)

    def _download_dataset(self):
        download_from_nextcloud(self.dataset_name, self.data_path)


register_data_loader(DATASET_NAME, ExampleDataLoader)
