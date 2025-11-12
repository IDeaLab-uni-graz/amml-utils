import imageio.v3 as imageio
import os
import pathlib
import torch
from torchvision.transforms import RandomCrop

from amml_utils.registry import register_dataset


DATASET_NAME = "IMAGENET"

STANDARD_WIDTH = 256
STANDARD_HEIGHT = 256


def standard_transform(image):
    image = image.mean(-1, dtype=float)
    return RandomCrop((STANDARD_HEIGHT, STANDARD_WIDTH)).forward(image)


def download_dataset(data_path):
    print(f"\033[1mWarning:\033[0m The dataset has a size of about 160GB!")
    import requests
    import tarfile
    for dataset_path, dataset_type in [("train", "train"), ("val", "val"), ("test_v10102019", "test")]:
        r = requests.get(f"https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_{dataset_type}.tar")
        filename = os.path.join(data_path, f"ILSVRC2012_img_{dataset_type}.tar")
        with open(filename, "wb") as fd:
            fd.write(r.content)
        with tarfile.TarFile(filename, "r") as tar_ref:
            tar_ref.extractall(os.path.join(data_path, DATASET_NAME, dataset_type))
        os.remove(filename)


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
                if not impath.as_uri().endswith("png"):
                    continue
                self.image_paths.append(impath)

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = torch.from_numpy(imageio.imread(self.image_paths[idx]))
        if self.transform:
            image = self.transform(image)
        return image


register_dataset(DATASET_NAME, download_dataset, CustomDataset)
