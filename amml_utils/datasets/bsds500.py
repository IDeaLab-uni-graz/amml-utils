import imageio.v3 as imageio
import os
import pathlib
import torch

from amml_utils.base_data_loader import BaseDataLoader
from amml_utils.registry import register_data_loader


DATASET_NAME = "BSDS500"


def _read_image(impath):
    return torch.from_numpy(imageio.imread(impath))

    
def standard_transform(image):
    image = (image / 255.0).mean(-1, keepdims=True).permute(2, 0, 1)
    if image.shape[1] == 481:
        image = torch.rot90(image, 1, (-2, -1))
    return image


def download_dataset(data_path):
    """
    If the dataset is already on NextCloud, this step can be skipped.
    Otherwise, the dataset should be downloaded from some online source.
    """
    import requests
    r = requests.get("https://github.com/BIDS/BSDS500/archive/master.zip")
    filename = os.path.join(data_path, "BSDS500.zip")
    with open(filename, "wb") as fd:
        fd.write(r.content)
    import zipfile
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(os.path.join(data_path, "BSDS500"))
    os.remove(filename)


class CustomDataset(torch.utils.data.Dataset):
    """Custom dataset that can be used for loading image data from files.

    Implements in particular the `__len__` method and the `__getitem__` method.

    Parameters
    ----------
    base_path
        Base path to the location where the dataset is stored.
        Important: This is assumed to be the path including the name of the dataset,
        for instance `/opt/project/data/EXAMPLE/
    data_type
        Either "full", "train", "test" or "val".
    """
    def __init__(self, data_path, data_type, transform=standard_transform):
        if data_type == "full":
            folder_names = ["train", "test", "val"]
        else:
            folder_names = [data_type]
        self.image_paths = []

        for name in folder_names:
            path = pathlib.Path(os.path.join(data_path, DATASET_NAME, "BSDS500-master/BSDS500/data/images", name))
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


class BSDS500DataLoader(BaseDataLoader):
    dataset_name = DATASET_NAME

    def __init__(self, data_path, transform=standard_transform):
        super().__init__(data_path, transform=transform)

    def _get_dataset_as_tensor(self, data_type="full"):
        return self._get_dataset(data_type=data_type)[...]

    def _get_dataset(self, data_type="full"):
        return CustomDataset(self.data_path, data_type, transform=self.transform)

    def _download_dataset(self):
        download_dataset(self.data_path)


register_data_loader(DATASET_NAME, BSDS500DataLoader)
