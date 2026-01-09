import os, pickle
import pathlib
from functools import lru_cache

from scipy.io import loadmat
import torch
import numpy as np
import warnings

from amml_utils.registry import register_dataset

DATASET_NAME = "CINEMRI"

class CineMRIDataset(torch.utils.data.Dataset):

    IMG_KEY = "cine"
    SMAP_KEY = "smap"
    ROI_KEY = "roi"
    ID_KEY = "id"

    SUBJECT_FOLDERS = ["01", "02", "07", "09"]

    def __init__(self, data_path, subset):

        if subset == "full":
            folder_names = self.SUBJECT_FOLDERS
        elif subset == "train":
            folder_names = [self.SUBJECT_FOLDERS[0]]
        elif subset == "test":
            folder_names = [self.SUBJECT_FOLDERS[1]]
        elif subset == "val":
            folder_names = [self.SUBJECT_FOLDERS[2]]
        elif subset in self.SUBJECT_FOLDERS:
            folder_names = [subset]
        else:
            raise ValueError(f"Invalid subset {subset}. Must be one of 'full','train','test','val' or {self.SUBJECT_FOLDERS}")

        self.data_paths = []

        for name in folder_names:
            path = pathlib.Path(os.path.join(data_path, DATASET_NAME, name))

            if not path.exists():
                raise FileNotFoundError(f"Folder not found: {path}")

            pkl_file, mat_file = None, None

            for impath in path.iterdir():
                if not impath.is_file():
                    continue
                suf = impath.suffix.lower()
                if suf == ".pkl":
                    pkl_file = impath
                elif suf == ".mat":
                    mat_file = impath

            if pkl_file is None:
                raise ValueError(f"Could not find {pkl_file} in {path}")
            if mat_file is None:
                raise ValueError(f"Could not find {mat_file} in {path}")

            self.data_paths.append((pkl_file, mat_file))

            self.root = data_path

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_image(p_str: str):
        p = pathlib.Path(p_str)

        with open(p, "rb") as f:
            obj = pickle.load(f)

        if not isinstance(obj, dict) or "data" not in obj or "dims" not in obj:
            raise ValueError(f"{p}: expected dict with 'data' and 'dims' keys")

        dims = obj["dims"] # (top, bottom, left, right)
        data = obj["data"]

        # Expect data as (..., H, W) with exactly 3 dims total
        if len(data.shape) != 3 or not isinstance(dims, (list, tuple)) or len(dims) != 4:
            raise ValueError(
                f"{p}: expected data.ndim == 3 and dims length 4, "
                f"got data shape {tuple(data.shape)} and dims {dims}"
            )

        data = np.transpose(data, (2,0,1))[:, np.newaxis, :, :]
        data = torch.as_tensor(data)

        return {
            "data": data,               # shape (T, 1, H, W)
            "dims": tuple(dims),
            "id": pathlib.Path(p).stem,
        }

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_smap(p_str: str):
        p = pathlib.Path(p_str)

        smap_data = loadmat(p)
        assert 'b1' in smap_data, f"{p}: expected 'b1' key in .mat file"

        smap = np.asarray(smap_data['b1'])  # shape (H, W, C)
        smap = np.transpose(smap, (2, 0, 1))[np.newaxis, :, :, :] # shape (1, C, H, W)
        smap = torch.from_numpy(smap)

        return smap # shape (1, C, H, W)

    def __getitem__(self, i):

        image_data = self._load_image(str(self.data_paths[i][0]))
        smap = self._load_smap(str(self.data_paths[i][1]))

        top, bottom, left, right = image_data["dims"]

        if smap.shape[-2:] != image_data["data"].shape[-2:]:
            warnings.warn("Coil data shape does not match image dataset size. Center-cropping coil data.")
            _, _, Ht, Wt = image_data["data"].shape  # (T,1,H,W)
            _, C, Hs, Ws = smap.shape  # (1,C,H,W)
            th, tw = Ht, Wt
            i = max((Hs - th) // 2, 0)
            j = max((Ws - tw) // 2, 0)
            smap = smap[..., i:i + th, j:j + tw]

        return {
            self.IMG_KEY: image_data["data"],
            self.ROI_KEY: (top, bottom, left, right),
            self.SMAP_KEY: smap,
            self.ID_KEY: image_data["id"],
        }

    def __repr__(self):
        return f"CineMRIDataset(root={self.root}, nfiles={len(self)})"

register_dataset(DATASET_NAME, CineMRIDataset)