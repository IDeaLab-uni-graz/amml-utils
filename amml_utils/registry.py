import os

from amml_utils.utils import download_from_nextcloud


_DATASET_REGISTRY = {}

BASE_DIRECTORY_ENV_VAR = "BASE_DIRECTORY"


def register_dataset(dataset_name, download_function, dataset):
    """Register a new dataset by name."""
    if dataset_name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' already registered.")
    _DATASET_REGISTRY[dataset_name] = [download_function, dataset]


def get_dataset(dataset_name, data_path=None, subset="full", **kwargs):
    """Instantiate a dataset by name."""
    if dataset_name not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {list_datasets()}")

    download_function, dataset = _DATASET_REGISTRY[dataset_name]

    if data_path is None:
        if BASE_DIRECTORY_ENV_VAR in os.environ:
            data_path = os.path.join(os.getenv(BASE_DIRECTORY_ENV_VAR), "data")
        else:
            print(f"\033[1mWarning:\033[0m Environment variable '{BASE_DIRECTORY_ENV_VAR}' not set and no data path provided! Using default data path now, which is: " + os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
            data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    if dataset_name not in os.listdir(data_path):
        if download_function is None:
            print(f"Downloading files for dataset '{dataset_name}' from nextcloud ...")
            download_from_nextcloud(dataset_name, data_path)
        else:
            print(f"Downloading files for dataset '{dataset_name}' via download function ...")
            download_function(data_path=data_path)
    return dataset(data_path, subset, **kwargs)


def list_datasets():
    return list(_DATASET_REGISTRY.keys())
