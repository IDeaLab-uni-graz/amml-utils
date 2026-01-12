import os
from types import SimpleNamespace
from amml_utils.utils import download_from_nextcloud

_DATASET_REGISTRY = {}

BASE_DIRECTORY_ENV_VAR = "BASE_DIRECTORY"


def register_dataset(dataset_name, dataset, download_function=None, version_check_function=None, description=None):
    """Register a new dataset by name."""
    if dataset_name in _DATASET_REGISTRY:
        raise ValueError(f"Dataset '{dataset_name}' already registered.")
    _DATASET_REGISTRY[dataset_name] = {
        'download': download_function,
        'version_check': version_check_function,
        'description': description,
        'type': dataset
    }


def get_dataset(dataset_name, data_path=None, subset="full", strict_version_check=False, try_redownload=True,
                force_download=False, **kwargs):
    """Instantiate a dataset by name."""
    if dataset_name not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {list_datasets()}")

    dataset_info = _DATASET_REGISTRY[dataset_name]
    d = SimpleNamespace(**dataset_info)

    if data_path is None:
        if BASE_DIRECTORY_ENV_VAR in os.environ:
            data_path = os.path.join(os.getenv(BASE_DIRECTORY_ENV_VAR), "data")
        else:
            print(
                f"\033[1mWarning:\033[0m Environment variable '{BASE_DIRECTORY_ENV_VAR}' not set and no data path provided! Using default data path now, which is: " + os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "data"))
            data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    if (dataset_name not in os.listdir(data_path)) or force_download:
        if d.download is None:
            print(f"Downloading files for dataset '{dataset_name}' from nextcloud ...")
            download_from_nextcloud(dataset_name, data_path)
        else:
            print(f"Downloading files for dataset '{dataset_name}' via download function ...")
            d.download(data_path=data_path)

    dataset = d.type(data_path, subset, **kwargs)

    if d.version_check is not None:
        passed, version_info = d.version_check(data_path)
        if not passed:
            if try_redownload:
                # INFO: try_redownload needs to be set to False so we don't get infinite recursion!
                print(
                    f"\033[1mWarning:\033[0m Version check of the dataset '{dataset_name}' failed! Context[ {version_info} ] Trying to re-download...")
                return get_dataset(dataset_name, data_path, subset=subset, strict_version_check=strict_version_check,
                                   try_redownload=False, force_download=True, **kwargs)

            if strict_version_check:
                raise RuntimeError(
                    f"Version check of the dataset '{dataset_name}' failed! Context[ {version_info} ] Aborting...")
            else:
                print(
                    f"\033[1mWarning:\033[0m Version check of the dataset '{dataset_name}' failed! Context[ {version_info} ] Continuing nonetheless...")

    return dataset


def list_datasets():
    return {k: v['description'] for k, v in _DATASET_REGISTRY.items()}
