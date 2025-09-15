import os

# Registry mapping IDs to loader classes
_DATASET_REGISTRY = {}

DATA_DIRECTORY_ENV_VAR = "BASE_DIRECTORY"


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
        if DATA_DIRECTORY_ENV_VAR in os.environ:
            data_path = os.getenv(DATA_DIRECTORY_ENV_VAR)
        else:
            print(f"\033[1mWarning:\033[0m Environment variable '{DATA_DIRECTORY_ENV_VAR}' not set and no data path provided! Using default data path now.")
            data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

    if dataset_name not in os.listdir(data_path):
        download_function(data_path=data_path)
    return dataset(data_path, subset, **kwargs)


def list_datasets():
    return list(_DATASET_REGISTRY.keys())
