from typing import Dict, Type
from pathlib import Path
from amml_utils.base_data_loader import BaseDataLoader

# Registry mapping IDs to loader classes
_LOADER_REGISTRY: Dict[str, Type[BaseDataLoader]] = {}


def register_data_loader(loader_id: str, loader_cls: Type[BaseDataLoader]):
    """Register a new loader class by ID."""
    if loader_id in _LOADER_REGISTRY:
        raise ValueError(f"Loader ID '{loader_id}' already registered.")
    _LOADER_REGISTRY[loader_id] = loader_cls


def get_data_loader(loader_id: str, data_path: Path, **kwargs) -> BaseDataLoader:
    """Instantiate a loader by ID."""
    if loader_id not in _LOADER_REGISTRY:
        raise KeyError(f"Loader '{loader_id}' not found. Available: {list(_LOADER_REGISTRY.keys())}")
    return _LOADER_REGISTRY[loader_id](data_path=data_path, **kwargs)


def list_datasets():
    return list(_LOADER_REGISTRY.keys())
