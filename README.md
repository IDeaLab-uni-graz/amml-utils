# amml-utils

Collection of Python utility scripts for the IDea_Lab AMML Group. [maintainer=@HenKlei]

## Datasets

A collection of datasets is provided via this package. The data itself is either hosted on UniCloud or downloaded from
a public repository.

The most important components are described in the following subsections:

### Working with existing datasets

You can mainly list all available datasets or download/load a particular dataset. Each dataset is downloaded only once
(as long as you use the same path at which to store the data and load it from) and whenever you try to access
a dataset, it is first checked if the dataset is already available. This avoids unnecessary downloading of datasets.

See also [test.py](amml_utils/test.py) for an example on how to work with the datasets.

#### List all available datasets

To get a list of all datasets currently available, run:
```python
from amml_utils import list_datasets

print(f"Available datasets: {list_datasets()}")
```
However, to **download** datasets from UniCloud, you need to have your credentials set as environment variables,
see the section below on "Datasets on UniCloud".

#### Load a specific datasets

To load a selected dataset (and potentially download it first in case it is not available on disk so far), run:
```python
from amml_utils import get_dataset, list_datasets

dataset = get_dataset(<NAME OF THE DATASET>)
```
You can also create a
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
from the dataset:
```python
data_loader = DataLoader(dataset)
```
If you want to access only a certain subset of the dataset, such as the training or the validation set, you can pass
an additional option `subset` to the `get_dataset` method:
```python
training_dataset = get_dataset(<NAME OF THE DATASET>, "train")
validation_dataset = get_dataset(<NAME OF THE DATASET>, "val")
```
Options for the `subset` argument are `"full"`, `"train"`, `"test"` and `"val"` to choose between all data,
training, test and validation data.
Moreover, you can specify the data path:
```python
dataset = get_dataset(<NAME OF THE DATASET>, data_path=<PATH TO THE DATA FOLDER>)
```
By default `get_dataset` looks for an environment variable called `BASE_DIRECTORY`. If this variable is available,
the datasets will be stored (are assumed to be stored) under `BASE_DIRECTORY/data/`. Otherwise, if no
particular `data_path` is provided, the datasets will be stored in a `data` folder next to the script running the
`get_dataset` function.

### Adding new datasets

To add new datasets to the collection of datasets, you need to add a new Python file to the
[datasets](amml_utils/datasets) directory. In this file, you create the dataset as a subclass of
[torch.utils.data.Dataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). The dataset
is supposed to implement at least a suitable constructor as `__init__` method, the `__len__` method
and a `__getitem__` method. The `__init__` method takes as first two arguments (in this order)
the `data_path` (path to the folder in which the datasets are stored) and the `subset`
(either `"full"`, `"train"`, `"test"` or `"val"` to choose between all data, training, test or validation data)
Additionally, depending on the source of the dataset (public repository
or UniCloud; see below for more extensive explanations) you also need to implement a method to download the data
to a given path, which takes the `data_path` as sole input. Afterwards, you register the dataset via its name
using the `register_dataset` method available in [amml_utils.registry](amml_utils/registry.py).

The file for your new dataset might look something like this:
```python
import torch

from amml_utils.registry import register_dataset


DATASET_NAME = <NAME OF THE DATASET>

def download_dataset(data_path):
    ...


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, subset):
        ...

    def __len__(self):
        ...

    def __getitem__(self, idx):
        ...


register_dataset(DATASET_NAME, download_dataset, CustomDataset)
```

#### Datasets in a public repository

For datasets that are publicly available it is recommended not to store them in UniCloud as well. Instead, the data
should be downloaded from the public repository. To this end, you need to implement a `download_dataset` function
that needs to be used when registering the dataset, i.e.
```python
register_dataset(DATASET_NAME, download_dataset, CustomDataset)
```
The `download_dataset` function is supposed to download the dataset to the provided `data_path`. However,
it is **not** required that the `download_dataset` method checks if the data is already there. This check is done
automatically before calling the method.

See also [bsds500.py](amml_utils/datasets/bsds500.py) for an example using a dataset stored in a public repository.

#### Datasets on UniCloud

In order to retrieve data from UniCloud, you need to set your credentials as the following environment variables:
`NEXTCLOUD_USERNAME`, `NEXTCLOUD_URL`, `NEXTCLOUD_PASSWORD`.
For instance:
```shell
NEXTCLOUD_USERNAME=hendrik.kleikamp
NEXTCLOUD_URL=https://cloud.uni-graz.at/
NEXTCLOUD_PASSWORD=<YOUR DEVICE PASSWORD>
```
where the `NEXTCLOUD_PASSWORD` has to be created as a device password in the UniCloud settings
(this is not your Uni Graz password!).

If you want to use the download method via nextcloud, you can pass `None` as `download_function`
to the `register_dataset` method, i.e.
```python
register_dataset(DATASET_NAME, None, CustomDataset)
```
See also [example.py](amml_utils/datasets/example.py) for an example using a dataset stored on UniCloud.

#### Todos:

- Add to all docker images
- Add templates in amml-python-template and amml-python-ml-template that load one of the datasets and show a basic example, e.g, training something small
