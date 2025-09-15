# amml-utils

Collection of Python utility scripts for the IDea_Lab AMML Group. [maintainer=@HenKlei]

#### Todos:

- Explain in readme how to add dataset.
- Add to all docker images
- Add templates in amml-python-template and amml-python-ml-template that load one of the datasets and show a basic example, e.g, training something small
- Replace class BaseDataLoader by a single function that is called with
   get_data_set(NAME,data_path=Optional,subset=full)
   where
   data_path= is optional, defaults to BASE_DIRECTORY/data if BASE_DIRECTORY is available via env, alternatively filepath/data with warning
   subset in {full,train,test,val)
- Idealy, a single dataset is a file like bsds500.py with two functions: class CustomDataset(torch.utils.Dataset) with his just standard torch dataset with __init__, __len__ and __getitem__ and an optional transform + the function download_dataset
