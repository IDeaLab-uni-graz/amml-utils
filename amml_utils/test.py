import os
from torch.utils.data import DataLoader

from amml_utils import get_data_loader, list_datasets


print(f"Available datasets: {list_datasets()}")

data_loader = get_data_loader("EXAMPLE", data_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))
tensor = data_loader.get_dataset_as_tensor("train")
print(f"Shape of training set loaded as tensor: {tensor.shape}")

dataset = data_loader.get_dataset()
print(f"Length of full dataset: {len(dataset)}")

data_loader = DataLoader(dataset, batch_size=len(dataset) // 3, shuffle=True)
print("Shapes of batches:")
for x in data_loader:
    print(f"\t{x.shape}")
