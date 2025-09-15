from torch.utils.data import DataLoader

from amml_utils import get_dataset, list_datasets


TEST_DATASET = "BSDS500"

print(f"Available datasets: {list_datasets()}")

train_dataset = get_dataset(TEST_DATASET, subset="train")
print(f"Length of training dataset: {len(train_dataset)}")

full_dataset = get_dataset(TEST_DATASET)
print(f"Length of full dataset: {len(full_dataset)}")

data_loader = DataLoader(full_dataset, batch_size=len(full_dataset) // 3, shuffle=True)
print("Shapes of batches:")
for x in data_loader:
    print(f"\t{x.shape}")
