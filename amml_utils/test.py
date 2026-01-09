from torch.utils.data import DataLoader

from amml_utils import get_dataset, list_datasets


TEST_DATASET = "RNN_Simulated_Bloch"

print(f"Available datasets: {list_datasets()}")

train_dataset = get_dataset(TEST_DATASET, subset="train")
print(f"Length of training dataset: {len(train_dataset)}")

# Show missing version file exception
get_dataset('Simulated_Bloch')
get_dataset('EXAMPLE')

full_dataset = get_dataset(TEST_DATASET, strict_version_check=True)
print(f"Length of full dataset: {len(full_dataset)}")

data_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)
print("Shapes of batches:")
# for x,y in data_loader:
#    print(f"\t{x.shape}")
#    print(f"\t{y.shape}")
