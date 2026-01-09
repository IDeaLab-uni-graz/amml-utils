import pandas as pd
import numpy as np
import os
import pathlib
import torch

from amml_utils.datasets.simulated_bloch import version_check, CustomDataset
from amml_utils.registry import register_dataset

# Format-equivalent version of Simulated_Bloch dataset (therefore re-uses that code)

DATASET_NAME = "RNN_Simulated_Bloch"
DATASET_DESCRIPTION = "Simulated Bloch dataset using IntegratorRNN"
DATASET_VERSION = "v1.1"
DATASET_VERSION_FILE = "version.txt"

def standard_transform(trajectory):
    return trajectory

register_dataset(DATASET_NAME, CustomDataset, description=DATASET_DESCRIPTION,
                 version_check_function=lambda path: version_check(path, DATASET_NAME, DATASET_VERSION,
                                                                   DATASET_VERSION_FILE))
