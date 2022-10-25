#
#
#   Datasets
#
#

import os

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional
from sklearn.model_selection import train_test_split

from config import PREPARED_DATA_ROOT_DIR


class NIHChestXrays(Dataset):
    """
    https://www.kaggle.com/datasets/nih-chest-xrays/data
    """

    names = ["AP", "PA"]

    def __init__(self, transform: Optional[Callable] = None, split: str = "train"):
        self.split = split
        self.transform = transform

        data = pd.read_pickle(os.path.join(PREPARED_DATA_ROOT_DIR, "nih-chest-xrays", "data",
                                           "Data_Entry_2017.pkl"))

        indices = np.arange(len(data))
        train_indices, valtest_indices = train_test_split(indices, test_size=0.4, random_state=0)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=0)

        if split == "train":
            data = data.iloc[train_indices]
        elif split == "val":
            data = data.iloc[val_indices]
        elif split == "test":
            data = data.iloc[test_indices]
        else:
            raise ValueError(f"Unknown split `{split}`")

        self.data = data

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img = Image.open(row["path"]).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, row["label"]

    def __len__(self):
        return len(self.data)


class ChestXRaysIndianaUniversity(Dataset):
    """
    https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
    """

    names = ["frontal", "lateral"]

    def __init__(self, transform: Optional[Callable] = None, split: str = "train"):
        self.split = split
        self.transform = transform

        data = pd.read_pickle(os.path.join(PREPARED_DATA_ROOT_DIR, "raddar", "chest-xrays-indiana-university",
                                           "indiana_projections.pkl"))

        indices = np.arange(len(data))
        train_indices, valtest_indices = train_test_split(indices, test_size=0.4, random_state=0)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=0)

        if split == "train":
            data = data.iloc[train_indices]
        elif split == "val":
            data = data.iloc[val_indices]
        elif split == "test":
            data = data.iloc[test_indices]
        else:
            raise ValueError(f"Unknown split `{split}`")

        self.data = data

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img = Image.open(row["path"]).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, row["label"]

    def __len__(self):
        return len(self.data)
