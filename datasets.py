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

from config import DATA_ROOT_DIR
from utils import convert_I_to_L, undersample


class NIHChestXrays(Dataset):
    """
    https://www.kaggle.com/datasets/nih-chest-xrays/data
    """

    names = ["AP", "PA"]

    def __init__(self, transform: Optional[Callable] = None, split: Optional[str] = None, undersampling: bool = False):
        assert split in ("train", "val", "test", None), f"Invalid `split` {split}"

        self.split = split
        self.transform = transform

        data = pd.read_pickle(os.path.join(DATA_ROOT_DIR, "nih-chest-xrays", "data",
                                           "Data_Entry_2017.pkl"))

        if undersampling:
            data = undersample(data, "label", random_state=42)

        indices = np.arange(len(data))
        train_indices, valtest_indices = train_test_split(indices, test_size=0.4, random_state=42)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=42)

        if split == "train":
            data = data.iloc[train_indices]
        elif split == "val":
            data = data.iloc[val_indices]
        elif split == "test":
            data = data.iloc[test_indices]

        self.data = data

    def count(self):
        return self.data.groupby("label").size()

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

    def __init__(self, transform: Optional[Callable] = None, split: Optional[str] = None, undersampling: bool = False):
        assert split in ("train", "val", "test", None), f"Invalid `split` {split}"

        self.split = split
        self.transform = transform

        data = pd.read_pickle(os.path.join(DATA_ROOT_DIR, "raddar", "chest-xrays-indiana-university",
                                           "indiana_projections.pkl"))

        if undersampling:
            data = undersample(data, "label", random_state=42)

        indices = np.arange(len(data))
        train_indices, valtest_indices = train_test_split(indices, test_size=0.4, random_state=0)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=0)

        if split == "train":
            data = data.iloc[train_indices]
        elif split == "val":
            data = data.iloc[val_indices]
        elif split == "test":
            data = data.iloc[test_indices]

        self.data = data

    def count(self):
        return self.data.groupby("label").size()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img = Image.open(row["path"]).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, row["label"]

    def __len__(self):
        return len(self.data)


class PadChest(Dataset):
    """
    https://bimcv.cipf.es/bimcv-projects/padchest/
    """

    names = ["AP", "PA", "AP-horizontal", "lateral"]

    def __init__(self, transform: Optional[Callable] = None, split: Optional[str] = None, goal: Optional[str] = None,
                 undersampling: bool = False):
        assert split in ("train", "val", "test", None), f"Invalid `split` {split}"
        assert goal in ("ap_vs_pa", "frontal_vs_lateral", None), f"Invalid `goal` {goal}"

        self.goal = goal
        self.split = split
        self.transform = transform

        data = pd.read_pickle(os.path.join(DATA_ROOT_DIR, "PadChest",
                                           "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.pkl"))

        if goal == "ap_vs_pa":
            data = data[data["label"].isin([0, 1, 2])]
            data["label"] = data["label"].replace({
                0: 0,
                1: 1,
                2: 0
            })
            self.names = ["AP", "PA"]
        elif goal == "frontal_vs_lateral":
            data["label"] = data["label"].replace({
                0: 0,
                1: 0,
                2: 0,
                3: 1
            })
            self.names = ["frontal", "lateral"]

        if undersampling:
            data = undersample(data, "label", random_state=42)

        indices = np.arange(len(data))
        train_indices, valtest_indices = train_test_split(indices, test_size=0.4, random_state=0)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=0)

        if split == "train":
            data = data.iloc[train_indices]
        elif split == "val":
            data = data.iloc[val_indices]
        elif split == "test":
            data = data.iloc[test_indices]

        self.data = data

    def count(self):
        return self.data.groupby("label").size()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        try:
            img = Image.open(row["path"])
            img = convert_I_to_L(img)
        except Exception as e:
            print(row["path"])
            raise e

        if self.transform is not None:
            img = self.transform(img)

        return img, row["label"]

    def __len__(self):
        return len(self.data)


class CheXpert(Dataset):
    """
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    names = ["AP", "PA", "lateral"]

    def __init__(self, transform: Optional[Callable] = None, split: Optional[str] = None, goal: Optional[str] = None,
                 undersampling: bool = False):
        assert split in ("train", "val", "test", None), f"Invalid `split` {split}"
        assert goal in ("ap_vs_pa", "frontal_vs_lateral", None), f"Invalid `goal` {goal}"

        self.goal = goal
        self.split = split
        self.transform = transform

        data = pd.read_pickle(os.path.join(DATA_ROOT_DIR, "CheXpert", "data.pkl"))

        if goal == "ap_vs_pa":
            data = data[data["label"].isin([0, 1])]
            self.names = ["AP", "PA"]
        elif goal == "frontal_vs_lateral":
            data["label"] = data["label"].replace({
                0: 0,
                1: 0,
                2: 1
            })
            self.names = ["frontal", "lateral"]

        if undersampling:
            data = undersample(data, "label", random_state=42)

        indices = np.arange(len(data))
        train_indices, valtest_indices = train_test_split(indices, test_size=0.4, random_state=0)
        val_indices, test_indices = train_test_split(valtest_indices, test_size=0.5, random_state=0)

        if split == "train":
            data = data.iloc[train_indices]
        elif split == "val":
            data = data.iloc[val_indices]
        elif split == "test":
            data = data.iloc[test_indices]

        self.data = data

    def count(self):
        return self.data.groupby("label").size()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        img = Image.open(row["path"]).convert("L")

        if self.transform is not None:
            img = self.transform(img)

        return img, row["label"]

    def __len__(self):
        return len(self.data)
