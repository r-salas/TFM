#
#
#   Datasets
#
#

import os
import h5py
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Optional, Sequence
from sklearn.model_selection import train_test_split

from config import DATA_ROOT_DIR
from utils import undersample


class NIHChestXrays(Dataset):
    """
    https://www.kaggle.com/datasets/nih-chest-xrays/data
    """

    names = ["AP", "PA"]

    def __init__(self, transform: Optional[Callable] = None, undersampling: bool = False, split: Optional[str] = None):
        assert split in ("train", "val", None), f"Invalid `split` {split}"

        self.split = split
        self.transform = transform

        with h5py.File(os.path.join(DATA_ROOT_DIR, "nih-chest-xrays", "nih-chest-xrays.h5"), "r") as f:
            self.labels = pd.Series(f["labels"][:])

        if split is not None:
            indices = np.arange(len(self.labels))
            train_indices, val_indices = train_test_split(indices, test_size=0.25, random_state=42)

            if split == "train":
                self.labels = self.labels.iloc[train_indices]
            elif split == "val":
                self.labels = self.labels.iloc[val_indices]

        if undersampling:
            self.labels = undersample(self.labels, random_state=42)

        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(os.path.join(DATA_ROOT_DIR, "nih-chest-xrays", "nih-chest-xrays.h5"), "r")

        orig_index = self.labels.index[index]

        img = Image.fromarray(self.file["data"][orig_index])
        label = self.file["labels"][orig_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class ChestXRaysIndianaUniversity(Dataset):
    """
    https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university
    """

    names = ["frontal", "lateral"]

    def __init__(self, transform: Optional[Callable] = None, undersampling: bool = False, split: Optional[str] = None):
        assert split in ("train", "val", None), f"Invalid `split` {split}"

        self.transform = transform
        self.undersampling = undersampling
        self.split = split

        with h5py.File(os.path.join(DATA_ROOT_DIR, "chest-xrays-indiana-university",
                                    "chest-xrays-indiana-university.h5"), "r") as f:
            self.labels = pd.Series(f["labels"][:])

        if split is not None:
            indices = np.arange(len(self.labels))
            train_indices, val_indices = train_test_split(indices, test_size=0.25, random_state=42)

            if split == "train":
                self.labels = self.labels.iloc[train_indices]
            elif split == "val":
                self.labels = self.labels.iloc[val_indices]

        if undersampling:
            self.labels = undersample(self.labels, random_state=42)

        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(os.path.join(DATA_ROOT_DIR, "chest-xrays-indiana-university",
                                               "chest-xrays-indiana-university.h5"), "r")

        orig_index = self.labels.index[index]

        img = Image.fromarray(self.file["data"][orig_index])
        label = self.file["labels"][orig_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class PadChest(Dataset):
    """
    https://bimcv.cipf.es/bimcv-projects/padchest/
    """

    names = ["AP", "PA", "AP-horizontal", "lateral"]

    def __init__(self, transform: Optional[Callable] = None, undersampling: bool = False,
                 exclude_labels: Optional[Sequence[int]] = None):
        self.transform = transform
        self.exclude_labels = exclude_labels

        with h5py.File(os.path.join(DATA_ROOT_DIR, "PadChest", "padchest.h5"), "r") as f:
            self.labels = pd.Series(f["labels"][:])

        if exclude_labels is not None:
            self.labels = self.labels[~self.labels.isin(exclude_labels)]

        if undersampling:
            self.labels = undersample(self.labels, random_state=42)

        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(os.path.join(DATA_ROOT_DIR, "PadChest", "padchest.h5"), "r")

        orig_index = self.labels.index[index]

        img = Image.fromarray(self.file["data"][orig_index])
        label = self.file["labels"][orig_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class CheXpert(Dataset):
    """
    https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    names = ["AP", "PA", "lateral"]

    def __init__(self, transform: Optional[Callable] = None, undersampling: bool = False,
                 exclude_labels: Optional[Sequence[int]] = None, split: Optional[str] = None):
        assert split in ("train", "val", None), f"Invalid `split` {split}"

        self.transform = transform
        self.undersampling = undersampling

        with h5py.File(os.path.join(DATA_ROOT_DIR, "CheXpert", "chexpert.h5"), "r") as f:
            self.labels = pd.Series(f["labels"][:])

        if split is not None:
            indices = np.arange(len(self.labels))
            train_indices, val_indices = train_test_split(indices, test_size=0.25, random_state=42)

            if split == "train":
                self.labels = self.labels.iloc[train_indices]
            elif split == "val":
                self.labels = self.labels.iloc[val_indices]

        if exclude_labels is not None:
            self.labels = self.labels[~self.labels.isin(exclude_labels)]

        if undersampling:
            self.labels = undersample(self.labels, random_state=42)

        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(os.path.join(DATA_ROOT_DIR, "CheXpert", "chexpert.h5"), "r")

        orig_index = self.labels.index[index]

        img = Image.fromarray(self.file["data"][orig_index])
        label = self.file["labels"][orig_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class UNIFESPXrayBodyPart(Dataset):
    """
    https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier
    """

    names = ['Abdomen', 'Ankle', 'Cervical Spine', 'Chest', 'Clavicles', 'Elbow', 'Feet', 'Finger', 'Forearm', 'Hand',
             'Hip', 'Knee', 'Lower Leg', 'Lumbar Spine', 'Others', 'Pelvis', 'Shoulder', 'Sinus', 'Skull', 'Thigh',
             'Thoracic Spine', 'Wrist']

    def __init__(self, transform: Optional[Callable] = None, exclude_labels: Optional[Sequence[int]] = None):
        self.transform = transform
        self.exclude_labels = exclude_labels

        with h5py.File(os.path.join(DATA_ROOT_DIR, "unifesp-x-ray-body-part-classifier",
                                    "unifesp-x-ray-body-part-classifier.h5"), "r") as f:
            self.labels = pd.Series(f["labels"][:])

        if exclude_labels is not None:
            def filter_fn(label_str):
                return any([int(x) in exclude_labels for x in label_str.decode().split(";")])
            self.labels = self.labels[~self.labels.apply(filter_fn)]

        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(os.path.join(DATA_ROOT_DIR, "unifesp-x-ray-body-part-classifier",
                                               "unifesp-x-ray-body-part-classifier.h5"), "r")

        orig_index = self.labels.index[index]

        img = Image.fromarray(self.file["data"][orig_index])
        label = self.file["labels"][orig_index].decode()

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)

    def __del__(self):
        if self.file is not None:
            self.file.close()


class MiniImagenet(Dataset):

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

        with h5py.File(os.path.join(DATA_ROOT_DIR, "mini-imagenet",
                                    "mini-imagenet.h5"), "r") as f:
            self.length = len(f["data"])

        self.file = None

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(os.path.join(DATA_ROOT_DIR, "mini-imagenet",
                                               "mini-imagenet.h5"), "r")

        img = Image.fromarray(self.file["data"][index])

        if self.transform is not None:
            img = self.transform(img)

        return img, np.float32(np.nan)

    def __len__(self):
        return self.length

    def __del__(self):
        if self.file is not None:
            self.file.close()
