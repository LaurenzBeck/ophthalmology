# -*- coding: utf-8 -*-
"""## datasets

This file contains pytorch.utils.Dataset classes.
"""

import os
from typing import List, Optional

import pandas as pd
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DiabeticRetinopythyDetection(Dataset):
    """torch.utils.data.Dataset for the Diabetic Retinopathy Detection dataset from:
    https://www.kaggle.com/c/diabetic-retinopathy-detection/data
    """

    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        transform=None,
        use_fraction: Optional[float] = None,
    ):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        if use_fraction:
            self.df = self.df.head(int(len(self.df) * use_fraction))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        image_name, level = entry["image"], entry["level"]
        image_path = os.path.join(self.image_dir, (image_name + ".jpeg"))
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(level, dtype=torch.long))

    def get_labels(self, indices: Optional[List[int]] = None):
        if indices:
            return list(self.df.loc[indices]["level"])
        else:
            return list(self.df["level"])


class IndianDiabeticRetinopythyDetection(Dataset):
    """torch.utils.data.Dataset for the Indian Diabetic Retinopathy Detection dataset from:
    https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    """

    def __init__(
        self,
        image_dir: str,
        csv_file: str,
        transform=None,
        use_fraction: Optional[float] = None,
    ):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        if use_fraction:
            self.df = self.df.head(int(len(self.df) * use_fraction))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        entry = self.df.iloc[index]
        image_name, level = entry["Image name"], entry["Retinopathy grade"]
        image_path = os.path.join(self.image_dir, (image_name + ".jpg"))
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return (image, torch.tensor(level, dtype=torch.long))

    def get_labels(self, indices: Optional[List[int]] = None):
        if indices:
            return list(self.df.loc[indices]["Retinopathy grade"])
        else:
            return list(self.df["Retinopathy grade"])


class IndianDiabeticRetinopythyDetectionLocalization(Dataset):
    """torch.utils.data.Dataset for the Indian Diabetic Retinopathy Detection dataset from:
    https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
    """

    def __init__(
        self,
        image_dir: str,
        csv_file_disk: str,
        csv_file_fovea: str,
        transform=None,
        use_fraction: Optional[float] = None,
    ):
        self.image_dir = image_dir
        self.df_disk = pd.read_csv(csv_file_disk).dropna(subset=["Image No"])
        self.df_fovea = pd.read_csv(csv_file_fovea).dropna(subset=["Image No"])
        self.transform = transform

        if use_fraction:
            self.df_disk = self.df_disk.head(
                int(len(self.df_disk) * use_fraction)
            )
            self.df_fovea = self.df_fovea.head(
                int(len(self.df_fovea) * use_fraction)
            )

    def __len__(self):
        return len(self.df_disk)

    def __getitem__(self, index):
        entry_disk = self.df_disk.iloc[index]
        entry_fovea = self.df_fovea.iloc[index]
        image_name = entry_disk["Image No"]
        disk_x = entry_disk["X- Coordinate"] / 4288  # image width
        disk_y = entry_disk["Y - Coordinate"] / 2848  # image height
        fovea_x = entry_fovea["X- Coordinate"] / 4288  # image width
        fovea_y = entry_fovea["Y - Coordinate"] / 2848  # image height
        image_path = os.path.join(self.image_dir, (image_name + ".jpg"))
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor([disk_x, disk_y, fovea_x, fovea_y], dtype=torch.float),
        )


class SimCLRWrapper(Dataset):
    """This class wraps a pytorch Dataset and performs two transformations
    on each __get_item__ call as needed for the SimCLR framework.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: List[torch.nn.Module] = [],
    ):
        """Initializer for The SimCLRWrapper

        Args:
            dataset (Dataset): Pytorch Dataset instance
            transform (List[torch.nn.Module], optional): torchvision transforms to apply after mixup is performed. Defaults to None.
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image, label = self.dataset[idx]

        x_1 = self.transform(image)
        x_2 = self.transform(image)

        return x_1, x_2, label
