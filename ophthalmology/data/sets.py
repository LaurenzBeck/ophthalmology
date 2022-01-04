# -*- coding: utf-8 -*-
"""This file contains pytorch.utils.Dataset classes."""

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

    def __init__(self, image_dir: str, csv_file: str, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

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

        return x_1, x_2
