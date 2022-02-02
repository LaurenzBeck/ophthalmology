# -*- coding: utf-8 -*-
"""## datamodules

This module contains the pytorch_lightning datamodules for the different tasks and datasets
"""

import math
import os
from typing import Iterator, List, Optional

import pytorch_lightning as pl
import torch
from loguru import logger as log
from pl_bolts.datamodules import (  # https://lightning-bolts.readthedocs.io/en/latest/deprecated/dataloaders/async.html
    async_dataloader,
)
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from ophthalmology.data import sets


class DiabeticRetinopythyDetection(pl.LightningDataModule):
    """Pytorch lightning datamodule for the DiabeticRetinopythyDetection dataset"""

    def __init__(
        self,
        train_transform: torch.nn.Module,
        image_dir: str = "",
        csv_file_train: str = "",
        csv_file_test: str = "",
        image_transform: Optional[torch.nn.Module] = None,
        train_test_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 1,
        seed: int = 42,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(DiabeticRetinopythyDetection, self).__init__()
        self.train_test_split = train_test_split
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_set = sets.DiabeticRetinopythyDetection(
            image_dir,
            csv_file_train,
            transforms.Compose([image_transform, train_transform]),
        )

        self.test_dataset = sets.DiabeticRetinopythyDetection(
            image_dir,
            csv_file_test,
            image_transform,
        )

        self.num_train_samples = math.floor(
            len(self.data_set) * self.train_test_split
        )
        self.num_val_samples = len(self.data_set) - self.num_train_samples

        log.info(
            f"splitted dataset into {self.num_train_samples} training samples and {self.num_val_samples} validation samples."
        )

        self.train_dataset, self.val_dataset = random_split(
            self.data_set,
            [self.num_train_samples, self.num_val_samples],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def test_dataloader(self):
        """
        :return: output - Testing data loader for the given input
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )


class SSLDiabeticRetinopythyDetection(pl.LightningDataModule):
    """SSL Pytorch lightning datamodule for the DiabeticRetinopythyDetection dataset"""

    def __init__(
        self,
        ssl_transform: torch.nn.Module,
        image_dir: str = "",
        csv_file: str = "",
        test_transform: Optional[torch.nn.Module] = None,
        train_test_split: float = 0.98,
        batch_size: int = 16,
        num_workers: int = 1,
        seed: int = 42,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(SSLDiabeticRetinopythyDetection, self).__init__()
        self.train_test_split = train_test_split
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.test_dataset = sets.DiabeticRetinopythyDetection(
            image_dir, csv_file, test_transform
        )

        self.data_set = sets.SimCLRWrapper(
            sets.DiabeticRetinopythyDetection(image_dir, csv_file, None),
            ssl_transform,
        )

        self.num_train_samples = math.floor(
            len(self.data_set) * self.train_test_split
        )
        self.num_val_samples = len(self.data_set) - self.num_train_samples

        log.info(
            f"splitted dataset into {self.num_train_samples} training samples and {self.num_val_samples} validation samples."
        )

        self.train_dataset, self.val_dataset = random_split(
            self.data_set,
            [self.num_train_samples, self.num_val_samples],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def test_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )
