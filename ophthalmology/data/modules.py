# -*- coding: utf-8 -*-
"""## datamodules

This module contains the pytorch_lightning datamodules for the different tasks and datasets
"""

import math
import os
from typing import Iterator, List, Optional

import medmnist
import pytorch_lightning as pl
import torch
from loguru import logger as log
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from ophthalmology import samplers
from ophthalmology.data import sets


class DiabeticRetinopythyDetection(pl.LightningDataModule):
    """Pytorch lightning datamodule for the DiabeticRetinopythyDetection dataset"""

    def __init__(
        self,
        train_transform: torch.nn.Module,
        image_dir: str = "",
        csv_file_train: str = "",
        csv_file_test: str = "",
        test_transform: Optional[torch.nn.Module] = None,
        train_test_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        seed: int = 42,
        balanced_sampling: bool = False,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(DiabeticRetinopythyDetection, self).__init__()
        self.train_test_split = train_test_split
        self.seed = seed
        self.balanced_sampling = balanced_sampling

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_set = sets.DiabeticRetinopythyDetection(
            image_dir, csv_file_train, train_transform
        )

        self.test_dataset = sets.DiabeticRetinopythyDetection(
            image_dir,
            csv_file_test,
            test_transform,
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
            pin_memory=self.pin_memory,
            sampler=samplers.ImbalancedDatasetSampler(
                self.train_dataset,
                callback_get_label=lambda dataset: dataset.dataset.get_labels(
                    dataset.indices
                ),
                seed=self.seed,
            )
            if self.balanced_sampling
            else None,
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
            pin_memory=self.pin_memory,
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
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )


class IndianDiabeticRetinopythyDetection(pl.LightningDataModule):
    """Pytorch lightning datamodule for the IndianDiabeticRetinopythyDetection dataset"""

    def __init__(
        self,
        train_transform: torch.nn.Module,
        image_dir_train: str = "",
        image_dir_test: str = "",
        csv_file_train: str = "",
        csv_file_test: str = "",
        test_transform: Optional[torch.nn.Module] = None,
        train_test_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        seed: int = 42,
        balanced_sampling: bool = False,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(IndianDiabeticRetinopythyDetection, self).__init__()
        self.train_test_split = train_test_split
        self.seed = seed
        self.balanced_sampling = balanced_sampling

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_set = sets.IndianDiabeticRetinopythyDetection(
            image_dir_train, csv_file_train, train_transform
        )

        self.test_dataset = sets.IndianDiabeticRetinopythyDetection(
            image_dir_test,
            csv_file_test,
            test_transform,
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
            pin_memory=self.pin_memory,
            sampler=samplers.ImbalancedDatasetSampler(
                self.train_dataset,
                callback_get_label=lambda dataset: dataset.dataset.get_labels(
                    dataset.indices
                ),
                seed=self.seed,
            )
            if self.balanced_sampling
            else None,
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
            pin_memory=self.pin_memory,
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
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )


class IndianDiabeticRetinopythyDetectionLocalization(pl.LightningDataModule):
    """Pytorch lightning datamodule for the IndianDiabeticRetinopythyDetection dataset"""

    def __init__(
        self,
        train_transform: torch.nn.Module,
        image_dir_train: str = "",
        image_dir_test: str = "",
        csv_file_train_disk: str = "",
        csv_file_test_disk: str = "",
        csv_file_train_fovea: str = "",
        csv_file_test_fovea: str = "",
        test_transform: Optional[torch.nn.Module] = None,
        train_test_split: float = 0.8,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        seed: int = 42,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(IndianDiabeticRetinopythyDetectionLocalization, self).__init__()
        self.train_test_split = train_test_split
        self.seed = seed

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.data_set = sets.IndianDiabeticRetinopythyDetectionLocalization(
            image_dir_train,
            csv_file_train_disk,
            csv_file_train_fovea,
            train_transform,
        )

        self.test_dataset = sets.IndianDiabeticRetinopythyDetectionLocalization(
            image_dir_test,
            csv_file_test_disk,
            csv_file_test_fovea,
            test_transform,
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
            pin_memory=self.pin_memory,
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
            pin_memory=self.pin_memory,
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
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            generator=torch.Generator().manual_seed(self.seed),
        )


class RetinaMNIST(pl.LightningDataModule):
    """Pytorch lightning datamodule for the RetinaMNIST dataset"""

    def __init__(
        self,
        data_dir,
        train_transform: torch.nn.Module = None,
        test_transform: Optional[torch.nn.Module] = None,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        seed: int = 42,
        balanced_sampling: bool = False,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(RetinaMNIST, self).__init__()
        self.seed = seed
        self.balanced_sampling = balanced_sampling

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = medmnist.dataset.RetinaMNIST(
            split="train",
            transform=train_transform,
            download=True,
            root=data_dir,
            target_transform=lambda target: torch.tensor(target.item()),
        )

        self.val_dataset = medmnist.dataset.RetinaMNIST(
            split="val",
            transform=test_transform,
            download=True,
            root=data_dir,
            target_transform=lambda target: torch.tensor(target.item()),
        )

        self.test_dataset = medmnist.dataset.RetinaMNIST(
            split="test",
            transform=test_transform,
            download=True,
            root=data_dir,
            target_transform=lambda target: torch.tensor(target.item()),
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            sampler=samplers.ImbalancedDatasetSampler(
                self.train_dataset,
                callback_get_label=lambda dataset: [
                    dataset[i][1].item() for i in range(len(dataset))
                ],
                seed=self.seed,
            )
            if self.balanced_sampling
            else None,
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
            pin_memory=self.pin_memory,
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
            pin_memory=self.pin_memory,
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
        pin_memory: bool = False,
        seed: int = 42,
        balanced_sampling: bool = False,
        use_test_fraction: Optional[float] = None,
    ):
        """
        Initialization of inherited lightning data module
        """
        super(SSLDiabeticRetinopythyDetection, self).__init__()
        self.train_test_split = train_test_split
        self.seed = seed
        self.balanced_sampling = balanced_sampling

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.test_dataset = sets.DiabeticRetinopythyDetection(
            image_dir, csv_file, test_transform, use_test_fraction
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
            pin_memory=self.pin_memory,
            sampler=samplers.ImbalancedDatasetSampler(
                self.train_dataset,
                callback_get_label=lambda dataset: dataset.dataset.dataset.get_labels(
                    dataset.indices
                ),
                seed=self.seed,
            )
            if self.balanced_sampling
            else None,
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
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )
