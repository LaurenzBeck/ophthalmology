# -*- coding: utf-8 -*-
"""## Supervised Optical Disk and Fovea Localization Task.

pytorch-lightning module for the supervised localization task.
"""

import math
from typing import List, Optional, Tuple, Union

import mlflow
import pytorch_lightning as pl
import snoop
import torch
import torchinfo
import torchmetrics
from loguru import logger as log
from matplotlib import pyplot as plt
from torch import nn

from ophthalmology import visualization


class Localization(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        num_train_samples: int,
        learning_rate: float = 3e-3,
        batch_size: int = 32,
        print_model_info_for_input: Optional[
            Union[Tuple[int, ...], List[int]]
        ] = None,
        epochs: int = 120,
        weight_decay: float = 1e-5,
    ):
        """Diabetic Retinopathy Optical Disk and Fovea Localization Task.

        Args:
            model (nn.Module): pytorch model
            loss (nn.Module): pytorch loss function
            num_train_samples (int)
            learning_rate (float, optional) Defaults to 3e-3.
            batch_size (int, optional) Defaults to 32.
            print_model_info_for_input (Tuple[int, ...], optional): If defined, Print an overview over the signal
                dimensions in each layer using the torchinfo.summary function for a given input size (e.g. (16, 1, 28, 28)).
                Defaults to None.
            epochs (int, optional) Defaults to 120.
            weight_decay (float, optional) Defaults to 1e-5.
        """
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["model", "loss"])

        # Turn off automatic optimization because the lr-scheduler makes problems in automatic mode...
        self.automatic_optimization = False

        self.model = model

        # Training Parameters
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_train_samples = num_train_samples

        # losses
        self.loss = loss

        if print_model_info_for_input:
            log.info("printing summary:")
            torchinfo.summary(
                self,
                input_size=tuple(print_model_info_for_input),
                device=self.device,
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx_):
        x, y = batch

        opt = self.optimizers()
        opt.zero_grad()

        pred = self(x)
        loss = self.loss(pred, y)
        self.log("train/loss", loss, prog_bar=True)

        self.manual_backward(loss)
        opt.step()

        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.lr = opt.param_groups[0]["lr"]
        self.log("lr", self.lr, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx_):
        x, y = batch

        pred = self(x)
        val_loss = self.loss(pred, y)
        # perform logging
        self.log("val/loss", val_loss)

        return val_loss

    def test_step(self, batch, batch_idx_):
        x, y = batch

        pred = self(x)
        loss = self.loss(pred, y)

        # perform logging
        self.log("test/loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # We need to multiply the epochs by train_iters_per_epoch, to enable the
        # lr_scheduler.step() call after every training_step. Internally only epoch
        # wise updates are implemented...
        train_iters_per_epoch = self.num_train_samples / float(self.batch_size)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=math.ceil(self.epochs * train_iters_per_epoch),
                last_epoch=-1,
                verbose=False,
            ),
            "name": "cosine_lr_scheduler",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
