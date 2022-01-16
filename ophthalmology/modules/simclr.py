# -*- coding: utf-8 -*-
"""## Contrastive representation learning task

SimCLR Pytorch Lightning implementation adopted from:
https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L61-L300
"""

import math
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from loguru import logger as log

from ophthalmology.layers import heads


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        """Normalized Temperature-scaled Cross Entropy Loss adopted from:
        https://github.com/PyTorchLightning/lightning-bolts/blob/75ddb4b96d069378a4c38e75460c8c7eef9f15b7/pl_bolts/models/self_supervised/simclr/simclr_module.py#L223

        Args:
            temperature (float): [description]
            eps (float, optional): [description]. Defaults to 1e-6.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, out_1, out_2):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """

        # out: [2 * batch_size, dim]
        out = torch.cat([out_1, out_2], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = (
            torch.Tensor(neg.shape)
            .fill_(math.e ** (1 / self.temperature))
            .to(neg.device)
        )
        neg = torch.clamp(
            neg - row_sub, min=self.eps
        )  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss


class SimCLR(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        print_model_info_for_input: Optional[
            Union[Tuple[int, ...], List[int]]
        ] = None,
        epochs: int = 60,
        num_train_samples: int = 15378,
        num_features: int = 64,
        num_hidden_projection_features: int = 265,
        num_projection_features: int = 128,
        weight_decay: float = 1e-5,
        temperature: float = 0.1,
    ):
        """SimCLR Pytorch Lightning implementation

        Args:
            learning_rate (float, optional): Defaults to 1e-3.
            batch_size (int, optional): Defaults to 1.
            image_size (int, optional): Size in pixel to resize the input images to. Defaults to 28.
            print_model_info_for_input (Tuple[int, ...], optional): If defined, Print an overview over the signal
                dimensions in each layer using the torchinfo.summary function for a given input size (e.g. (16, 1, 28, 28)).
                Defaults to None.
        """
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["model"])

        # Turn off automatic optimization because the lr-scheduler makes problems in automatic mode...
        self.automatic_optimization = False

        self.model = model

        self.projection = heads.ProjectionHead(
            input_dim=num_features,
            hidden_dim=num_hidden_projection_features,
            output_dim=num_projection_features,
        )

        # Training Parameters
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_train_samples = num_train_samples
        self.weight_decay = weight_decay

        # losses
        self.train_loss = NTXentLoss(temperature)

        if print_model_info_for_input:
            log.info("printing summary")
            torchinfo.summary(
                self,
                input_size=tuple(print_model_info_for_input),
                device=self.device,
            )

        log.info("SimCLR Module ready")

    def forward(self, x):
        return self.model(x.squeeze())

    def training_step(self, batch, batch_idx_):
        x1, x2 = batch

        opt = self.optimizers()
        opt.zero_grad()

        # get h representations
        h1 = self(x1)
        h2 = self(x2)

        # get z projections
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.train_loss(z1, z2)
        self.log(
            "train/loss", loss, on_step=True, on_epoch=False, prog_bar=True
        )

        self.manual_backward(loss)
        opt.step()

        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
        self.lr = opt.param_groups[0]["lr"]
        self.log("lr", self.lr, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx_):
        x1, x2 = batch

        # get h representations
        h1 = self(x1)
        h2 = self(x2)

        # get z projections
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        val_loss = self.train_loss(z1, z2)
        self.log(
            "val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return val_loss

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=("bias", "bn")
    ):
        """Prepare the parameters to only weight-decay non bias and bn layers.
        This has proven to be a best practice.
        Code from: https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py

        Args:
            named_params ([type]): [description]
            weight_decay ([type]): [description]
            skip_list (tuple, optional): [description]. Defaults to ("bias", "bn").

        Returns:
            [type]: [description]
        """
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        params = self.exclude_from_wt_decay(
            self.named_parameters(), weight_decay=self.weight_decay
        )

        optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
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
