# -*- coding: utf-8 -*-
"""This is a general training script supporting:
- single training runs
- multiruns with hydra-style grid-search

The runs are configured through the hydra framework.
https://hydra.cc/docs/intro/
"""

import os
from typing import List, OrderedDict

import hydra
import pandas as pd
import pytorch_lightning as pl
import snoop
import torch
import torchvision
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf/", config_name="extract_features_config")
def main(config: DictConfig):

    pl.utilities.seed.seed_everything(config.seed, workers=True)

    model: torch.nn.Module = hydra.utils.call(config.model)

    if config.load_model:
        model.load_state_dict(
            torch.load(hydra.utils.to_absolute_path(config.load_model))
        )
        log.info(f"loading model {config.load_model}")
    else:
        log.info("using a fresh model")

    train_transforms: torch.nn.Module = hydra.utils.instantiate(
        config.train_transforms
    )

    test_transforms: torch.nn.Module = hydra.utils.instantiate(
        config.test_transforms
    )

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule,
        train_transform=train_transforms,
        test_transform=test_transforms,
    )


if __name__ == "__main__":
    main()
