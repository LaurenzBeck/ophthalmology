# -*- coding: utf-8 -*-
"""This is a general training script supporting:
- single training runs
- multiruns with hydra-style grid-search

The runs are configured through the hydra framework.
https://hydra.cc/docs/intro/
"""

import os
from typing import List

import hydra
import mlflow
import pytorch_lightning as pl
import torch
import torchvision
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

from ophthalmology import callbacks


@hydra.main(config_path="../conf/", config_name="ssl_config")
def main(config: DictConfig):

    if config.get("print_config"):
        log.info(f"hydra configuration:\n{OmegaConf.to_yaml(config)}")

    if "seed" in config:
        pl.utilities.seed.seed_everything(config.seed, workers=True)

    model: torch.nn.Module = hydra.utils.call(config.model)

    if config.load_model:
        model.load_state_dict(torch.load(config.load_model))
        log.info(f"loading model {config.load_model}")
    else:
        log.info("using a fresh model")

    ssl_transforms: torch.nn.Module = hydra.utils.instantiate(
        config.ssl_transforms
    )

    test_transforms: torch.nn.Module = hydra.utils.instantiate(
        config.test_transforms
    )

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config.datamodule,
        ssl_transform=ssl_transforms,
        test_transform=test_transforms,
    )

    lightning_module: pl.LightningModule = hydra.utils.instantiate(
        config.lightning_module,
        model=model,
        num_train_samples=len(datamodule.train_dataset),
    )

    trainer_callbacks: List[pl.Callback] = [
        hydra.utils.instantiate(cb_conf)
        for _, cb_conf in config.callbacks.items()
        if "callbacks" in config and "_target_" in cb_conf
    ] + [
        callbacks.SSLOnlineEvaluator(
            datamodule.test_dataloader(),
            drop_p=0.2,
            hidden_dim=1024,
            z_dim=2048,
            num_classes=5,
        ),
        callbacks.LogDataSamplesCallback(datamodule.train_dataset, rows=5),
        callbacks.LogSignalPropagationPlotCallback(
            input_shape=[
                16,
                3,
                config.image_size,
                config.image_size,
            ]
        ),
    ]

    trainer: pl.Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=trainer_callbacks, _convert_="partial"
    )

    mlflow.set_tracking_uri(f"file:{config.environment.logdir}/mlruns")
    mlflow.set_experiment(config.logger.experiment_name)

    with mlflow.start_run(
        run_id=config.logger.run_id, run_name=config.logger.run_name
    ) as run:
        mlflow.log_param("run_id", run.info.run_id)

        mlflow.pytorch.autolog(
            log_models=False,
            silent=False,
        )

        # Train the model ⚡
        trainer.fit(lightning_module, datamodule)

        if config.save_model:
            log.info(f"saving model to {config.save_model}")
            torch.save(lightning_module.model.state_dict(), config.save_model)
            mlflow.pytorch.log_model(lightning_module.model, "pytorch_model")


if __name__ == "__main__":
    main()
