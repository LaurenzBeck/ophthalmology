# -*- coding: utf-8 -*-
"""Python utilities"""

from typing import List

import mlflow
from omegaconf import DictConfig


def log_hyperparameters(
    config: DictConfig, extra_fields: List[str] = []
) -> None:
    """This method logs some configured hyperparameters to mlflow."""

    def to_dict(config: DictConfig, key: str) -> dict:
        return {str(key) + "." + k: val for k, val in config[key].items()}

    mlflow.log_params(to_dict(config, "trainer"))
    mlflow.log_params(to_dict(config, "model"))
    mlflow.log_params(to_dict(config, "lightning_module"))
    mlflow.log_params(to_dict(config, "datamodule"))
    mlflow.log_param("seed", config["seed"])

    for field in extra_fields:
        mlflow.log_params(to_dict(config, field))
