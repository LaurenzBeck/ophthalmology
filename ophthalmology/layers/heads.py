# -*- coding: utf-8 -*-
"""Different heads that expect their inputs to be flattened."""

from collections import OrderedDict
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F

from ophthalmology.layers import activations


class MultilayerClassificationHead(nn.Module):
    def __init__(
        self,
        in_features: int = 8,
        hidden_layers: List[int] = [16, 16],
        num_classes: int = 5,
        dropout: Optional[float] = None,
    ):
        """Basic Module consisting of 3 consecutive Conv + BN + ReLU Layers
        with optional striding determined by the downscale option.

        Args:
            in_features (int, optional): number of input feature maps or channels.
            hidden_layers (list[int], optional): list of numbers of neurons in each hidden layer. number of hidden
                layers is determinded by the lenth of the list.
            num_classes (int, optional): number of classes for the classification head. Defaults to 5 for IGBT.
            dropout (Optional[float], optional): If provided include dropout with the given drop probability.
        """
        super(MultilayerClassificationHead, self).__init__()
        hidden_fc_layers = [
            nn.Sequential(
                nn.Linear(hidden_layers[n], hidden_layers[n + 1]),
                activations.SELU(),
            )
            for n in range(len(hidden_layers) - 1)
        ]
        self.head = nn.Sequential(
            nn.AlphaDropout(dropout if dropout else 0.0),
            nn.Linear(in_features, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            activations.SELU(),
            *hidden_fc_layers,
            nn.AlphaDropout(dropout if dropout else 0.0),
            nn.BatchNorm1d(hidden_layers[-1]),
        )
        self.last_classification_layer = nn.Linear(
            hidden_layers[-1], num_classes
        )

        for param in self.head.parameters():
            # biases zero
            if len(param.shape) == 1:
                nn.init.constant_(param, 0.1)  # ? change to 0.1 from 0.1
            # others using lecun-normal initialization
            else:
                nn.init.kaiming_normal_(
                    param, mode="fan_in", nonlinearity="linear"
                )

    def forward(self, x):
        out = self.head(x)
        out = self.last_classification_layer(out)
        return out


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 84,
        hidden_dim=64,
        output_dim=32,
        normalize_output: bool = True,
    ):
        """Projection Head used in SSL

        Args:
            input_dim (int, optional): [description]. Defaults to 128.
            hidden_dim (int, optional): [description]. Defaults to 64.
            output_dim (int, optional): [description]. Defaults to 32.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.normalize_output = normalize_output

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("linear", nn.Linear(self.input_dim, self.hidden_dim)),
                    ("bn", nn.BatchNorm1d(self.hidden_dim)),
                    ("selu", nn.SELU()),
                    (
                        "representation_layer",
                        nn.Linear(self.hidden_dim, self.output_dim, bias=False),
                    ),
                ]
            )
        )

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="linear"
                )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1) if self.normalize_output else x
