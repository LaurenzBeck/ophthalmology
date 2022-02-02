# -*- coding: utf-8 -*-
"""## ResNet models

Implementations of the ResNet familiy from the paper "Deep Residual Learning for Image Recognition":
https://arxiv.org/abs/1512.03385v1
"""

from typing import Optional

import einops
import torch.nn as nn
from pl_bolts.models.self_supervised import resnets
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

from ophthalmology.layers import heads


class ResNetBackbone(nn.Module):
    """Resnet Backbone using the implementation from pl-bolts:
    https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/utils/self_supervised.py
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
    ):
        """constructor for the ResnetBackbone

        Args:
            name (str): one of [
                                    "ResNet",
                                    "resnet18",
                                    "resnet34",
                                    "resnet50",
                                    "resnet101",
                                    "resnet152",
                                    "resnext50_32x4d",
                                    "resnext101_32x8d",
                                    "wide_resnet50_2",
                                    "wide_resnet101_2",
                                ]
            pretrained (bool, optional): When true, use the pretrained resnet on imagenet. Defaults to False.
        """
        super().__init__()
        self.encoder = torchvision_ssl_encoder(
            name,
            pretrained,
        )

    def forward(self, x):
        # torchvision_ssl_encoder returns a list of predictions, which we need to unpack
        x = self.encoder(x)[0]

        return x


class ResNet(nn.Module):
    """Resnet using the implementation from pl-bolts:
    https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/utils/self_supervised.py
    """

    def __init__(
        self,
        name: str,
        num_output_units: int,
        num_resnet_features: int = 2048,
        pretrained: bool = False,
    ):
        """constructor for the ResnetBackbone

        Args:
            name (str): one of [
                                    "ResNet",
                                    "resnet18",
                                    "resnet34",
                                    "resnet50",
                                    "resnet101",
                                    "resnet152",
                                    "resnext50_32x4d",
                                    "resnext101_32x8d",
                                    "wide_resnet50_2",
                                    "wide_resnet101_2",
                                ]
            num_output_units (int): number of neurons in the last fc layer.
            num_resnet_features (int): number of output features from the chosen resnet model.
            pretrained (bool, optional): When true, use the pretrained resnet on imagenet. Defaults to False.
        """
        super().__init__()

        self.encoder = torchvision_ssl_encoder(
            name,
            pretrained,
        )

        self.fc = nn.Linear(num_resnet_features, num_output_units)

    def forward(self, x):
        # torchvision_ssl_encoder returns a list of predictions, which we need to unpack
        x = self.encoder(x)[0]
        x = self.fc(x)

        return x
