# -*- coding: utf-8 -*-
"""## ResNet models

Implementations of the ResNet familiy from the paper "Deep Residual Learning for Image Recognition":
https://arxiv.org/abs/1512.03385v1
"""

from typing import Optional

import einops
import torch.nn as nn
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
        x = self.encoder(x)

        return x
