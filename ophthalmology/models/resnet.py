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
        super().__init__()
        self.encoder = torchvision_ssl_encoder(
            name,
            pretrained,
        )

    def forward(self, x):
        x = self.encoder(x)

        return x
