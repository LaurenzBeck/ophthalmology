# -*- coding: utf-8 -*-
"""Atrous Spatial Pyramid Pooling Layer

paper: Rethinking Atrous Convolution for Semantic Image Segmentation
https://arxiv.org/abs/1706.05587

adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py
"""

from typing import Optional

import nfnets
import torch
from torch import nn

from ophthalmology.layers import activations


class ASSPLayer(nn.Module):
    """Atrous Spatial Pyramid Pooling Layer"""

    def __init__(
        self,
        in_channels: int,
        pyramid_stage_channels: int = 8,
        out_channels: Optional[int] = None,
    ):
        """Construct an Atrous Spatial Pyramid Pooling Layer

        Args:
            in_channels (int): number of input channels
            pyramid_stage_channels (int, optional): each stage in the pyramid has this many filters/channels.
                With 3 stages, the hidden channel size before the final 1x1 convolution is 3*pyramid_stage_channels. Defaults to 8.
            out_channels (Optional[int], optional): Number of output channels. If set to None, then out_channels=in_channels. Defaults to None.
        """
        super(ASSPLayer, self).__init__()

        # every layer should not change the width/height dimensions.
        self.pyramid_conv_layers = nn.ModuleList(
            [
                self.conv2d(
                    in_channels, pyramid_stage_channels, 1
                ),  # 1x1 conv x
                self.conv2d(
                    in_channels, pyramid_stage_channels, 3, padding=1
                ),  # 3x3 conv xxx
                self.conv2d(
                    in_channels,
                    pyramid_stage_channels,
                    3,
                    padding=2,
                    dilation=2,
                ),  # 3x3 conv x.x.x
            ]
        )
        for pyramid_conv_layer in self.pyramid_conv_layers:
            nn.init.kaiming_normal_(
                pyramid_conv_layer.weight, mode="fan_in", nonlinearity="linear"
            )

        self.bn = nn.BatchNorm2d(
            len(self.pyramid_conv_layers) * pyramid_stage_channels
        )
        self.activation = activation.SELU()

        self.reduction_conv = self.conv2d(
            len(self.pyramid_conv_layers) * pyramid_stage_channels,
            (out_channels if out_channels is not None else in_channels),
            1,
        )  # 1x1 conv x

        nn.init.kaiming_normal_(
            self.reduction_conv.weight, mode="fan_in", nonlinearity="linear"
        )

    @staticmethod
    def conv2d(*args, **kwargs):
        return nfnets.ScaledStdConv2d(*args, **kwargs)
        # return nn.Conv2d(*args, **kwargs)
        # return nfnets.WSConv2d(*args, **kwargs)

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.pyramid_conv_layers], dim=1)
        out = self.bn(out)
        out = self.activation(out)
        out = self.reduction_conv(out)
        out = self.activation(out)

        return out
