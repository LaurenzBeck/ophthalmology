# -*- coding: utf-8 -*-
"""Convolutional Stems"""

from enum import Enum
from typing import Optional

import nfnets
import torch.nn as nn
import torch.nn.functional as F

from ophthalmology.layers import activations, assp, squeeze_and_excitation


class Downscale(str, Enum):
    """Enumeration for implemented downscaling factors for convolutional layers.
    This can either be implemented by pooling layers or using striding.
    """

    HALF = "2"
    QUARTER = "4"


class BasicConvolutionalStem(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        downscale: Optional[Downscale] = None,
    ):
        """Basic Module consisting of 3 consecutive Conv + BN + ReLU Layers
        with optional striding determined by the downscale option.

        Args:
            in_channels (int): number of input feature maps or channels.
            out_channels (int): number of output feature maps or channels.
            downscale (Optional[Downscale]): Wether to use one of the supported
                downscaling factors by using strided convolutions. Defaults to None.
        """
        super(BasicConvolutionalStem, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2 if downscale == Downscale.QUARTER else 1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2 if downscale in (Downscale.QUARTER, Downscale.HALF) else 1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        return out


class ASSPConvolutionalStem(nn.Sequential):
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        out_channels: int = 8,
        downscale: Optional[Downscale] = None,
    ):
        """Basic Module consisting of 3 consecutive Conv + BN + ReLU Layers
        with optional striding determined by the downscale option.

        Args:
            in_channels (int): number of input feature maps or channels.
            out_channels (int): number of output feature maps or channels.
            downscale (Optional[Downscale]): Wether to use one of the supported
                downscaling factors by using strided convolutions. Defaults to None.
        """
        super(ASSPConvolutionalStem, self).__init__(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=5,
                stride=2 if downscale == Downscale.QUARTER else 1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=2
                if downscale in (Downscale.QUARTER, Downscale.HALF)
                else 1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            assp.ASSPLayer(
                hidden_channels,
                pyramid_stage_channels=hidden_channels,
                out_channels=out_channels,
            ),
        )


class NFASSPConvolutionalStem(nn.Sequential):
    @staticmethod
    def conv2d(*args, **kwargs):
        return nfnets.ScaledStdConv2d(*args, **kwargs)
        # return nn.Conv2d(*args, **kwargs)
        # return nfnets.WSConv2d(*args, **kwargs)

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 16,
        out_channels: int = 8,
        downscale: Optional[Downscale] = None,
    ):
        """Basic Module consisting of 3 consecutive Conv + BN + ReLU Layers
        with optional striding determined by the downscale option.

        Args:
            in_channels (int): number of input feature maps or channels.
            out_channels (int): number of output feature maps or channels.
            downscale (Optional[Downscale]): Wether to use one of the supported
                downscaling factors by using strided convolutions. Defaults to None.
        """
        super(NFASSPConvolutionalStem, self).__init__(
            self.conv2d(
                in_channels,
                hidden_channels // 2,
                kernel_size=5,
                stride=2 if downscale == Downscale.QUARTER else 1,
                padding=2,
            ),
            nn.BatchNorm2d(hidden_channels // 2),  #!no nf anymore
            activations.SELU(),
            self.conv2d(
                hidden_channels // 2,
                hidden_channels,
                kernel_size=3,
                stride=2
                if downscale in (Downscale.QUARTER, Downscale.HALF)
                else 1,
                padding=1,
            ),
            activations.SELU(),
            assp.NFASSPLayer(
                hidden_channels,
                pyramid_stage_channels=hidden_channels,
                out_channels=out_channels,
            ),
            squeeze_and_excitation.SELayer(out_channels),
        )

        for module in super().modules():
            if isinstance(module, nfnets.ScaledStdConv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="linear"
                )
