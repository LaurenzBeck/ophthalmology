# -*- coding: utf-8 -*-
"""## Residual Blocks"""

import nfnets
import torch
import torch.nn as nn
import torch.nn.functional as F

from ophthalmology.layers import activations, involution, squeeze_and_excitation


class BasicResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
    ):
        """Basic Module consisting of 2 consecutive Conv + BN + ReLU Layers.

        Args:
            in_channels (int, optional): number of input feature maps or channels.
        """
        super(BasicResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out


class SEResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
    ):
        """Module consisting of 2 consecutive Conv + BN + ReLU Layers followed by a SELayer.

        Args:
            in_channels (int, optional): number of input feature maps or channels.
        """
        super(SEResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            squeeze_and_excitation.SELayer(in_channels),
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out


class NFSEResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        alpha: float = 0.2,
        beta: float = 0.9165,
        involution_layer: bool = False,
    ):
        """Module consisting of 2 consecutive Conv + BN + ReLU Layers followed by a SELayer.

        Args:
            in_channels (int, optional): number of input feature maps or channels.
        """
        super(NFSEResidualBlock, self).__init__()
        self.alpha = torch.nn.parameter.Parameter(
            torch.Tensor([alpha]), requires_grad=True
        )
        self.beta = beta

        self.block = nn.Sequential(
            self.conv2d(in_channels, in_channels // 2, kernel_size=1),
            activations.SELU(),
            nn.BatchNorm2d(in_channels // 2),
            self.conv2d(
                in_channels // 2, in_channels // 2, kernel_size=3, padding=1
            ),
            activations.SELU(),
            involution.Involution2d(
                in_channels // 2, in_channels // 2, kernel_size=3, padding=1
            )
            if involution_layer
            else self.conv2d(
                in_channels // 2, in_channels // 2, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(in_channels // 2),
            activations.SELU(),
            self.conv2d(in_channels // 2, in_channels, kernel_size=1),
            activations.SELU(),
            squeeze_and_excitation.SELayer(in_channels),
            nn.BatchNorm2d(in_channels),
        )

        for module in self.block.modules():
            if isinstance(module, nfnets.ScaledStdConv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="linear"
                )

    def forward(self, x):
        skip = x
        out = self.block(x / self.beta)
        return out * self.alpha + skip

    @staticmethod
    def conv2d(*args, **kwargs):
        return nfnets.ScaledStdConv2d(*args, **kwargs)
        # return nn.Conv2d(*args, **kwargs)
        # return nfnets.WSConv2d(*args, **kwargs)
