# -*- coding: utf-8 -*-
from typing import Optional

import einops
import torch.nn as nn

from ophthalmology.layers import convolutional_stems, heads, residual_blocks


class AttentiveNFResNet(nn.Module):
    """Resnet with a lot of additional layers and tricks including:
    * Atrous Spatial Pyramid Pooling
    * Squeeze and Excitation Layers
    * Self Normalizing elements
    * Ideas from the Normalizer Free Resnets paper
    * Involution Layers
    * Learnable output gains in each residual block"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        downscale: Optional[
            convolutional_stems.Downscale
        ] = convolutional_stems.Downscale.QUARTER,
        num_res_blocks: int = 4,
        num_classes: int = 5,
        dropout: Optional[float] = None,
        nf: bool = False,
        involution: bool = False,
        alpha: float = 0.55,
        beta: float = 1.14,
    ):
        super().__init__()
        self.stem = (
            convolutional_stems.NFASSPConvolutionalStem(
                in_channels, out_channels, out_channels, downscale
            )
            if nf
            else convolutional_stems.ASSPConvolutionalStem(
                in_channels, out_channels, out_channels, downscale
            )
        )
        # residual_blocks
        self.rsb = nn.Sequential(
            *[
                (
                    residual_blocks.NFSEResidualBlock(
                        out_channels,
                        involution_layer=involution,
                        alpha=alpha,
                        beta=beta,
                    )
                    if nf
                    else residual_blocks.SEResidualBlock(out_channels)
                )
                for _ in range(num_res_blocks)
            ],
        )
        self.head = heads.MultilayerClassificationHead(
            out_channels, [32, 16], num_classes, dropout
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.rsb(x)
        x = einops.reduce(x, "b c h w -> b c", reduction="mean")
        x = self.head(x)
        return x


class AttentiveNFResNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 80,
        downscale: Optional[
            convolutional_stems.Downscale
        ] = convolutional_stems.Downscale.QUARTER,
        num_res_blocks: int = 8,
        num_features: int = 64,
        involution: bool = True,
        alpha: float = 0.55,
        beta: float = 1.14,
    ):
        super().__init__()
        self.stem = convolutional_stems.NFASSPConvolutionalStem(
            in_channels, out_channels, out_channels, downscale
        )
        # residual_blocks
        self.rsb = nn.Sequential(
            *[
                residual_blocks.NFSEResidualBlock(
                    out_channels,
                    involution_layer=involution,
                    alpha=alpha,
                    beta=beta,
                )
                for _ in range(num_res_blocks)
            ],
        )
        self.head = nn.Sequential(
            nn.Linear(out_channels, num_features),
            nn.SELU(),
            nn.BatchNorm1d(num_features),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.rsb(x)
        x = einops.reduce(x, "b c h w -> b c", reduction="mean")
        x = self.head(x)
        return x
