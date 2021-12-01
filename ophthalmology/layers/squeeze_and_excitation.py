# -*- coding: utf-8 -*-
"""Squeeze and Excitation Layer adapted from : """

import einops
import snoop
import torch.nn as nn


class SELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio

        self.excite = nn.Sequential(
            nn.Linear(num_channels, num_channels_reduced),
            nn.SELU(),
            nn.Linear(num_channels_reduced, num_channels),
            nn.Sigmoid(),
        )

        for module in self.excite.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="linear"
                )

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, _, _ = input_tensor.size()

        features = einops.reduce(
            input_tensor, "b c h w -> b c", reduction="mean"
        )

        # We have to multiply by two here to preserve signal variance.
        weights = 2 * self.excite(features)

        output_tensor = input_tensor * weights.view(
            batch_size, num_channels, 1, 1
        )

        return output_tensor
