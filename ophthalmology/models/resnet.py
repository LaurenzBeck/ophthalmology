# -*- coding: utf-8 -*-
"""Wrapper class around lightning-bolts resnet implementation"""

import pl_bolts


class ResNet:
    def __init__(self, name: str, pretrained: bool = False):
        """Wrapper class around https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/utils/self_supervised.py

        This is intended to allow the use of hydra.utils.instanciate instead of hydra.utils.call
        The main reason for this strange wrapper is to stay consistent with the class based models.

        Args:
            name (str): [description] one of:
            [
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
            pretrained (bool, optional): If true, load pre-trained imagenet weights. Defaults to False.

        Returns:
            torch.nn.Module: ResNet variant defined by name
        """
        return pl_bolts.utils.self_supervised.torchvision_ssl_encoder(
            name, pretrained
        )
