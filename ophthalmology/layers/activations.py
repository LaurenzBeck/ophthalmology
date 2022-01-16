# -*- coding: utf-8 -*-
"""## Activation Functions"""

import torch


class SELU(torch.nn.Module):
    """Gamma scaled ELU activation (from NFNets)"""

    def __init__(self):
        super(SELU, self).__init__()

    def forward(self, x):
        return torch.nn.functional.selu(x) * 1.0008515119552612
