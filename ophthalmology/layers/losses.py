# -*- coding: utf-8 -*-
"""Collection of Loss functions"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Single Label Focal Loss

    dapted from:
    https://github.com/Jianf-Wang/RSG/blob/main/Imbalanced_Classification/losses.py"""

    def __init__(self, weight=None, gamma=0.0, reduction="mean"):
        """FocalLoss constructor

        Args:
            weight (Tensor, optional): a manual rescaling weight given to each class. If given, has to be a Tensor of size C.
            gamma (float, optional): positive focal loss hyperparameter. Defaults to 0.0 (Cross Entropy).
            reduction (str, optional): pytorch style loss reduction strategy. Defaults to "mean".
        """
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def focal_loss(self, input_values, gamma):
        """Computes the focal loss"""
        p = torch.exp(-input_values)
        loss = (1 - p) ** gamma * input_values

        if self.reduction == "mean":
            loss = loss.mean()

        return loss

    def forward(self, input, target):
        ce = F.cross_entropy(
            input, target, reduction="none", weight=self.weight
        )
        return self.focal_loss(ce, self.gamma)


class NLLLoss(nn.Module):
    """Negative Log-Likelihood loss

    Code from:
    https://github.com/leehomyc/mixup_pytorch/blob/master/main_cifar10.py
    """

    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, input, targets):
        log_probs = F.log_softmax(input, dim=1)
        loss = -log_probs * targets
        loss = torch.sum(loss) / input.size(0)
        return loss


class SoftmaxMSELoss(nn.Module):
    """Loss combining both Softmax and MSELoss"""

    def __init__(self):
        super(SoftmaxMSELoss, self).__init__()

    def forward(self, input, targets):
        soft_probs = F.softmax(input, dim=1)
        loss = F.mse_loss(soft_probs, targets)

        return loss


class MixupLoss(nn.Module):
    """Wraps a criterion/loss and performs convex combination
    of the individual loss terms.
    """

    def __init__(self, criterion: nn.Module):
        super(MixupLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred, targets):
        y_a, y_b, lam = targets
        loss = lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(
            pred, y_b
        )
        return loss.mean()


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, eps: float = 1e-6):
        """Normalized Temperature-scaled Cross Entropy Loss adopted from:
        https://github.com/PyTorchLightning/lightning-bolts/blob/75ddb4b96d069378a4c38e75460c8c7eef9f15b7/pl_bolts/models/self_supervised/simclr/simclr_module.py#L223

        Args:
            temperature (float): [description]
            eps (float, optional): [description]. Defaults to 1e-6.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, out_1, out_2):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """

        # out: [2 * batch_size, dim]
        out = torch.cat([out_1, out_2], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = (
            torch.Tensor(neg.shape)
            .fill_(math.e ** (1 / self.temperature))
            .to(neg.device)
        )
        neg = torch.clamp(
            neg - row_sub, min=self.eps
        )  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + self.eps)).mean()

        return loss
