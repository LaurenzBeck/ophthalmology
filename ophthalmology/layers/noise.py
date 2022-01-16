# -*- coding: utf-8 -*-
"""Collection of Modules that add noise to an input.
Mostly useful as transformations in data augmentation
"""

import numpy as np
import torch


class RandomGaussianNoise(torch.nn.Module):
    def __init__(self, var: float, apply_prob: float = 0.5):
        """Randomply add gaussian noise with a given variance

        Args:
            var (float): Upper bound on the variance that is added to the image.
                var determined by np.random.uniform(0, var)
            apply_prob (float, optional): Probability of applying this transformation. Defaults to 0.5.
        """
        super(RandomGaussianNoise, self).__init__()
        self.var = var
        self.apply_prob = apply_prob

    def forward(self, x):
        if np.random.uniform() < self.apply_prob:
            return x + torch.randn_like(x) * np.random.uniform(high=self.var)
        else:
            return x


class RandomSaltAndPepperNoise(torch.nn.Module):
    def __init__(
        self,
        treshold: float = 0.005,
        salt_value: float = 1.0,
        pepper_value: float = -1.0,
        apply_prob: float = 0.5,
    ):
        """Randomply add gaussian noise with a given variance

        Args:
            treshhold (float, optional): Probability of setting value to 0.0.
            salt_value (float, optional): Upper Value set if random > 1-treshold.
            pepper_value (float, optional): Lower Value set if random < treshold.
            apply_prob (float, optional): Probability of applying this transformation. Defaults to 0.5.
        """
        super(RandomSaltAndPepperNoise, self).__init__()
        self.treshold = treshold
        self.salt_value = salt_value
        self.pepper_value = pepper_value
        self.apply_prob = apply_prob

    def forward(self, x):
        if np.random.uniform() < self.apply_prob:
            noise = torch.rand_like(x)
            x[
                noise >= (1 - np.random.uniform(high=self.treshold))
            ] = np.random.uniform(high=self.salt_value)
            x[
                noise <= np.random.uniform(high=self.treshold)
            ] = np.random.uniform(low=self.pepper_value, high=0.0)
        return x


class RandomNullNoise(torch.nn.Module):
    def __init__(self, treshold: float = 0.005, apply_prob: float = 0.5):
        """Randomply set random values that are below the given threshold to 0.

        Args:
            treshhold (float, optional): Probability of setting value to 0.0.
            apply_prob (float, optional): Probability of applying this transformation. Defaults to 0.5.
        """
        super(RandomNullNoise, self).__init__()
        self.treshold = treshold
        self.apply_prob = apply_prob

    def forward(self, x):
        if np.random.uniform() < self.apply_prob:
            noise = torch.rand_like(x)
            x[noise <= np.random.uniform(high=self.treshold)] = 0.0
        return x


class RandomMeanOffset(torch.nn.Module):
    def __init__(self, std: float = 0.1, apply_prob: float = 0.5):
        """Randomly offset every image in the current batch by a value
        drawn from a gaussian distribution with the given std and mean=0.
        By offsetting each image in a batch sepperately and not all by the same value,
        the affect on batch statistics should be minimal.

        Args:
            std (float, optional): Standard deviation of the gaussian distribution to sample the offset from.
            apply_prob (float, optional): Probability of applying this transformation. Defaults to 0.5.
        """
        super(RandomMeanOffset, self).__init__()
        self.std = std
        self.apply_prob = apply_prob

    def forward(self, x):
        if np.random.uniform() < self.apply_prob:
            x += torch.normal(mean=0.0, std=self.std, size=(1, 1, 1))
        return x
