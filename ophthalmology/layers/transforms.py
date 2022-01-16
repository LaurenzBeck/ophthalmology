# -*- coding: utf-8 -*-
"""Collection of data transformations and augmentations"""

from typing import List, Tuple

import torch
import torchvision
from self_supervised.vision import simclr


class SimCLRAug(torch.nn.Module):
    def __init__(
        self,
        size: int,
        rotate: bool = True,
        jitter: bool = True,
        bw: bool = True,
        blur: bool = True,
        resize_scale=(0.2, 1.0),
        resize_ratio=(0.75, 1.3333333333333333),
        rotate_deg: int = 30,
        jitter_s: float = 0.6,
        blur_s=(4, 32),
        same_on_batch: bool = False,
        flip_p: float = 0.5,
        rotate_p: float = 0.3,
        jitter_p: float = 0.3,
        bw_p: float = 0.3,
        blur_p: float = 0.3,
        stats=([0.3211, 0.2243, 0.1602], [0.2617, 0.1825, 0.1308]),
        cuda: bool = True,
    ):
        """#Wrapper for the function self_supervised.vision.simclr.get_simclr_aug_pipelines
        which internally calls:
        https://github.com/KeremTurgutlu/self_supervised/blob/af20e89b35e687d669d58e4629101ad45b6e7407/self_supervised/augmentations.py#L134

        Args:
            size (int): final quadratic patch_size for the RandomResizedCrop transform.
            rotate (bool, optional): Whether to rotate the images. Defaults to True.
            jitter (bool, optional): Whether to use the ColorJitter transform. Defaults to True.
            bw (bool, optional): Whether to use the RandomGrayscale transform. Defaults to True.
            blur (bool, optional): Whether to use the RandomGaussianBlur transform. Defaults to True.
            resize_scale (Tuple[float, float], optional): Specifies the lower and upper bounds for the random area of the crop, before resizing. The scale is defined with respect to the area of the original image. Defaults to (0.2, 1.0).
            resize_ratio ([type], optional): lower and upper bounds for the random aspect ratio of the crop, before resizing. Defaults to Tuple[float, float](0.75, 1.3333333333333333).
            rotate_deg (int, optional): Range of degrees to select from. If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees). Defaults to 30.
            jitter_s (float, optional): Apply a random transformation to the brightness, contrast, saturation with 0.8*jitter_s and hue with 0.2*jitter_s of a tensor image. Defaults to 0.6.
            blur_s (Tuple[int, int], optional): Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. If it is Tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range. Defaults to (4, 32).
            same_on_batch (bool, optional): apply the same transformation across the batch. Defaults to False.
            flip_p (float, optional): probability to perform Horizontal Flipping. Defaults to 0.5.
            rotate_p (float, optional): probability to perform rotation. Defaults to 0.3.
            jitter_p (float, optional): probability to perform jittering. Defaults to 0.3.
            bw_p (float, optional): probability to perform grayscale transformation. Defaults to 0.3.
            blur_p (float, optional): probability to perform blurring. Defaults to 0.3.
            stats ([type], optional): Channel statistics used for Normalization. First List are the channel means, second List contains the channels stds. Defaults to Tuple[List[float, float, float], List[float, float, float]]([0.3211, 0.2243, 0.1602], [0.2617, 0.1825, 0.1308]).
            cuda (bool, optional): Wether to use the GPU for the transformations. Defaults to True.
        """
        super(SimCLRAug, self).__init__()
        self.aug = torchvision.transforms.Compose(
            simclr.get_simclr_aug_pipelines(
                size,
                rotate=rotate,
                jitter=jitter,
                bw=bw,
                blur=blur,
                resize_scale=resize_scale,
                resize_ratio=resize_ratio,
                rotate_deg=rotate_deg,
                jitter_s=jitter_s,
                blur_s=blur_s,
                same_on_batch=same_on_batch,
                flip_p=flip_p,
                rotate_p=rotate_p,
                jitter_p=jitter_p,
                bw_p=bw_p,
                blur_p=blur_p,
                stats=stats,
                cuda=cuda,
            )
        )

    def forward(self, x):
        return self.aug(x)
