# -*- coding: utf-8 -*-
"""# Visualization

This module contains functions that produce plots and visualizations
needed for logging, data exploration and the final dashboards.
"""

import itertools
from typing import List, Optional, Union

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Colormap
from torch.utils.data import Dataset


def plot_confusion_matrix(
    cm: Union[np.array, torch.tensor],
    classes: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: Union[str, Colormap] = plt.cm.Blues,
) -> plt.Figure:
    """Create a matplotlib confusion matrix plot from a np.array or torch.tensor.

    Args:
        cm (np.array | torch.tensor): Raw Confusion Matrix as np.array or torch.tensor.
        classes (Optional[List[str]], optional): If defined replace class indices on axes with class labels. Defaults to None.
        normalize (bool, optional): If True, Normalize the count of each class to 1 to see percentages instead of absolute counts. Defaults to False.
        title (str, optional): Figure Title. Defaults to "Confusion Matrix".
        cmap ([str | plt.Colormap, optional): Matplotlib colormap. Defaults to plt.cm.Blues.

    Returns:
        plt.Figure: [description]
    """
    if isinstance(cm, torch.Tensor):
        cm = cm.cpu().numpy()
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if classes:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else ".0f"  # "d" for integers
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return fig


def visualize_samples_from_dataset(
    dataset: Dataset, rows: int = 5
) -> plt.Figure:
    """Visualize a grid of samples without titles/labels in a single plot.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to visualize samples from.
        rows (int, optional): How many samples will be in one row. Total number of samples will be rows^2. Defaults to 5.

    Returns:
        plt.Figure: matplotlib figure
    """
    fig = plt.figure(figsize=(rows * 2, rows * 2))

    for idx in range(rows * rows):
        plt.subplot(rows, rows, idx + 1)
        img = dataset[np.random.randint(0, len(dataset))][0]
        plt.imshow(einops.rearrange(img.squeeze().numpy(), "c w h -> w h c"))
        plt.axis("off")
        plt.tight_layout(pad=0.0)

    return fig


def visualize_signal_propagation(
    name_values: list, title: str, *args, **kwargs
) -> plt.Figure:
    """Visualize Signal Propagation Plot using matplolib and
    the utilities from the timm package.
    See: https://github.com/mehdidc/signal_propagation_plot/blob/main/signal_propagation_plot/pytorch.py

    Args:
        name_values (torch.nn.Module): pytorch model
        input_shape (List[int], optional): Input Size of the model.

    Returns:
        plt.Figure: matplotlib figure
    """
    labels = [".".join(name.split(".")[-3:]) for name, _ in name_values]
    values = [value for _, value in name_values]
    depth = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 6))

    plt.plot(depth, values, *args, **kwargs)
    plt.xticks(depth, labels, rotation_mode="anchor")
    plt.grid()
    plt.title(title)
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment="right",
        fontsize=6,
    )

    return fig
