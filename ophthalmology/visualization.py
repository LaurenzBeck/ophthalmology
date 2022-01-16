# -*- coding: utf-8 -*-
"""# Visualization

This module contains functions that produce plots and visualizations
needed for logging, data exploration and the final dashboards.
"""

import itertools
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Colormap


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
