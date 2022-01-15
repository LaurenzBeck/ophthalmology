# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import PIL
import torch
import typer
from loguru import logger as log
from torchvision import transforms
from tqdm import tqdm


def calculate_statisticst(
    image_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Path to the folder containing the Diabetic Retinopathy Detection Images.",
    ),
    csv_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=True,
        resolve_path=True,
        help="Path to the csv file containing the Diabetic Retinopathy Detection Image names and labels.",
    ),
    shuffle: bool = typer.Option(
        False,
        "--shuffle",
        "-s",
        help="Shuffle the csv_file before iterating over it.",
    ),
    use_only: Optional[int] = typer.Option(
        None,
        "--use-only",
        "-o",
        help="If defined, use only this many samples to calculate the statistics.",
    ),
):
    """
    This scripts iterates over the filenames from csv_file, which are found in image_dir
    and calculates the channel statistics (mean and std) needed for normalization.

    The result will be printed at the end.
    """

    df = pd.read_csv(csv_file)

    if shuffle:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    to_tensor_transform = transforms.ToTensor()

    means = []
    stds = []

    for idx, image_name, level in tqdm(
        df.itertuples(), total=(len(df) if not use_only else use_only)
    ):
        try:
            image_path = os.path.join(image_dir, (image_name + ".jpeg"))
            image = PIL.Image.open(image_path)
            tensor = to_tensor_transform(image)

            means.append(tensor.mean(dim=[1, 2]))
            stds.append(tensor.std(dim=[1, 2]))
        except OSError:
            log.warning(f"found unreadable entry: {idx=},{image_name=}")

        if use_only:
            if idx == use_only:
                break

    log.info(
        f"calculation finished. \n means: {torch.stack(means).mean(dim=0)} \n stds: {torch.stack(stds).mean(dim=0)}"
    )
    print(
        f"means: {torch.stack(means).mean(dim=0)} \n stds: {torch.stack(stds).mean(dim=0)}"
    )


if __name__ == "__main__":
    typer.run(calculate_statisticst)
