# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pandas as pd
import PIL
import typer
from loguru import logger as log
from torchvision import transforms
from tqdm import tqdm


def clean_dataset(
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
):
    """
    During the download and decompression of the >80GB dataset, some images might miss some parts.
    This cli tool reads every image and creates a new csv file with only the entries that can be decoded correctly.

    The result will be saved as "<csv_file>_cleaned.csv"
    """

    df = pd.read_csv(csv_file)
    df_cleaned = pd.DataFrame(columns=["image", "level"], index=None)

    to_tensor_transform = transforms.ToTensor()

    for idx, image_name, level in tqdm(df.itertuples()):
        try:
            image_path = os.path.join(image_dir, (image_name + ".jpeg"))
            image = PIL.Image.open(image_path)
            tensor = to_tensor_transform(image)

            df_cleaned.loc[len(df_cleaned)] = (image_name, level)
        except OSError:
            log.warning(f"found unreadable entry: {idx=},{image_name=}")

    cleaned_csv_path = str(csv_file)[:-4] + "_cleaned.csv"
    log.info(
        f"finished the cleaning process.\nwriting the results to: {cleaned_csv_path}"
    )

    df_cleaned.to_csv(cleaned_csv_path, index=False)


if __name__ == "__main__":
    typer.run(clean_dataset)
