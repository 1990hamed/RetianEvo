"""Dataframe and image DataLoader factories."""

from pathlib import Path

import albumentations as A
import pandas as pd
import torch
from torch.utils.data import DataLoader

from retina_evo.data.datasets import FundusDataset


def load_split_frame(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    """Read a label CSV and attach the image path for each id_code."""
    df = pd.read_csv(csv_path)
    df["image_path"] = df["id_code"].map(lambda x: image_dir / f"{x}.png")
    return df


def make_image_loader(
    df: pd.DataFrame,
    transform: A.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader of transformed fundus images from a split dataframe.

    Enable ``drop_last`` for training so a final batch of size one cannot break
    the head's ``BatchNorm1d`` layers.
    """
    dataset = FundusDataset(
        image_paths=df["image_path"].tolist(),
        labels=df["diagnosis"].tolist(),
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
