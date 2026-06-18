"""Dataset over fundus images with an albumentations transform."""

from pathlib import Path

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset


class FundusDataset(Dataset):
    """Fundus images loaded from disk with an optional albumentations transform."""

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: A.Compose | None = None,
    ):
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Read, RGB-convert and transform the image at ``index`` with its label."""
        image = cv2.imread(str(self.image_paths[index]))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, self.labels[index]
