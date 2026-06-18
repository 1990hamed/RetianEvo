"""Albumentations pipelines for training and evaluation."""

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(
                height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CLAHE(clip_limit=3, tile_grid_size=(8, 8)),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.3, 1)),
            A.ISONoise(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def eval_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(
                height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
