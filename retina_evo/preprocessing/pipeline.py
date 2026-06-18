"""Batch preprocessing of image directories and label CSVs."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from retina_evo.preprocessing.image_ops import preprocess_image


def _process_one(
    image_path: Path,
    output_dir: Path,
    image_size: int,
    sigma_x: int,
    tolerance: int,
) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        tqdm.write(f"Warning: failed to read {image_path}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image, image_size, sigma_x, tolerance)
    cv2.imwrite(
        str(output_dir / image_path.name),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )
    return True


def process_directory(
    input_dir: Path,
    output_dir: Path,
    image_size: int,
    sigma_x: int,
    tolerance: int = 7,
    workers: int = 8,
) -> int:
    """Preprocess every PNG in ``input_dir`` into ``output_dir``.

    Already-processed images are skipped, so an interrupted run can resume.
    Returns the number of images written this run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(input_dir.glob("*.png"))
    pending = [p for p in image_paths if not (output_dir / p.name).exists()]
    skipped = len(image_paths) - len(pending)
    if skipped:
        print(f"{input_dir.name}: skipping {skipped} already-processed images")

    if not pending:
        return 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            tqdm(
                pool.map(
                    lambda p: _process_one(
                        p, output_dir, image_size, sigma_x, tolerance
                    ),
                    pending,
                ),
                total=len(pending),
                desc=f"Processing {input_dir.name}",
            )
        )
    return sum(results)


def merge_labels(csv_path: Path, output_path: Path) -> pd.DataFrame:
    """Collapse the 5-grade diagnosis into binary: 0 stays 0, 1-4 become 1."""
    df = pd.read_csv(csv_path)
    df["diagnosis"] = (df["diagnosis"] != 0).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
