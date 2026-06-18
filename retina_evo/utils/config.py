"""Typed configuration loaded from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout for raw data, preprocessing output and artifacts."""

    raw_images: dict[str, Path]
    labels: dict[str, Path]
    preprocessed_dir: Path
    artifacts_dir: Path

    def preprocessed_images(self, split: str) -> Path:
        """Directory of preprocessed images for a split."""
        return self.preprocessed_dir / split

    def merged_labels(self, split: str) -> Path:
        """Binary-merged label CSV for a split."""
        return self.preprocessed_dir / "labels" / self.labels[split].name

    def checkpoint_file(self) -> Path:
        """Pickle file the evolution loop checkpoints into."""
        return self.artifacts_dir / "checkpoint" / "checkpoint.pkl"

    def ga_results_dir(self) -> Path:
        """Directory for per-generation GA result CSVs."""
        return self.artifacts_dir / "genetic_results"

    def best_model_dir(self) -> Path:
        """Directory holding the best individual and its training log."""
        return self.artifacts_dir / "best_model"


@dataclass(frozen=True)
class DatasetConfig:
    """Number of grading classes and whether to merge them to binary."""

    num_classes: int
    merge_to_binary: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    """Fundus preprocessing parameters."""

    image_size: int
    sigma_x: int
    crop_tolerance: int
    workers: int = 8


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters for training a decoded CNN."""

    batch_size: int
    lr: float
    weight_decay: float
    label_smoothing: float
    early_stopping_patience: int
    num_workers: int


@dataclass(frozen=True)
class GAConfig:
    """Genetic-algorithm settings plus the conv/fc gene sampling ranges."""

    population_size: int
    generations: int
    cxpb: float
    mupb: float
    tournsize: int
    elite_frac: float
    control_ranges: dict[str, tuple[int, ...]]
    conv_ranges: dict[str, Any]
    fc_ranges: dict[str, Any]


@dataclass(frozen=True)
class Config:
    """Top-level configuration aggregating every section."""

    seed: int
    paths: PathsConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    ga: GAConfig

    def label_csv(self, split: str) -> Path:
        """Label CSV for a split, honoring binary merging when enabled."""
        if self.dataset.merge_to_binary:
            return self.paths.merged_labels(split)
        return self.paths.labels[split]


def _as_paths(raw: dict[str, str], base: Path) -> dict[str, Path]:
    """Resolve a mapping of relative path strings against ``base``."""
    return {key: base / value for key, value in raw.items()}


def _parse_ranges(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise a GA range block: tuples for scalars, list-of-tuples for nested.

    A value is treated as per-layer pairs when its first element is itself a
    list (e.g. ``num_neurons: [[16, 32], ...]``); otherwise it is a single
    ``(low, high[, step])`` range.
    """
    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if value and isinstance(value[0], list):
            parsed[key] = [tuple(pair) for pair in value]
        else:
            parsed[key] = tuple(value)
    return parsed


def _build_paths(raw: dict[str, Any], base: Path) -> PathsConfig:
    """Construct ``PathsConfig`` from the raw ``paths`` block."""
    return PathsConfig(
        raw_images=_as_paths(raw["raw_images"], base),
        labels=_as_paths(raw["labels"], base),
        preprocessed_dir=base / raw["preprocessed_dir"],
        artifacts_dir=base / raw["artifacts_dir"],
    )


def _build_ga(raw: dict[str, Any]) -> GAConfig:
    """Construct ``GAConfig`` from the raw ``ga`` block."""
    return GAConfig(
        population_size=raw["population_size"],
        generations=raw["generations"],
        cxpb=raw["cxpb"],
        mupb=raw["mupb"],
        tournsize=raw["tournsize"],
        elite_frac=raw["elite_frac"],
        control_ranges=_parse_ranges(raw["control_ranges"]),
        conv_ranges=_parse_ranges(raw["conv_ranges"]),
        fc_ranges=_parse_ranges(raw["fc_ranges"]),
    )


def load_config(path: str | Path, base_dir: str | Path | None = None) -> Config:
    """Load and validate the project configuration.

    Relative paths in the YAML are resolved against ``base_dir`` (defaults to
    the current working directory).
    """
    path = Path(path)
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    config = Config(
        seed=raw["seed"],
        paths=_build_paths(raw["paths"], base),
        dataset=DatasetConfig(**raw["dataset"]),
        preprocessing=PreprocessingConfig(**raw["preprocessing"]),
        training=TrainingConfig(**raw["training"]),
        ga=_build_ga(raw["ga"]),
    )
    _validate(config)
    return config


def _validate(config: Config) -> None:
    """Run all cross-section validation checks."""
    _validate_dataset(config.dataset)
    _validate_paths(config.paths)
    _validate_ga(config.ga)


def _validate_dataset(dataset: DatasetConfig) -> None:
    """Ensure the class count is consistent with binary merging."""
    if dataset.num_classes < 2:
        raise ValueError("dataset.num_classes must be at least 2")
    if dataset.merge_to_binary and dataset.num_classes != 2:
        raise ValueError("dataset.merge_to_binary requires num_classes == 2")


def _validate_paths(paths: PathsConfig) -> None:
    """Ensure raw images and labels are defined for every split."""
    for split in SPLITS:
        if split not in paths.raw_images or split not in paths.labels:
            raise ValueError(f"paths must define raw_images and labels for '{split}'")


def _validate_ga(ga: GAConfig) -> None:
    """Ensure the GA ranges define every gene the pipeline relies on."""
    if not 0.0 <= ga.elite_frac < 1.0:
        raise ValueError("ga.elite_frac must be in [0, 1)")
    for key in ("conv_layers", "fc_layers", "epochs"):
        if key not in ga.control_ranges:
            raise ValueError(f"ga.control_ranges must define '{key}'")
    for key in ("conv_type", "filter_size", "num_neurons", "dropout"):
        if key not in ga.conv_ranges:
            raise ValueError(f"ga.conv_ranges must define '{key}'")
    for key in ("layer_type", "num_neurons", "dropout"):
        if key not in ga.fc_ranges:
            raise ValueError(f"ga.fc_ranges must define '{key}'")
