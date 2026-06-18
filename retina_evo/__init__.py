"""RetinaEvo: evolve custom CNN classifiers for diabetic retinopathy grading.

A genetic algorithm (DEAP) searches over conv + fc gene encodings; each
individual is decoded into a :class:`RetinaCNN` and trained end-to-end on
preprocessed fundus images.

The primary public API of every sub-package is re-exported here for convenience
(``from retina_evo import Trainer``). Re-exports are resolved lazily (PEP 562):
``import retina_evo`` stays cheap and pulls in a sub-package's heavy
dependencies (torch, albumentations, deap) only when one of its symbols is
actually accessed. The sub-packages (:mod:`retina_evo.ga`,
:mod:`retina_evo.models`, ...) expose the full surface.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

try:
    __version__ = version("retina-evo")
except PackageNotFoundError:  # package not installed (e.g. running from a checkout)
    __version__ = "0.0.0.dev0"

# Public name -> sub-package it lives in, resolved on first attribute access.
_EXPORTS = {
    "FundusDataset": "retina_evo.data",
    "eval_transform": "retina_evo.data",
    "load_split_frame": "retina_evo.data",
    "make_image_loader": "retina_evo.data",
    "train_transform": "retina_evo.data",
    "build_toolbox": "retina_evo.ga",
    "evaluate_individual": "retina_evo.ga",
    "load_best_individual": "retina_evo.ga",
    "run_evolution": "retina_evo.ga",
    "save_best_individual": "retina_evo.ga",
    "RetinaCNN": "retina_evo.models",
    "build_network_from_individual": "retina_evo.models",
    "merge_labels": "retina_evo.preprocessing",
    "preprocess_image": "retina_evo.preprocessing",
    "process_directory": "retina_evo.preprocessing",
    "EarlyStopping": "retina_evo.training",
    "Trainer": "retina_evo.training",
    "evaluate_model": "retina_evo.training",
    "SPLITS": "retina_evo.utils",
    "Config": "retina_evo.utils",
    "get_device": "retina_evo.utils",
    "load_config": "retina_evo.utils",
    "set_seed": "retina_evo.utils",
}

__all__ = [
    "SPLITS",
    "Config",
    "EarlyStopping",
    "FundusDataset",
    "RetinaCNN",
    "Trainer",
    "__version__",
    "build_network_from_individual",
    "build_toolbox",
    "eval_transform",
    "evaluate_individual",
    "evaluate_model",
    "get_device",
    "load_best_individual",
    "load_config",
    "load_split_frame",
    "make_image_loader",
    "merge_labels",
    "preprocess_image",
    "process_directory",
    "run_evolution",
    "save_best_individual",
    "set_seed",
    "train_transform",
]


def __getattr__(name: str) -> object:
    """Lazily import and cache a re-exported symbol (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # cache so subsequent lookups skip the import
    return value


def __dir__() -> list[str]:
    """Include the lazily re-exported names in ``dir(retina_evo)``."""
    return sorted([*globals(), *_EXPORTS])


if TYPE_CHECKING:  # let type checkers and IDEs see the re-exported names
    from retina_evo.data import (
        FundusDataset,
        eval_transform,
        load_split_frame,
        make_image_loader,
        train_transform,
    )
    from retina_evo.ga import (
        build_toolbox,
        evaluate_individual,
        load_best_individual,
        run_evolution,
        save_best_individual,
    )
    from retina_evo.models import RetinaCNN, build_network_from_individual
    from retina_evo.preprocessing import (
        merge_labels,
        preprocess_image,
        process_directory,
    )
    from retina_evo.training import EarlyStopping, Trainer, evaluate_model
    from retina_evo.utils import SPLITS, Config, get_device, load_config, set_seed
