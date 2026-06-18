"""Training loop, evaluation and early stopping."""

from retina_evo.training.early_stopping import EarlyStopping
from retina_evo.training.trainer import Trainer, evaluate_model

__all__ = ["EarlyStopping", "Trainer", "evaluate_model"]
