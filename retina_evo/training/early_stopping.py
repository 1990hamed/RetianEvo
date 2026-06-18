"""Early stopping with optional best-weights restoration."""

import copy

from torch import nn


class EarlyStopping:
    """Stop training when validation loss stalls for ``patience`` epochs."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """Configure patience, the improvement threshold and weight restoration."""
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_state: dict | None = None
        self.best_loss: float | None = None
        self.counter = 0

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
        """Track validation loss; returns True when training should stop."""
        if self.best_loss is None or self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore(model)
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        """Load the best observed weights back into ``model`` (if any were saved)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
