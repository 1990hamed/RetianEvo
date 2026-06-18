"""GA fitness: train a decoded CNN on fundus images and score it.

Each individual is decoded into a full ``RetinaCNN`` and trained end-to-end on
the (preprocessed) image loaders. Its fitness is the best validation accuracy
reached within the gene-encoded epoch budget.
"""

import torch
from torch.utils.data import DataLoader

from retina_evo.models.classifier import build_network_from_individual
from retina_evo.training.trainer import Trainer
from retina_evo.utils.config import TrainingConfig


def evaluate_individual(
    individual: list,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_cfg: TrainingConfig,
    device: torch.device | None = None,
) -> tuple[float]:
    """Fitness = best validation accuracy of the CNN encoded by ``individual``.

    The epoch budget is the third control gene (``[conv_layers, fc_layers,
    epochs]``). The trained history is stashed on the individual so the
    evolution loop can persist the best model's training curve.
    """
    model = build_network_from_individual(individual, num_classes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.lr,
        weight_decay=training_cfg.weight_decay,
    )
    epochs = individual[0][2]
    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
        label_smoothing=training_cfg.label_smoothing,
        patience=training_cfg.early_stopping_patience,
    )
    history = trainer.train()
    individual.history = dict(history)
    return (max(history["val_acc"], default=0.0),)
