"""Generic training loop shared by GA head training and full-model fine-tuning."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from retina_evo.training.early_stopping import EarlyStopping
from retina_evo.utils.seed import get_device


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    loss_fn: nn.Module | None = None,
) -> dict:
    """Full evaluation pass returning loss, accuracy, labels and predictions."""
    device = device if device is not None else get_device()
    loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total = 0
    all_labels: list[int] = []
    all_preds: list[int] = []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(outputs.argmax(dim=1).cpu().tolist())

    accuracy = sum(p == t for p, t in zip(all_preds, all_labels, strict=True)) / total
    return {
        "accuracy": accuracy,
        "loss": total_loss / total,
        "labels": all_labels,
        "predictions": all_preds,
    }


class Trainer:
    """Supervised training loop with early stopping and an LR scheduler."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        device: torch.device | None = None,
        label_smoothing: float = 0.0,
        patience: int = 5,
        use_lr_scheduler: bool = True,
        clip_grad: float | None = None,
        verbose: bool = False,
    ):
        """Configure the model, optimizer, loss, scheduler and early stopping."""
        self.device = device if device is not None else get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.clip_grad = clip_grad
        self.verbose = verbose
        self.early_stopping = EarlyStopping(patience=patience)
        self.scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=2, factor=0.5
            )
            if use_lr_scheduler
            else None
        )
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train(self) -> dict[str, list[float]]:
        """Train for up to ``epochs``, returning the per-epoch metric history."""
        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._eval_epoch(self.val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.scheduler:
                self.scheduler.step(val_acc)

            if self.verbose:
                print(
                    f"[Epoch {epoch + 1}/{self.epochs}] "
                    f"Train: [Loss: {train_loss:.3f}, Acc: {train_acc:.3f}] "
                    f"Val: [Loss: {val_loss:.3f}, Acc: {val_acc:.3f}]"
                )

            if self.early_stopping(self.model, val_loss):
                break
        else:
            # Finished all epochs: end with the best weights seen so far.
            self.early_stopping.restore(self.model)

        return self.history

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch, returning ``(loss, accuracy)``."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Run one evaluation epoch, returning ``(loss, accuracy)``."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).long()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)
        return total_loss / total, correct / total

    def evaluate(self, loader: DataLoader) -> dict:
        """Full evaluation pass via ``evaluate_model`` (loss, acc, labels, preds)."""
        return evaluate_model(self.model, loader, self.device, self.loss_fn)
