"""RetinaEvo command-line entrypoint.

Three pipeline stages are exposed as subcommands::

    uv run main.py preprocess   # circle-crop/sharpen/resize images + merge labels
    uv run main.py evolve       # genetic search for a CNN architecture
    uv run main.py evaluate     # train the best individual and test it

All stages read ``config/config.yaml`` (override with ``--config``).
"""

import argparse
import json
from functools import partial

import torch
from torch.utils.data import DataLoader

from retina_evo.data.loaders import load_split_frame, make_image_loader
from retina_evo.data.transforms import eval_transform, train_transform
from retina_evo.ga.evolution import (
    build_toolbox,
    load_best_individual,
    run_evolution,
    save_best_individual,
)
from retina_evo.ga.fitness import evaluate_individual
from retina_evo.models.classifier import build_network_from_individual
from retina_evo.preprocessing.pipeline import merge_labels, process_directory
from retina_evo.training.trainer import Trainer, evaluate_model
from retina_evo.utils.config import SPLITS, Config, load_config
from retina_evo.utils.seed import get_device, set_seed


def build_image_loader(config: Config, split: str, *, train: bool) -> DataLoader:
    """Build an image DataLoader for a split (the train split is shuffled/augmented)."""
    frame = load_split_frame(
        config.label_csv(split), config.paths.preprocessed_images(split)
    )
    image_size = config.preprocessing.image_size
    transform = train_transform(image_size) if train else eval_transform(image_size)
    return make_image_loader(
        frame,
        transform,
        batch_size=config.training.batch_size,
        shuffle=train,
        num_workers=config.training.num_workers,
        drop_last=train,
    )


def run_preprocess(config: Config) -> None:
    """Preprocess every split's images and write binary-merged label CSVs."""
    prep = config.preprocessing
    for split in SPLITS:
        process_directory(
            config.paths.raw_images[split],
            config.paths.preprocessed_images(split),
            image_size=prep.image_size,
            sigma_x=prep.sigma_x,
            tolerance=prep.crop_tolerance,
            workers=prep.workers,
        )
        if config.dataset.merge_to_binary:
            merge_labels(config.paths.labels[split], config.paths.merged_labels(split))


def run_evolve(config: Config, resume: bool = False) -> None:
    """Evolve CNN architectures and persist the best individual found."""
    device = get_device()
    train_loader = build_image_loader(config, "train", train=True)
    val_loader = build_image_loader(config, "val", train=False)

    evaluate_fn = partial(
        evaluate_individual,
        num_classes=config.dataset.num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=config.training,
        device=device,
    )
    toolbox = build_toolbox(config.ga, evaluate_fn)
    _, _, hof = run_evolution(
        toolbox,
        config.ga,
        checkpoint_path=config.paths.checkpoint_file(),
        results_dir=config.paths.ga_results_dir(),
        resume=resume,
    )
    save_best_individual(hof, config.paths.best_model_dir())
    print(f"Best fitness (val accuracy): {hof[0].fitness.values[0]:.4f}")


def run_evaluate(config: Config) -> None:
    """Train the best evolved CNN on the train split and report test accuracy."""
    device = get_device()
    individual = load_best_individual(config.paths.best_model_dir())
    model = build_network_from_individual(individual, config.dataset.num_classes)

    train_loader = build_image_loader(config, "train", train=True)
    val_loader = build_image_loader(config, "val", train=False)
    test_loader = build_image_loader(config, "test", train=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    trainer = Trainer(
        model,
        optimizer,
        train_loader,
        val_loader,
        epochs=individual[0][2],
        device=device,
        label_smoothing=config.training.label_smoothing,
        patience=config.training.early_stopping_patience,
        verbose=True,
    )
    trainer.train()

    metrics = evaluate_model(model, test_loader, device)
    summary = {"test_accuracy": metrics["accuracy"], "test_loss": metrics["loss"]}
    best_dir = config.paths.best_model_dir()
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Test accuracy: {summary['test_accuracy']:.4f} | "
        f"loss: {summary['test_loss']:.4f}"
    )


def main() -> None:
    """Parse CLI arguments and dispatch to the requested pipeline stage."""
    parser = argparse.ArgumentParser(
        description="RetinaEvo: preprocess, evolve, evaluate."
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to the YAML config."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preprocess", help="Preprocess images and merge labels.")
    evolve_parser = subparsers.add_parser(
        "evolve", help="Run the genetic architecture search."
    )
    evolve_parser.add_argument(
        "--resume", action="store_true", help="Resume from the last checkpoint."
    )
    subparsers.add_parser("evaluate", help="Train and test the best evolved CNN.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.command == "preprocess":
        run_preprocess(config)
    elif args.command == "evolve":
        run_evolve(config, resume=args.resume)
    elif args.command == "evaluate":
        run_evaluate(config)


if __name__ == "__main__":
    main()
