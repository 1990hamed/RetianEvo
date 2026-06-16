# RetinaEvo

RetinaEvo discovers custom CNN architectures for **diabetic retinopathy grading**
from fundus images using a **Genetic Algorithm**. Instead of hand-designing a
network, a population of candidate architectures is evolved with
[DEAP](https://deap.readthedocs.io/): each individual encodes a convolutional
stack and a fully-connected head as genes, is decoded into a full PyTorch
network, trained end-to-end on preprocessed images, and scored by its validation
accuracy. The best architecture found is then retrained and evaluated on a held-out
test split.

## How it works

An individual is `[control_genes, conv_genes, fc_genes]`:

- **control_genes** — `[conv_layers, fc_layers, epochs]`: how many conv and FC
  layers to use and the per-individual training budget.
- **conv_genes** — one `[conv_type, filter_size, num_neurons, dropout]` quad per
  conv layer. `conv_type` selects the block: `1` = `Conv→BN→ReLU`, `2` adds
  `MaxPool`, `3` also adds `Dropout`.
- **fc_genes** — one `[layer_type, num_neurons, dropout]` triple per FC layer.
  `layer_type` `1` = `Linear→ReLU→BN`, `2` also adds `Dropout`.

A global `AdaptiveAvgPool2d` between the conv stack and the head fixes the head's
input width, so any evolved conv configuration decodes into a valid network. The
GA loop ([evolution.py](retina_evo/ga/evolution.py)) is elitist with tournament
selection, anneals crossover/mutation probabilities over the run, and checkpoints
after every generation so it can be resumed.

The gene sampling ranges, GA hyper-parameters, preprocessing, and training
settings are all driven by [config/config.yaml](config/config.yaml).

## Project layout

```
retina_evo/
├── data/            # dataset, transforms, and DataLoader construction
├── preprocessing/   # fundus circle-crop, Ben-Graham sharpening, label merging
├── ga/              # gene generation, operators, fitness, evolution loop
├── models/          # decode an individual into a RetinaCNN
├── training/        # Trainer + early stopping
└── utils/           # typed config loader, seeding/device helpers
config/config.yaml   # all pipeline settings
main.py              # CLI entrypoint (preprocess / evolve / evaluate)
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management
(Python 3.13+).

```bash
uv sync --all-groups
```

## Data

Place raw fundus images and label CSVs as configured under `paths` in
[config/config.yaml](config/config.yaml):

```
data/
├── train_images/   val_images/   test_images/   # raw .png fundus images
└── train_1.csv     valid.csv      test.csv       # CSVs with a "diagnosis" column
```

Labels use the standard 5-grade diabetic retinopathy scale (0–4). With
`dataset.merge_to_binary: true` (the default), grades are collapsed to a
referable-DR binary task (`0` stays `0`, `1–4` become `1`). Set
`num_classes: 5` and `merge_to_binary: false` for full grading.

## Usage

The pipeline runs in three stages, all reading `config/config.yaml` (override
with `--config`):

```bash
# 1. Circle-crop / sharpen / resize images and merge labels
uv run main.py preprocess

# 2. Run the genetic architecture search (--resume continues from checkpoint)
uv run main.py evolve
uv run main.py evolve --resume

# 3. Retrain the best evolved CNN and report test accuracy
uv run main.py evaluate
```

Preprocessing skips already-processed images, and `evolve` checkpoints every
generation, so interrupted runs of either stage can be resumed.

### Artifacts

Outputs are written under `paths.artifacts_dir`:

- `checkpoint/checkpoint.pkl` — resumable GA state
- `genetic_results/GA_Results.csv` — best fitness per generation
- `best_model/best_individual.json` — winning genes and fitness
- `best_model/best_model_training_log.csv` — training curve of the best model
- `best_model/test_metrics.json` — final test accuracy and loss

## Development

```bash
# Run tests
uv run pytest
uv run pytest tests/test_genes.py   # a single file

# Lint and format
uv run ruff check .
uv run ruff format .
uv run ruff check --fix .
```

Code style: Ruff with line length 88, targeting Python 3.13; double quotes and
space indentation. Commits follow
[Conventional Commits](https://www.conventionalcommits.org/)
(`type(scope): description`). Before pushing, `uv run ruff check .` and
`uv run pytest` must both pass.
