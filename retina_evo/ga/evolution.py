"""DEAP toolbox setup and the checkpointed evolution loop."""

import csv
import json
import pickle
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools

from retina_evo.ga.genes import make_individual_generator
from retina_evo.ga.operators import mate, mutate
from retina_evo.utils.config import GAConfig


def init_deap_types() -> None:
    """Register DEAP's ``FitnessMax`` and ``Individual`` types once (idempotent)."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)


def build_toolbox(ga_cfg: GAConfig, evaluate_fn: Callable) -> base.Toolbox:
    """Assemble the DEAP toolbox: individual factory, operators and selection."""
    init_deap_types()
    toolbox = base.Toolbox()
    generator = make_individual_generator(
        ga_cfg.control_ranges, ga_cfg.conv_ranges, ga_cfg.fc_ranges
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_fn)
    toolbox.register("mate", mate)
    toolbox.register(
        "mutate",
        mutate,
        control_ranges=ga_cfg.control_ranges,
        conv_ranges=ga_cfg.conv_ranges,
        fc_ranges=ga_cfg.fc_ranges,
    )
    toolbox.register("select", tools.selTournament, tournsize=ga_cfg.tournsize)
    return toolbox


def _fitness_key(ind) -> tuple:
    """Statistics key returning an individual's fitness tuple."""
    return ind.fitness.values


def run_evolution(
    toolbox: base.Toolbox,
    ga_cfg: GAConfig,
    checkpoint_path: Path,
    results_dir: Path,
    resume: bool = False,
) -> tuple[list, tools.Logbook, tools.HallOfFame]:
    """Run the elitist GA loop, checkpointing after every generation.

    With ``resume=True`` an existing checkpoint is loaded and evolution continues
    from the saved generation. Crossover/mutation probabilities are annealed over
    the run (less crossover, more mutation as it progresses).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if resume and checkpoint_path.exists():
        with checkpoint_path.open("rb") as fh:
            checkpoint = pickle.load(fh)
        pop = checkpoint["population"]
        logbook = checkpoint["logbook"]
        hof = checkpoint["halloffame"]
        gen = checkpoint["generation"]
        print(f"Resumed from checkpoint at generation {gen}.")
    else:
        pop = toolbox.population(n=ga_cfg.population_size)
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "best_fitness"]
        hof = tools.HallOfFame(1)
        gen = 0

    stats = tools.Statistics(key=_fitness_key)
    stats.register("best_fitness", np.max)

    while gen < ga_cfg.generations:
        gen += 1
        print(f"\n--- Generation {gen}/{ga_cfg.generations} ---")

        progress = gen / ga_cfg.generations
        current_cxpb = ga_cfg.cxpb * (1 - 0.5 * progress)
        current_mupb = ga_cfg.mupb * (1 + 0.5 * progress)

        elites = tools.selBest(pop, int(ga_cfg.elite_frac * ga_cfg.population_size))
        offspring = toolbox.select(pop, len(pop) - len(elites))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2], strict=False):
            if random.random() < current_cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < current_mupb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop = elites + offspring

        invalid = [ind for ind in pop if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)
        print(logbook.stream)

        _append_results_csv(results_dir / "GA_Results.csv", logbook[-1])

        checkpoint = {
            "population": pop,
            "logbook": logbook,
            "halloffame": hof,
            "generation": gen,
        }
        with checkpoint_path.open("wb") as fh:
            pickle.dump(checkpoint, fh)

    return pop, logbook, hof


def _append_results_csv(csv_path: Path, row: dict) -> None:
    """Append one logbook record to the GA results CSV, writing a header once."""
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_best_individual(hof: tools.HallOfFame, best_dir: Path) -> None:
    """Persist the best individual's genes, fitness and training history."""
    best = hof[0]
    best_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "control_genes": best[0],
        "conv_genes": best[1],
        "fc_genes": best[2],
        "fitness": best.fitness.values[0],
    }
    (best_dir / "best_individual.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    history = getattr(best, "history", None)
    if history:
        history_df = pd.DataFrame(history)
        history_df.insert(0, "epoch", range(1, len(history_df) + 1))
        history_df.to_csv(best_dir / "best_model_training_log.csv", index=False)


def load_best_individual(best_dir: Path) -> list:
    """Load the saved best individual as ``[control, conv_genes, fc_genes]``."""
    path = best_dir / "best_individual.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Best individual not found: {path}. Run 'main.py evolve' first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [payload["control_genes"], payload["conv_genes"], payload["fc_genes"]]
