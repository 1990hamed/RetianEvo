"""Crossover and mutation operators for the hierarchical (conv + fc) GA.

Individuals are ``[control, conv_genes, fc_genes]`` where ``control`` is
``[conv_layers, fc_layers, epochs]``. All operators modify individuals in place,
as DEAP's variation loop expects. Conv layers are kept ordered by ascending
channel count (shallow-to-deep) and FC layers by descending width (a funnel
head).
"""

import copy
import random
from typing import Any

from retina_evo.ga.genes import iter_layer_genes

CONV_NEURON_INDEX = 2  # index of num_neurons within a conv gene
FC_NEURON_INDEX = 1  # index of num_neurons within an fc gene
LAYER_COUNT_GENES = ("conv_layers", "fc_layers")


def recombine_layer_pool(
    parent_a: list[list[int]],
    parent_b: list[list[int]],
    target_count: int,
    neuron_index: int,
    ascending: bool,
) -> list[list[int]]:
    """Select ``target_count`` layers from both parents' pooled layers.

    Distinct parent layers are pooled and drawn largest-first (weighted random
    without replacement); any shortfall is filled by cloning already-selected
    layers. The result is ordered by width via ``neuron_index`` (ascending for
    conv stacks, descending for fc heads).
    """
    pool: list[list[int]] = []
    for layer in parent_a + parent_b:
        if layer not in pool:
            pool.append(copy.deepcopy(layer))
    pool.sort(key=lambda layer: layer[neuron_index], reverse=True)

    selected: list[list[int]] = []
    while len(selected) < target_count and pool:
        weights = [1 / (i + 1) for i in range(len(pool))]
        chosen = random.choices(pool, weights=weights, k=1)[0]
        selected.append(chosen)
        pool.remove(chosen)

    while len(selected) < target_count and selected:
        selected.append(copy.deepcopy(random.choice(selected)))

    selected.sort(key=lambda layer: layer[neuron_index], reverse=not ascending)
    return selected


def mate(ind1: list, ind2: list) -> tuple[list, list]:
    """Swap control genes, then recombine conv and fc layers from both parents."""
    parent_conv = (copy.deepcopy(ind1[1]), copy.deepcopy(ind2[1]))
    parent_fc = (copy.deepcopy(ind1[2]), copy.deepcopy(ind2[2]))

    for i in range(len(ind1[0])):
        ind1[0][i], ind2[0][i] = ind2[0][i], ind1[0][i]

    for offspring in (ind1, ind2):
        conv_target, fc_target = offspring[0][0], offspring[0][1]
        offspring[1] = recombine_layer_pool(
            *parent_conv, conv_target, CONV_NEURON_INDEX, ascending=True
        )
        offspring[2] = recombine_layer_pool(
            *parent_fc, fc_target, FC_NEURON_INDEX, ascending=False
        )

    return ind1, ind2


def mutate_control_genes(
    control: list[int],
    control_ranges: dict[str, tuple[int, ...]],
    mutation_prob: float,
) -> None:
    """Mutate ``[conv_layers, fc_layers, epochs]`` in place within their ranges.

    Layer-count genes jump to a different legal value; ``epochs`` steps by one
    increment, clamped to its range.
    """
    for i, (key, value) in enumerate(control_ranges.items()):
        if random.random() >= mutation_prob:
            continue
        low, high, *step = value
        step = step[0] if step else 1
        if key in LAYER_COUNT_GENES:
            choices = [v for v in range(low, high + 1, step) if v != control[i]]
            if choices:
                control[i] = random.choice(choices)
        else:
            control[i] = max(min(control[i] + random.choice([-step, step]), high), low)


def resize_layer_stack(
    genes: list[list[int]], target_count: int, ranges: dict[str, Any]
) -> None:
    """Grow ``genes`` to ``target_count`` (sampling new layers) or truncate to it."""
    while len(genes) < target_count:
        genes.append(list(iter_layer_genes(ranges, len(genes), target_count)))
    del genes[target_count:]


def _step_dropout(value: int, low: int, high: int, step: int) -> int:
    """Nudge a dropout percentage by +/- one step, clamped to ``[low, high]``."""
    return max(low, min(value + random.choice([-step, step]), high))


def mutate_conv_genes(
    conv_genes: list[list[int]], conv_ranges: dict[str, Any], mutation_prob: float
) -> None:
    """Resample each conv gene's type/filter/channels and step its dropout."""
    type_low, type_high = conv_ranges["conv_type"]
    filter_low, filter_high = conv_ranges["filter_size"]
    neuron_pairs = conv_ranges["num_neurons"]
    dropout_low, dropout_high, dropout_step = conv_ranges["dropout"]

    for layer in conv_genes:
        if random.random() < mutation_prob:
            layer[0] = random.randint(type_low, type_high)
        if random.random() < mutation_prob:
            layer[1] = random.randint(filter_low, filter_high)
        if random.random() < mutation_prob:
            layer[2] = random.randint(*random.choice(neuron_pairs))
        if random.random() < mutation_prob:
            layer[3] = _step_dropout(layer[3], dropout_low, dropout_high, dropout_step)


def mutate_fc_genes(
    fc_genes: list[list[int]], fc_ranges: dict[str, Any], mutation_prob: float
) -> None:
    """Resample each fc gene's type/width and step its dropout."""
    type_low, type_high = fc_ranges["layer_type"]
    neuron_pairs = fc_ranges["num_neurons"]
    dropout_low, dropout_high, dropout_step = fc_ranges["dropout"]

    for layer in fc_genes:
        if random.random() < mutation_prob:
            layer[0] = random.randint(type_low, type_high)
        if random.random() < mutation_prob:
            layer[1] = random.randint(*random.choice(neuron_pairs))
        if random.random() < mutation_prob:
            layer[2] = _step_dropout(layer[2], dropout_low, dropout_high, dropout_step)


def mutate(
    ind: list,
    control_ranges: dict[str, tuple[int, ...]],
    conv_ranges: dict[str, Any],
    fc_ranges: dict[str, Any],
    mutation_prob: float = 1.0,
) -> tuple[list]:
    """Perturb control genes, resize the conv/fc stacks, then tweak each layer."""
    mutate_control_genes(ind[0], control_ranges, mutation_prob)

    conv_target, fc_target = ind[0][0], ind[0][1]
    resize_layer_stack(ind[1], conv_target, conv_ranges)
    resize_layer_stack(ind[2], fc_target, fc_ranges)

    mutate_conv_genes(ind[1], conv_ranges, mutation_prob)
    mutate_fc_genes(ind[2], fc_ranges, mutation_prob)

    ind[1].sort(key=lambda layer: layer[CONV_NEURON_INDEX])
    ind[2].sort(key=lambda layer: layer[FC_NEURON_INDEX], reverse=True)
    return (ind,)
