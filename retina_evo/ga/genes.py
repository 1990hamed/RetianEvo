"""Gene generation for the hierarchical GA (conv + fc), generator style.

An individual is ``[control_genes, conv_genes, fc_genes]``:
- control_genes: ``[conv_layers, fc_layers, epochs]`` drawn from ``control_ranges``
- conv_genes: one ``[conv_type, filter_size, num_neurons, dropout]`` quad per conv
  layer, drawn from ``conv_ranges``
- fc_genes: one ``[layer_type, num_neurons, dropout]`` triple per FC layer,
  drawn from ``fc_ranges``

The per-block samplers are Python generators (``yield``); ``list(...)`` around a
call materialises one gene vector. ``make_individual_generator`` returns the
zero-arg callable that DEAP's ``tools.initIterate`` consumes to build a whole
individual.
"""

import random
from collections.abc import Callable, Iterator
from typing import Any


def iter_control_genes(ranges: dict[str, tuple[int, ...]]) -> Iterator[int]:
    """Yield the top-level genes (conv_layers, fc_layers, epochs)."""
    for value in ranges.values():
        low, high, *step = value
        step = step[0] if step else 1
        yield random.choice(range(low, high + 1, step))


def iter_layer_genes(
    ranges: dict[str, Any], index: int, num_layers: int
) -> Iterator[int]:
    """Yield one layer's gene vector from ``ranges`` for layer ``index``.

    Each entry in ``ranges`` is either:
    - a flat ``(low, high[, step])`` range -> sampled with ``random.choice``, or
    - a list of ``(low, high)`` pairs (its elements are themselves sequences)
      -> the pair at ``index`` when the list is long enough (positional,
      depth-ordered), else a random pair, then sampled with ``random.randint``.
    """
    for value in ranges.values():
        if value and isinstance(value[0], (list, tuple)):
            low, high = (
                value[index] if len(value) >= num_layers else random.choice(value)
            )
            yield random.randint(low, high)
        else:
            low, high, *step = value
            step = step[0] if step else 1
            yield random.choice(range(low, high + 1, step))


def iter_parametric_genes(
    ranges: dict[str, Any], num_layers: int
) -> Iterator[list[int]]:
    """Yield ``num_layers`` per-layer gene vectors from ``ranges``."""
    for i in range(num_layers):
        yield list(iter_layer_genes(ranges, i, num_layers))


def make_individual_generator(
    control_ranges: dict[str, tuple[int, ...]],
    conv_ranges: dict[str, Any],
    fc_ranges: dict[str, Any],
) -> Callable[[], Iterator[list]]:
    """Build the zero-arg generator DEAP uses to create one individual.

    ``tools.initIterate(container, generator)`` calls ``generator()`` and wraps
    the yielded items with ``container``. Yielding the three gene blocks
    separately therefore produces an individual shaped exactly as
    ``[control_genes, conv_genes, fc_genes]``:

    - ``control_genes`` -> ``[conv_layers, fc_layers, epochs]``
    - ``conv_genes`` -> ``conv_layers`` x ``[conv_type, filter_size, num_neurons, dropout]``
    - ``fc_genes`` -> ``fc_layers`` x ``[layer_type, num_neurons, dropout]``

    The first two control genes drive how many conv and fc layer-gene vectors
    are sampled; repetition across a population is handled by DEAP.
    """

    def generate_individual() -> Iterator[list]:
        control_genes = list(iter_control_genes(control_ranges))
        conv_layers, fc_layers = control_genes[0], control_genes[1]
        yield control_genes
        yield list(iter_parametric_genes(conv_ranges, conv_layers))
        yield list(iter_parametric_genes(fc_ranges, fc_layers))

    return generate_individual
