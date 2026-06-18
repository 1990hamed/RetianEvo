"""Tests for crossover and mutation operators."""

import copy

from retina_evo.ga.operators import (
    mate,
    mutate,
    mutate_control_genes,
    mutate_conv_genes,
    mutate_fc_genes,
    recombine_layer_pool,
    resize_layer_stack,
)

CONTROL_RANGES = {
    "conv_layers": (3, 6),
    "fc_layers": (1, 2),
    "epochs": (10, 100, 10),
}
CONV_RANGES = {
    "conv_type": (1, 3),
    "filter_size": (3, 5),
    "num_neurons": [(16, 32), (16, 32), (64, 128), (64, 128), (256, 512), (256, 512)],
    "dropout": (10, 50, 10),
}
FC_RANGES = {
    "layer_type": (1, 2),
    "num_neurons": [(16, 32), (16, 32), (64, 128), (64, 128), (256, 512), (256, 512)],
    "dropout": (10, 50, 10),
}


def _individual(conv_layers=4, fc_layers=2, epochs=20) -> list:
    conv_genes = [[1, 3, 16 + i * 10, 10] for i in range(conv_layers)]
    conv_genes.sort(key=lambda layer: layer[2])
    fc_genes = [[1, 256 - i * 10, 20] for i in range(fc_layers)]
    fc_genes.sort(key=lambda layer: layer[1], reverse=True)
    return [[conv_layers, fc_layers, epochs], conv_genes, fc_genes]


def test_recombine_layer_pool_returns_target_count():
    parent_a = [[1, 3, 16, 10], [1, 3, 64, 20]]
    parent_b = [[1, 3, 128, 10], [1, 3, 256, 20]]
    result = recombine_layer_pool(
        parent_a, parent_b, target_count=3, neuron_index=2, ascending=True
    )
    assert len(result) == 3


def test_recombine_layer_pool_orders_ascending():
    parent_a = [[1, 3, 16, 10], [1, 3, 64, 20]]
    parent_b = [[1, 3, 128, 10], [1, 3, 256, 20]]
    result = recombine_layer_pool(
        parent_a, parent_b, target_count=4, neuron_index=2, ascending=True
    )
    widths = [layer[2] for layer in result]
    assert widths == sorted(widths)


def test_recombine_layer_pool_orders_descending():
    parent_a = [[1, 256, 10], [1, 64, 20]]
    parent_b = [[1, 128, 10], [1, 16, 20]]
    result = recombine_layer_pool(
        parent_a, parent_b, target_count=4, neuron_index=1, ascending=False
    )
    widths = [layer[1] for layer in result]
    assert widths == sorted(widths, reverse=True)


def test_recombine_layer_pool_clones_when_pool_too_small():
    parent_a = [[1, 3, 16, 10]]
    parent_b = [[1, 3, 16, 10]]
    result = recombine_layer_pool(
        parent_a, parent_b, target_count=3, neuron_index=2, ascending=True
    )
    assert len(result) == 3


def test_recombine_layer_pool_does_not_mutate_parents():
    parent_a = [[1, 3, 16, 10]]
    parent_b = [[1, 3, 128, 10]]
    parent_a_copy, parent_b_copy = copy.deepcopy(parent_a), copy.deepcopy(parent_b)
    recombine_layer_pool(
        parent_a, parent_b, target_count=2, neuron_index=2, ascending=True
    )
    assert parent_a == parent_a_copy
    assert parent_b == parent_b_copy


def test_mate_swaps_control_genes():
    ind1, ind2 = (
        _individual(conv_layers=3, fc_layers=1),
        _individual(conv_layers=5, fc_layers=2),
    )
    orig1_control, orig2_control = copy.deepcopy(ind1[0]), copy.deepcopy(ind2[0])
    offspring1, offspring2 = mate(ind1, ind2)
    assert offspring1[0] == orig2_control
    assert offspring2[0] == orig1_control


def test_mate_returns_layer_counts_matching_control():
    ind1, ind2 = (
        _individual(conv_layers=3, fc_layers=1),
        _individual(conv_layers=5, fc_layers=2),
    )
    offspring1, offspring2 = mate(ind1, ind2)
    for offspring in (offspring1, offspring2):
        conv_target, fc_target = offspring[0][0], offspring[0][1]
        assert len(offspring[1]) == conv_target
        assert len(offspring[2]) == fc_target


def test_mate_conv_genes_ascending_fc_genes_descending():
    ind1, ind2 = (
        _individual(conv_layers=4, fc_layers=2),
        _individual(conv_layers=4, fc_layers=2),
    )
    offspring1, offspring2 = mate(ind1, ind2)
    for offspring in (offspring1, offspring2):
        conv_widths = [layer[2] for layer in offspring[1]]
        fc_widths = [layer[1] for layer in offspring[2]]
        assert conv_widths == sorted(conv_widths)
        assert fc_widths == sorted(fc_widths, reverse=True)


def test_mutate_control_genes_layer_counts_change_value():
    control = [3, 1, 10]
    mutate_control_genes(control, CONTROL_RANGES, mutation_prob=1.0)
    assert control[0] != 3
    assert 3 <= control[0] <= 6
    assert control[1] != 1
    assert 1 <= control[1] <= 2


def test_mutate_control_genes_epochs_steps_by_increment():
    control = [3, 1, 50]
    mutate_control_genes(control, CONTROL_RANGES, mutation_prob=1.0)
    assert control[2] in (40, 60)


def test_mutate_control_genes_epochs_clamped_to_range():
    control = [3, 1, 10]
    mutate_control_genes(control, CONTROL_RANGES, mutation_prob=1.0)
    assert 10 <= control[2] <= 100
    control = [3, 1, 100]
    mutate_control_genes(control, CONTROL_RANGES, mutation_prob=1.0)
    assert 10 <= control[2] <= 100


def test_mutate_control_genes_no_mutation_when_prob_zero():
    control = [3, 1, 50]
    mutate_control_genes(control, CONTROL_RANGES, mutation_prob=0.0)
    assert control == [3, 1, 50]


def test_resize_layer_stack_grows():
    genes = [[1, 3, 16, 10]]
    resize_layer_stack(genes, target_count=4, ranges=CONV_RANGES)
    assert len(genes) == 4


def test_resize_layer_stack_truncates():
    genes = [[1, 3, 16, 10], [1, 3, 32, 10], [1, 3, 64, 10]]
    resize_layer_stack(genes, target_count=1, ranges=CONV_RANGES)
    assert len(genes) == 1
    assert genes[0] == [1, 3, 16, 10]


def test_resize_layer_stack_noop_when_already_target():
    genes = [[1, 3, 16, 10], [1, 3, 32, 10]]
    original = copy.deepcopy(genes)
    resize_layer_stack(genes, target_count=2, ranges=CONV_RANGES)
    assert genes == original


def test_mutate_conv_genes_within_ranges():
    conv_genes = [[1, 3, 16, 10] for _ in range(4)]
    mutate_conv_genes(conv_genes, CONV_RANGES, mutation_prob=1.0)
    for conv_type, filter_size, num_neurons, dropout in conv_genes:
        assert 1 <= conv_type <= 3
        assert 3 <= filter_size <= 5
        assert 16 <= num_neurons <= 512
        assert 10 <= dropout <= 50


def test_mutate_conv_genes_no_change_when_prob_zero():
    conv_genes = [[1, 3, 16, 10], [2, 4, 64, 20]]
    original = copy.deepcopy(conv_genes)
    mutate_conv_genes(conv_genes, CONV_RANGES, mutation_prob=0.0)
    assert conv_genes == original


def test_mutate_fc_genes_within_ranges():
    fc_genes = [[1, 256, 20] for _ in range(2)]
    mutate_fc_genes(fc_genes, FC_RANGES, mutation_prob=1.0)
    for layer_type, num_neurons, dropout in fc_genes:
        assert 1 <= layer_type <= 2
        assert 16 <= num_neurons <= 512
        assert 10 <= dropout <= 50


def test_mutate_fc_genes_no_change_when_prob_zero():
    fc_genes = [[1, 256, 20], [2, 64, 30]]
    original = copy.deepcopy(fc_genes)
    mutate_fc_genes(fc_genes, FC_RANGES, mutation_prob=0.0)
    assert fc_genes == original


def test_mutate_returns_tuple_with_individual():
    ind = _individual()
    result = mutate(ind, CONTROL_RANGES, CONV_RANGES, FC_RANGES, mutation_prob=1.0)
    assert result == (ind,)


def test_mutate_resizes_layers_to_match_new_control():
    ind = _individual(conv_layers=3, fc_layers=1)
    (mutated,) = mutate(ind, CONTROL_RANGES, CONV_RANGES, FC_RANGES, mutation_prob=1.0)
    conv_target, fc_target = mutated[0][0], mutated[0][1]
    assert len(mutated[1]) == conv_target
    assert len(mutated[2]) == fc_target


def test_mutate_keeps_conv_ascending_and_fc_descending():
    ind = _individual(conv_layers=5, fc_layers=2)
    (mutated,) = mutate(ind, CONTROL_RANGES, CONV_RANGES, FC_RANGES, mutation_prob=1.0)
    conv_widths = [layer[2] for layer in mutated[1]]
    fc_widths = [layer[1] for layer in mutated[2]]
    assert conv_widths == sorted(conv_widths)
    assert fc_widths == sorted(fc_widths, reverse=True)
