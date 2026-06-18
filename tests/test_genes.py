"""Tests for individual generation from GA ranges."""

from retina_evo.ga.genes import make_individual_generator

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


def _make_individual() -> list:
    generate = make_individual_generator(CONTROL_RANGES, CONV_RANGES, FC_RANGES)
    return list(generate())


def test_individual_has_three_blocks():
    assert len(_make_individual()) == 3


def test_layer_counts_match_control_genes():
    for _ in range(20):
        control, conv_genes, fc_genes = _make_individual()
        conv_layers, fc_layers, _epochs = control
        assert len(conv_genes) == conv_layers
        assert len(fc_genes) == fc_layers


def test_control_genes_within_ranges():
    control, _conv, _fc = _make_individual()
    conv_layers, fc_layers, epochs = control
    assert 3 <= conv_layers <= 6
    assert 1 <= fc_layers <= 2
    assert epochs in range(10, 101, 10)


def test_conv_genes_within_ranges():
    _control, conv_genes, _fc = _make_individual()
    for conv_type, filter_size, num_neurons, dropout in conv_genes:
        assert 1 <= conv_type <= 3
        assert 3 <= filter_size <= 5
        assert 16 <= num_neurons <= 512
        assert dropout in range(10, 51, 10)


def test_fc_genes_within_ranges():
    _control, _conv, fc_genes = _make_individual()
    for layer_type, num_neurons, dropout in fc_genes:
        assert 1 <= layer_type <= 2
        assert 16 <= num_neurons <= 512
        assert dropout in range(10, 51, 10)
