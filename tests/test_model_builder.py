"""Tests for decoding GA genes into CNN blocks and a full network."""

import torch
from torch import nn

from retina_evo.models.classifier import (
    RetinaCNN,
    build_conv_block,
    build_conv_stack,
    build_fc_block,
    build_network_from_individual,
)


def test_conv_block_type_1_is_conv_bn_relu():
    block = build_conv_block(1, 3, 8, 30, in_channels=3)
    assert [type(layer) for layer in block] == [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]


def test_conv_block_type_2_adds_maxpool():
    block = build_conv_block(2, 5, 16, 40, in_channels=8)
    assert [type(layer) for layer in block] == [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
    ]
    conv = block[0]
    assert conv.kernel_size == (5, 5)
    assert conv.out_channels == 16


def test_conv_block_type_3_adds_maxpool_and_dropout():
    block = build_conv_block(3, 3, 32, 50, in_channels=16)
    assert [type(layer) for layer in block] == [
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout,
    ]
    assert block[-1].p == 0.5


def test_fc_block_type_1_is_linear_relu_bn():
    block = build_fc_block(1, 64, 30, in_features=32)
    assert [type(layer) for layer in block] == [nn.Linear, nn.ReLU, nn.BatchNorm1d]


def test_fc_block_type_2_adds_dropout():
    block = build_fc_block(2, 32, 40, in_features=64)
    assert [type(layer) for layer in block] == [
        nn.Linear,
        nn.ReLU,
        nn.BatchNorm1d,
        nn.Dropout,
    ]
    assert block[-1].p == 0.4


def test_conv_stack_reports_last_channel_count():
    conv_genes = [[1, 3, 8, 30], [2, 5, 16, 40], [3, 3, 32, 50]]
    stack, out_channels = build_conv_stack(conv_genes, max_conv_layers=3, in_channels=3)
    assert out_channels == 32
    assert isinstance(stack, nn.Sequential)


def test_build_network_forward_shape():
    individual = [
        [3, 2, 10],
        [[1, 3, 8, 30], [2, 5, 16, 40], [3, 3, 32, 50]],
        [[1, 64, 30], [2, 32, 40]],
    ]
    model = build_network_from_individual(individual, num_classes=2)
    assert isinstance(model, RetinaCNN)

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 2)


def test_control_genes_cap_layer_counts():
    # Only the first conv_layers / fc_layers genes feed the network.
    individual = [
        [2, 1, 10],
        [[1, 3, 8, 30], [2, 5, 16, 40], [3, 3, 32, 50]],
        [[1, 64, 30], [2, 32, 40]],
    ]
    model = build_network_from_individual(individual, num_classes=3)

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 3)
