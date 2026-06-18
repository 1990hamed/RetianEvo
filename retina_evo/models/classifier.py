"""Build a custom CNN from a GA individual's conv + fc genes.

An individual is ``[control_genes, conv_genes, fc_genes]`` (see
``retina_evo.ga.genes``):

- ``control_genes`` -> ``[conv_layers, fc_layers, epochs]``
- each conv gene -> ``[conv_type, filter_size, num_neurons, dropout_percent]``
- each fc gene -> ``[layer_type, num_neurons, dropout_percent]``

Conv block (selected by ``conv_type``):

- ``1``: ``Conv2d -> BatchNorm2d -> ReLU``
- ``2``: the above **+** ``MaxPool2d(2, 2)``
- ``3``: the above **+** ``Dropout``

FC block (selected by ``layer_type``):

- ``1``: ``Linear -> ReLU -> BatchNorm1d``
- ``2``: the above **+** ``Dropout``

``num_neurons`` is the conv layer's output-channel count (sampled per depth in
``genes.py``); ``filter_size`` is the conv kernel size. A global
``AdaptiveAvgPool2d((1, 1))`` between the conv stack and the head fixes the
head's input width to the last conv layer's channel count, so any evolved conv
configuration yields a valid network.
"""

import torch
from torch import nn

# Gene-value sentinels shared with the GA operators and config ranges.
CONV_TYPE_WITH_POOL = 2  # conv_type >= this appends MaxPool2d
CONV_TYPE_WITH_DROPOUT = 3  # conv_type == this also appends Dropout
FC_TYPE_WITH_DROPOUT = 2  # layer_type == this appends Dropout

Individual = list  # [control_genes, conv_genes, fc_genes]


def build_conv_block(
    conv_type: int,
    filter_size: int,
    out_channels: int,
    dropout_percent: int,
    in_channels: int,
) -> list[nn.Module]:
    """Decode one conv gene into its ordered list of layers.

    ``Conv2d -> BatchNorm2d -> ReLU`` always; ``MaxPool2d(2, 2)`` when
    ``conv_type >= CONV_TYPE_WITH_POOL``; ``Dropout`` when
    ``conv_type == CONV_TYPE_WITH_DROPOUT``.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=filter_size,
            padding=filter_size // 2,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    ]
    if conv_type >= CONV_TYPE_WITH_POOL:
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    if conv_type == CONV_TYPE_WITH_DROPOUT:
        layers.append(nn.Dropout(dropout_percent / 100.0))
    return layers


def build_conv_stack(
    conv_genes: list[list[int]],
    max_conv_layers: int,
    in_channels: int = 3,
) -> tuple[nn.Sequential, int]:
    """Stack conv blocks from genes, returning the module and its output channels.

    At most ``max_conv_layers`` genes are consumed; the returned channel count is
    the final block's ``num_neurons`` (or ``in_channels`` when there are no
    conv layers).
    """
    layers: list[nn.Module] = []
    channels = in_channels
    num_layers = min(max_conv_layers, len(conv_genes))

    for conv_type, filter_size, num_neurons, dropout in conv_genes[:num_layers]:
        layers.extend(
            build_conv_block(conv_type, filter_size, num_neurons, dropout, channels)
        )
        channels = num_neurons

    return nn.Sequential(*layers), channels


def build_fc_block(
    layer_type: int,
    num_neurons: int,
    dropout_percent: int,
    in_features: int,
) -> list[nn.Module]:
    """Decode one fc gene into its ordered list of layers.

    ``Linear -> ReLU -> BatchNorm1d`` always; ``Dropout`` when
    ``layer_type == FC_TYPE_WITH_DROPOUT``.
    """
    layers: list[nn.Module] = [
        nn.Linear(in_features, num_neurons),
        nn.ReLU(),
        nn.BatchNorm1d(num_neurons),
    ]
    if layer_type == FC_TYPE_WITH_DROPOUT:
        layers.append(nn.Dropout(dropout_percent / 100.0))
    return layers


def build_classifier_head(
    fc_genes: list[list[int]],
    max_fc_layers: int,
    num_classes: int,
    in_features: int,
) -> nn.Sequential:
    """Build the fully-connected head, ending in a ``Linear`` to ``num_classes``."""
    layers: list[nn.Module] = []
    features = in_features
    num_layers = min(max_fc_layers, len(fc_genes))

    for layer_type, num_neurons, dropout in fc_genes[:num_layers]:
        layers.extend(build_fc_block(layer_type, num_neurons, dropout, features))
        features = num_neurons

    layers.append(nn.Linear(features, num_classes))
    return nn.Sequential(*layers)


def initialize_weights(module: nn.Module) -> None:
    """Kaiming-initialise conv/linear weights and reset BatchNorm layers."""
    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)


class RetinaCNN(nn.Module):
    """Evolved CNN: conv stack -> global average pool -> fc head.

    The ``AdaptiveAvgPool2d((1, 1))`` collapses spatial dimensions so the head
    receives a fixed-width vector (the conv stack's output channels) regardless
    of the evolved conv configuration.
    """

    def __init__(self, conv_stack: nn.Sequential, head: nn.Sequential):
        super().__init__()
        self.conv_stack = conv_stack
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = head
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run images through the conv stack, global pooling and the head."""
        features = self.conv_stack(x)
        pooled = self.flatten(self.global_pool(features))
        return self.head(pooled)


def build_network_from_individual(
    individual: Individual,
    num_classes: int,
    in_channels: int = 3,
) -> RetinaCNN:
    """Decode a full ``[control, conv_genes, fc_genes]`` individual into a CNN.

    ``control_genes`` is ``[conv_layers, fc_layers, epochs]``: the conv stack uses
    the first ``conv_layers`` conv genes and the head the first ``fc_layers`` fc
    genes.
    """
    control_genes, conv_genes, fc_genes = individual[0], individual[1], individual[2]
    conv_layers, fc_layers = control_genes[0], control_genes[1]

    conv_stack, conv_out_channels = build_conv_stack(
        conv_genes, conv_layers, in_channels
    )
    head = build_classifier_head(fc_genes, fc_layers, num_classes, conv_out_channels)
    return RetinaCNN(conv_stack, head)
