"""Custom-CNN model builders decoded from GA individuals."""

from retina_evo.models.classifier import (
    CONV_TYPE_WITH_DROPOUT,
    CONV_TYPE_WITH_POOL,
    FC_TYPE_WITH_DROPOUT,
    RetinaCNN,
    build_classifier_head,
    build_conv_block,
    build_conv_stack,
    build_fc_block,
    build_network_from_individual,
    initialize_weights,
)

__all__ = [
    "CONV_TYPE_WITH_DROPOUT",
    "CONV_TYPE_WITH_POOL",
    "FC_TYPE_WITH_DROPOUT",
    "RetinaCNN",
    "build_classifier_head",
    "build_conv_block",
    "build_conv_stack",
    "build_fc_block",
    "build_network_from_individual",
    "initialize_weights",
]
