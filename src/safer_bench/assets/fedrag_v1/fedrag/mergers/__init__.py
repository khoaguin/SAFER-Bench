"""Merger module for federated RAG result aggregation.

Provides multiple strategies for combining retrieval results from
distributed data owners.
"""

from typing import Any, Dict, Type

from .base import BaseMerger, MergerConfig, MergerResult
from .borda import BordaConfig, BordaMerger
from .combmnz import CombMNZConfig, CombMNZMerger
from .combsum import CombSUMConfig, CombSUMMerger
from .rrf import RRFConfig, RRFMerger
from .weighted_rrf import WeightedRRFConfig, WeightedRRFMerger

MERGER_REGISTRY: Dict[str, tuple[Type[BaseMerger], Type[MergerConfig]]] = {
    "rrf": (RRFMerger, RRFConfig),
    "borda": (BordaMerger, BordaConfig),
    "combsum": (CombSUMMerger, CombSUMConfig),
    "combmnz": (CombMNZMerger, CombMNZConfig),
    "weighted_rrf": (WeightedRRFMerger, WeightedRRFConfig),
}


def create_merger(merger_type: str, **config: Any) -> BaseMerger:
    """Factory function to create merger instances.

    Args:
        merger_type: One of 'rrf', 'borda', 'combsum', 'combmnz', 'weighted_rrf'
                     (case-insensitive)
        **config: Strategy-specific parameters (including knn)

    Returns:
        Configured merger instance

    Raises:
        ValueError: If merger_type is not recognized
    """
    # Normalize to lowercase for case-insensitive matching
    merger_type = merger_type.lower().strip()

    if merger_type not in MERGER_REGISTRY:
        raise ValueError(
            f"Unknown merger type: {merger_type}. Available: {list(MERGER_REGISTRY.keys())}"
        )

    merger_class, config_class = MERGER_REGISTRY[merger_type]
    merger_config = config_class(**config)
    return merger_class(config=merger_config)


__all__ = [
    "create_merger",
    "BaseMerger",
    "MergerConfig",
    "MergerResult",
    "RRFMerger",
    "RRFConfig",
    "BordaMerger",
    "BordaConfig",
    "CombSUMMerger",
    "CombSUMConfig",
    "CombMNZMerger",
    "CombMNZConfig",
    "WeightedRRFMerger",
    "WeightedRRFConfig",
    "MERGER_REGISTRY",
]
