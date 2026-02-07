"""Weighted RRF - RRF with per-source weights for trust/quality."""

from typing import Optional

import numpy as np
from pydantic import Field

from .base import BaseMerger, MergerConfig, MergerResult, _empty_result


class WeightedRRFConfig(MergerConfig):
    """Configuration for Weighted RRF merger."""

    k_rrf: int = Field(default=60, description="RRF constant parameter")
    weights: Optional[list[float]] = Field(
        default=None,
        description="Per-DO weights (indexed by DO). If None, uses uniform weights.",
    )


class WeightedRRFMerger(BaseMerger):
    """Weighted RRF - RRF with per-source weights for trust/quality.

    Extends standard RRF by allowing different weights for each data owner,
    enabling trust-aware fusion where higher-quality DOs contribute more.

    Without source information, falls back to standard RRF behavior.
    """

    def __init__(self, config: WeightedRRFConfig):
        super().__init__(config)
        self.k_rrf = config.k_rrf
        self.weights = config.weights

    def merge(
        self,
        documents: list[str],
        scores: list[float],
        sources: Optional[list[int]] = None,
    ) -> MergerResult:
        """Weighted RRF: score(d) = sum(w_i / (k + rank(d))) where w_i is source weight."""
        if not documents:
            return _empty_result()

        if sources is None:
            # Fall back to standard RRF if no source info
            from .rrf import RRFConfig, RRFMerger

            rrf_config = RRFConfig(knn=self.knn, k_rrf=self.k_rrf)
            return RRFMerger(rrf_config).merge(documents, scores, sources)

        num_dos = max(sources) + 1 if sources else 1
        weights = self.weights if self.weights else [1.0] * num_dos

        sorted_indices = np.argsort(scores)  # L2: lower is better

        doc_scores: dict[str, dict] = {}
        for rank, idx in enumerate(sorted_indices):
            doc = documents[idx]
            source = sources[idx]
            doc_hash = self.get_hash(doc)

            weight = weights[source] if source < len(weights) else 1.0
            # rank + 1 converts 0-indexed to 1-indexed (per RRF paper)
            rrf_score = weight / (self.k_rrf + rank + 1)

            if doc_hash in doc_scores:
                doc_scores[doc_hash]["score"] += rrf_score
                doc_scores[doc_hash]["sources"].add(source)
            else:
                doc_scores[doc_hash] = {
                    "score": rrf_score,
                    "doc": doc,
                    "sources": {source},
                }

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        top_k = sorted_docs[: self.knn]

        return MergerResult(
            documents=[d["doc"] for d in top_k],
            scores=[d["score"] for d in top_k],
            source_counts=[len(d["sources"]) for d in top_k],
        )
