"""Reciprocal Rank Fusion (RRF) merger.

Reference: Cormack, Clarke & Büttcher, SIGIR 2009
"Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods"
"""

from typing import Optional

import numpy as np
from pydantic import Field

from .base import BaseMerger, MergerConfig, MergerResult, _empty_result


class RRFConfig(MergerConfig):
    """Configuration for RRF merger."""

    k_rrf: int = Field(default=60, description="RRF constant parameter")


class RRFMerger(BaseMerger):
    """Reciprocal Rank Fusion merger.

    RRF combines rankings by assigning score 1/(k + rank) to each document,
    where k=60 is a constant that mitigates the impact of outlier rankings.
    """

    def __init__(self, config: RRFConfig):
        super().__init__(config)
        self.k_rrf = config.k_rrf

    def merge(
        self,
        documents: list[str],
        scores: list[float],
        sources: Optional[list[int]] = None,
    ) -> MergerResult:
        """RRF: score(d) = sum(1 / (k + rank(d))) where rank is 1-indexed."""
        if not documents:
            return _empty_result()

        sorted_indices = np.argsort(scores)  # L2: lower is better

        doc_scores: dict[str, dict] = {}
        for rank, idx in enumerate(sorted_indices):
            doc = documents[idx]
            doc_hash = self.get_hash(doc)
            # rank + 1 converts 0-indexed to 1-indexed (per original RRF paper)
            rrf_score = 1.0 / (self.k_rrf + rank + 1)
            source = sources[idx] if sources else 0

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
