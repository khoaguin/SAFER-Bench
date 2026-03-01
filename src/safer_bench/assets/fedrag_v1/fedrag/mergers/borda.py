"""Borda Count merger - position-based voting.

Reference: Aslam & Montague, SIGIR 2001 - "Models for Metasearch"
"""

from typing import Optional

import numpy as np

from .base import BaseMerger, MergerConfig, MergerResult, _empty_result


class BordaConfig(MergerConfig):
    """Configuration for Borda Count merger."""

    pass  # Only needs base knn parameter


class BordaMerger(BaseMerger):
    """Borda Count merger - position-based voting.

    Each document receives points based on its rank position:
    - Rank 1 gets n points, rank 2 gets n-1 points, etc.

    Note: Scores are not comparable to RRF or CombSUM due to different scales.
    Use only for ranking within this merger strategy.
    """

    def __init__(self, config: BordaConfig):
        super().__init__(config)

    def merge(
        self,
        documents: list[str],
        scores: list[float],
        sources: Optional[list[int]] = None,
    ) -> MergerResult:
        """Borda: score(d) = sum(n - rank(d)) where n = total docs."""
        if not documents:
            return _empty_result()

        n = len(documents)
        sorted_indices = np.argsort(scores)  # L2: lower is better

        doc_scores: dict[str, dict] = {}
        for rank, idx in enumerate(sorted_indices):
            doc = documents[idx]
            doc_hash = self.get_hash(doc)
            borda_score = n - rank
            source = sources[idx] if sources else 0

            if doc_hash in doc_scores:
                doc_scores[doc_hash]["score"] += borda_score
                doc_scores[doc_hash]["sources"].add(source)
            else:
                doc_scores[doc_hash] = {
                    "score": borda_score,
                    "doc": doc,
                    "sources": {source},
                }

        sorted_docs = sorted(
            doc_scores.values(), key=lambda x: x["score"], reverse=True
        )
        top_k = sorted_docs[: self.knn]

        result = MergerResult(
            documents=[d["doc"] for d in top_k],
            scores=[d["score"] for d in top_k],
            source_counts=[len(d["sources"]) for d in top_k],
        )
        self._log_merge_stats("Borda", len(documents), len(doc_scores), sources, result)
        return result
