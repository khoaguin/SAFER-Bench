"""CombSUM - sum of normalized similarity scores.

Reference: Fox & Shaw, TREC-2 1994 - "Combination of Multiple Searches"
"""

from typing import Literal, Optional

import numpy as np
from pydantic import Field

from .base import BaseMerger, MergerConfig, MergerResult, _empty_result


class CombSUMConfig(MergerConfig):
    """Configuration for CombSUM merger."""

    normalization: Literal["minmax"] = Field(
        default="minmax",
        description="Score normalization method (only minmax supported)",
    )


class CombSUMMerger(BaseMerger):
    """CombSUM - sum of normalized similarity scores.

    Converts L2 distances to similarity scores, normalizes to [0,1], then sums.
    Documents that appear multiple times have their scores accumulated.
    """

    def __init__(self, config: CombSUMConfig):
        super().__init__(config)
        self.normalization = config.normalization

    def _normalize_scores(self, scores: list[float]) -> np.ndarray:
        """Convert L2 distances to normalized similarity scores in [0, 1].

        IMPORTANT: L2 distance is lower-is-better, so we must:
        1. Convert to similarity (higher-is-better) by inverting
        2. Normalize to [0, 1] range

        Args:
            scores: FAISS L2 distances (lower = more similar)

        Returns:
            Normalized similarity scores in [0, 1] (higher = more similar)
        """
        scores_arr = np.array(scores, dtype=np.float64)

        # Step 1: Convert L2 distances to similarities (invert)
        # Using 1/(1+d) to avoid division by zero and keep bounded
        similarities = 1.0 / (1.0 + scores_arr)

        # Step 2: Min-max normalize to [0, 1]
        min_s, max_s = similarities.min(), similarities.max()
        if max_s - min_s > 1e-8:
            normalized = (similarities - min_s) / (max_s - min_s)
        else:
            # All scores are the same - return uniform weights
            normalized = np.ones_like(similarities)

        return normalized

    def merge(
        self,
        documents: list[str],
        scores: list[float],
        sources: Optional[list[int]] = None,
    ) -> MergerResult:
        """CombSUM: score(d) = sum(normalized_similarity(d))."""
        if not documents:
            return _empty_result()

        normalized = self._normalize_scores(scores)

        doc_scores: dict[str, dict] = {}
        for idx, (doc, norm_score) in enumerate(zip(documents, normalized)):
            doc_hash = self.get_hash(doc)
            source = sources[idx] if sources else 0

            if doc_hash in doc_scores:
                doc_scores[doc_hash]["score"] += norm_score
                doc_scores[doc_hash]["sources"].add(source)
            else:
                doc_scores[doc_hash] = {
                    "score": float(norm_score),
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
        self._log_merge_stats(
            "CombSUM", len(documents), len(doc_scores), sources, result
        )
        return result
