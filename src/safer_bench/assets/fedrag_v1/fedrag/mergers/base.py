"""Base classes for document mergers in federated RAG."""

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class MergerResult(BaseModel):
    """Result from a merger operation."""

    documents: list[str] = Field(description="Top-k merged documents")
    scores: list[float] = Field(description="Final scores for each document")
    source_counts: list[int] = Field(
        description="Number of unique sources that returned each doc"
    )

    model_config = {"frozen": True}


def _empty_result() -> MergerResult:
    """Return an empty result for edge cases."""
    return MergerResult(documents=[], scores=[], source_counts=[])


class MergerConfig(BaseModel):
    """Base configuration for mergers."""

    knn: int = Field(default=8, description="Number of documents to return")

    model_config = {"extra": "allow"}  # Allow subclass-specific params


class BaseMerger(ABC):
    """Abstract base class for document mergers."""

    def __init__(self, config: MergerConfig):
        self.config = config
        self.knn = config.knn

    @abstractmethod
    def merge(
        self,
        documents: list[str],
        scores: list[float],
        sources: Optional[list[int]] = None,
    ) -> MergerResult:
        """Merge documents from multiple sources.

        Args:
            documents: All documents from all DOs (flattened)
            scores: FAISS L2 distances (lower = better)
            sources: DO index for each document (for weighted/MNZ strategies)

        Returns:
            MergerResult with top-k documents and metadata
        """
        pass

    @staticmethod
    def get_hash(doc: str) -> str:
        """Create SHA-256 hash for document deduplication."""
        return hashlib.sha256(doc.encode()).hexdigest()
