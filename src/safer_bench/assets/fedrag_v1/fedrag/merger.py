"""Document merger implementations for FedRAG.

This module provides different strategies for merging documents
retrieved from multiple federated data owners.
"""

import hashlib
from collections import defaultdict
from typing import List

import numpy as np


def get_hash(doc: str) -> bytes:
    """Create and return an SHA-256 hash for the given document."""
    return hashlib.sha256(doc.encode()).digest()


def merge_rrf(
    documents: List[str], scores: List[float], knn: int, k_rrf: int = 60
) -> List[str]:
    """Merge documents using Reciprocal Rank Fusion (RRF).

    Args:
        documents: List of retrieved documents
        scores: Corresponding retrieval scores
        knn: Number of top documents to return
        k_rrf: RRF constant (default: 60)

    Returns:
        List of merged documents ranked by RRF score
    """
    RRF_dict = defaultdict(dict)
    sorted_scores = np.array(scores).argsort()[::-1]  # Higher scores first
    sorted_documents = [documents[i] for i in sorted_scores]

    if k_rrf == 0:
        # If k_rrf is 0, simply return sorted documents by score
        return sorted_documents[:knn]

    for doc_idx, doc in enumerate(sorted_documents):
        doc_hash = get_hash(doc)
        RRF_dict[doc_hash]["rank"] = 1 / (k_rrf + doc_idx + 1)
        RRF_dict[doc_hash]["doc"] = doc

    RRF_docs = sorted(RRF_dict.values(), key=lambda x: x["rank"], reverse=True)
    return [rrf_res["doc"] for rrf_res in RRF_docs][:knn]


def merge_borda(
    documents: List[str], scores: List[float], knn: int, weighting: str = "linear"
) -> List[str]:
    """Merge documents using Borda count voting.

    Args:
        documents: List of retrieved documents
        scores: Corresponding retrieval scores
        knn: Number of top documents to return
        weighting: Weighting scheme ('linear' or 'exponential')

    Returns:
        List of merged documents ranked by Borda count
    """
    doc_votes = defaultdict(float)
    sorted_scores = np.array(scores).argsort()[::-1]  # Higher scores first
    n_docs = len(documents)

    for rank, doc_idx in enumerate(sorted_scores):
        doc = documents[doc_idx]
        doc_hash = get_hash(doc)

        if weighting == "linear":
            # Linear weighting: n_docs - rank
            vote_weight = n_docs - rank
        elif weighting == "exponential":
            # Exponential weighting: 2^(n_docs - rank)
            vote_weight = 2 ** (n_docs - rank)
        else:
            vote_weight = 1  # Equal votes

        if doc_hash in doc_votes:
            doc_votes[doc_hash] += vote_weight
        else:
            doc_votes[doc_hash] = vote_weight

    # Sort by vote count and return top-k documents
    # Note: We need to maintain doc_hash -> doc mapping
    doc_mapping = {}
    for i, doc in enumerate(documents):
        doc_hash = get_hash(doc)
        doc_mapping[doc_hash] = doc

    sorted_docs = sorted(doc_votes.items(), key=lambda x: x[1], reverse=True)
    return [doc_mapping[doc_hash] for doc_hash, _ in sorted_docs[:knn]]


def merge_score_avg(
    documents: List[str], scores: List[float], knn: int, normalization: str = "min-max"
) -> List[str]:
    """Merge documents using score averaging.

    Args:
        documents: List of retrieved documents
        scores: Corresponding retrieval scores
        knn: Number of top documents to return
        normalization: Normalization method ('min-max', 'z-score', or 'none')

    Returns:
        List of merged documents ranked by averaged scores
    """
    # Group documents by hash and collect scores
    doc_scores = defaultdict(list)
    doc_mapping = {}

    for doc, score in zip(documents, scores):
        doc_hash = get_hash(doc)
        doc_scores[doc_hash].append(score)
        doc_mapping[doc_hash] = doc

    # Apply normalization
    all_scores = list(scores)
    if normalization == "min-max":
        min_score, max_score = min(all_scores), max(all_scores)
        score_range = max_score - min_score
        if score_range > 0:
            all_scores = [(s - min_score) / score_range for s in all_scores]
    elif normalization == "z-score":
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        if std_score > 0:
            all_scores = [(s - mean_score) / std_score for s in all_scores]

    # Recompute normalized doc_scores
    if normalization != "none":
        doc_scores = defaultdict(list)
        for i, (doc, _) in enumerate(zip(documents, scores)):
            doc_hash = get_hash(doc)
            doc_scores[doc_hash].append(all_scores[i])

    # Average scores for each document
    doc_avg_scores = {}
    for doc_hash, score_list in doc_scores.items():
        doc_avg_scores[doc_hash] = np.mean(score_list)

    # Sort by average score and return top-k
    sorted_docs = sorted(doc_avg_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_mapping[doc_hash] for doc_hash, _ in sorted_docs[:knn]]


def merge_combmnz(
    documents: List[str], scores: List[float], knn: int, boost_factor: float = 2.0
) -> List[str]:
    """Merge documents using CombMNZ (Combination with boost for multiple retrievals).

    Args:
        documents: List of retrieved documents
        scores: Corresponding retrieval scores
        knn: Number of top documents to return
        boost_factor: Boost multiplier for documents retrieved by multiple DOs

    Returns:
        List of merged documents ranked by CombMNZ score
    """
    doc_scores = defaultdict(list)
    doc_mapping = {}

    for doc, score in zip(documents, scores):
        doc_hash = get_hash(doc)
        doc_scores[doc_hash].append(score)
        doc_mapping[doc_hash] = doc

    # Calculate CombMNZ score: sum(scores) * boost_factor^(num_occurrences - 1)
    doc_combmnz_scores = {}
    for doc_hash, score_list in doc_scores.items():
        sum_scores = sum(score_list)
        num_retrievals = len(score_list)

        # Apply boost for multiple retrievals
        boost = boost_factor ** (num_retrievals - 1) if num_retrievals > 1 else 1.0
        doc_combmnz_scores[doc_hash] = sum_scores * boost

    # Sort by CombMNZ score and return top-k
    sorted_docs = sorted(doc_combmnz_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_mapping[doc_hash] for doc_hash, _ in sorted_docs[:knn]]


def merge_documents(
    documents: List[str],
    scores: List[float],
    knn: int,
    merger_type: str = "rrf",
    **kwargs,
) -> List[str]:
    """Main merger function that dispatches to specific merger implementations.

    Args:
        documents: List of retrieved documents
        scores: Corresponding retrieval scores
        knn: Number of top documents to return
        merger_type: Type of merger to use ('rrf', 'borda', 'score_avg', 'combmnz')
        **kwargs: Additional parameters specific to each merger

    Returns:
        List of merged documents
    """
    merger_type = merger_type.lower()

    if merger_type == "rrf":
        k_rrf = kwargs.get("k_rrf", 60)
        return merge_rrf(documents, scores, knn, k_rrf)
    elif merger_type == "borda":
        weighting = kwargs.get("weighting", "linear")
        return merge_borda(documents, scores, knn, weighting)
    elif merger_type == "score_avg" or merger_type == "scoreavg":
        normalization = kwargs.get("normalization", "min-max")
        return merge_score_avg(documents, scores, knn, normalization)
    elif merger_type == "combmnz":
        boost_factor = kwargs.get("boost_factor", 2.0)
        return merge_combmnz(documents, scores, knn, boost_factor)
    else:
        # Fallback to RRF
        print(f"Warning: Unknown merger type '{merger_type}', falling back to RRF")
        k_rrf = kwargs.get("k_rrf", 60)
        return merge_rrf(documents, scores, knn, k_rrf)
