#!/usr/bin/env python3
"""Pre-build FAISS indexes for partition directories.

This script builds FAISS indexes for full dataset partitions before running
benchmarks, avoiding runtime timeouts for large datasets like MIMIC-IV-Note
specialty partitions.

Usage:
    # Build indexes for all full partitions
    uv run python scripts/prebuild_faiss_indexes.py

    # Build indexes for a specific federation
    uv run python scripts/prebuild_faiss_indexes.py --federation specialty_7do

    # Build index for a specific data owner in a federation
    uv run python scripts/prebuild_faiss_indexes.py --federation specialty_7do --do do1

    # Build indexes for subset partitions (usually already done)
    uv run python scripts/prebuild_faiss_indexes.py --mode subsets

    # Force rebuild even if index exists
    uv run python scripts/prebuild_faiss_indexes.py --force
"""

import argparse
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Default configuration (matches fedrag/retriever.yaml)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384


def build_faiss_index(
    chunk_dir: Path,
    output_dir: Path,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    batch_size: int = 32,
) -> bool:
    """Build a FAISS index from chunk files.

    This function replicates the exact logic from fedrag/retriever.py:build_faiss_index()
    to ensure consistent index building across the codebase.

    Args:
        chunk_dir: Directory containing .jsonl chunk files
        output_dir: Directory to save faiss.index and all_doc_ids.npy
        embedding_model: SentenceTransformer model name
        embedding_dim: Expected embedding dimension
        batch_size: Batch size for encoding

    Returns:
        True if successful, False otherwise
    """
    index_path = output_dir / "faiss.index"
    doc_ids_path = output_dir / "all_doc_ids.npy"

    # Erase previous files (matches retriever.py behavior)
    try:
        index_path.unlink()
        doc_ids_path.unlink()
    except (OSError, FileNotFoundError):
        pass

    # Load embedding model
    logger.info(f"Loading embedding model: {embedding_model}")
    emb_model = SentenceTransformer(embedding_model)

    # Get all chunk files (using list() to match retriever.py)
    all_files = list(chunk_dir.glob("*.jsonl"))
    if not all_files:
        logger.warning(f"No chunk files found in {chunk_dir}")
        return False

    logger.info(f"🔨 Building FAISS index for {output_dir.parent.name}")
    logger.info(f"📁 Found {len(all_files)} chunk files to process")

    all_embeddings = []
    all_doc_ids = []

    # Loop through all .jsonl files - exact logic from retriever.py:60-85
    for filename in tqdm(all_files):
        batch_content, batch_ids = [], []
        with open(filename, "r", encoding="utf-8") as infile:
            for line in infile:
                doc = json.loads(line)
                doc_id = doc.get("id", "")
                content = doc.get("content", "")
                batch_ids.append(doc_id)
                batch_content.append(content)

                # Note: uses > not >= to match retriever.py:70
                if len(batch_ids) > batch_size:
                    # Generate embeddings for the batch
                    batch_embeddings = emb_model.encode(
                        batch_content, convert_to_numpy=True
                    )
                    all_embeddings.extend(batch_embeddings)
                    all_doc_ids.extend(batch_ids)
                    batch_content, batch_ids = [], []

            # Process last batch (within file scope)
            if batch_content:
                batch_embeddings = emb_model.encode(
                    batch_content, convert_to_numpy=True
                )
                all_embeddings.extend(batch_embeddings)
                all_doc_ids.extend(batch_ids)

    # Filter out embeddings with incorrect dimensions - matches retriever.py:87-92
    filtered_embeddings = [
        embedding
        for embedding in all_embeddings
        if embedding is not None and embedding.shape == (embedding_dim,)
    ]

    if not filtered_embeddings:
        logger.error("No valid embeddings generated")
        return False

    # FAISS needs float32 - matches retriever.py:94
    embeddings = np.array(filtered_embeddings).astype("float32")
    d = embeddings.shape[1]  # Dimensionality of the embeddings

    # Quantizer for IVF - matches retriever.py:98
    quantizer = faiss.IndexFlatL2(d)

    # Number of clusters - matches retriever.py:101
    nlist = int(np.sqrt(len(embeddings)))

    # METRIC_L2 measures dissimilarity - matches retriever.py:104
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # Train the index - matches retriever.py:107
    logger.info(f"Training index with {nlist} clusters...")
    index.train(embeddings)

    # Add the embeddings to the index - matches retriever.py:110
    index.add(embeddings)

    # Save the index - matches retriever.py:113
    faiss.write_index(index, str(index_path))

    # Save document IDs - matches retriever.py:116
    np.save(str(doc_ids_path), np.array(all_doc_ids))

    logger.info(
        f"✅ FAISS index built successfully: {len(all_doc_ids)} documents indexed"
    )
    return True


def find_partitions(
    base_dir: Path,
    mode: str = "full",
    federation: str | None = None,
    do: str | None = None,
) -> list[Path]:
    """Find partition directories that need index building.

    Args:
        base_dir: Project root directory
        mode: 'full' or 'subsets'
        federation: Optional federation name filter (e.g., 'specialty_7do')
        do: Optional data owner filter (e.g., 'do1', 'do2')

    Returns:
        List of partition private directories
    """
    partitions_base = base_dir / "datasets" / mode / "partitions"

    if not partitions_base.exists():
        logger.warning(f"Partitions directory not found: {partitions_base}")
        return []

    partitions = []

    # Pattern: partitions/{federation}/{do}/private/
    if federation:
        federation_dirs = [partitions_base / federation]
    else:
        federation_dirs = [d for d in partitions_base.iterdir() if d.is_dir()]

    for fed_dir in federation_dirs:
        if not fed_dir.exists():
            logger.warning(f"Federation directory not found: {fed_dir}")
            continue

        # Filter by specific DO if provided
        if do:
            do_dirs = [fed_dir / do]
        else:
            do_dirs = sorted(fed_dir.iterdir())

        for do_dir in do_dirs:
            if not do_dir.is_dir():
                continue

            private_dir = do_dir / "private"
            if private_dir.exists():
                partitions.append(private_dir)

    return partitions


def main():
    parser = argparse.ArgumentParser(
        description="Pre-build FAISS indexes for partition directories"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "subsets"],
        default="full",
        help="Dataset mode to process (default: full)",
    )
    parser.add_argument(
        "--federation",
        type=str,
        default=None,
        help="Specific federation to process (e.g., specialty_7do)",
    )
    parser.add_argument(
        "--do",
        type=str,
        default=None,
        help="Specific data owner to process (e.g., do1, do2). Requires --federation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    args = parser.parse_args()

    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    logger.info(f"Project root: {project_root}")

    # Validate --do requires --federation
    if args.do and not args.federation:
        logger.error("--do requires --federation to be specified")
        return

    # Find partitions to process
    partitions = find_partitions(project_root, args.mode, args.federation, args.do)
    logger.info(f"Found {len(partitions)} partitions to process")

    if not partitions:
        logger.warning("No partitions found to process")
        return

    # Track results
    built = 0
    skipped = 0
    failed = 0

    for partition_dir in partitions:
        chunk_dir = partition_dir / "chunk"
        index_path = partition_dir / "faiss.index"

        # Get partition name for logging
        partition_name = f"{partition_dir.parents[1].name}/{partition_dir.parent.name}"

        if not chunk_dir.exists():
            logger.warning(f"[{partition_name}] No chunk directory found, skipping")
            skipped += 1
            continue

        if index_path.exists() and not args.force:
            logger.info(
                f"[{partition_name}] Index already exists, skipping (use --force to rebuild)"
            )
            skipped += 1
            continue

        logger.info(f"[{partition_name}] Building FAISS index...")

        try:
            success = build_faiss_index(
                chunk_dir=chunk_dir,
                output_dir=partition_dir,
                batch_size=args.batch_size,
            )

            if success:
                # Copy index to mock directory (Syft RDS requires matching extensions)
                mock_dir = partition_dir.parent / "mock"
                if mock_dir.exists():
                    logger.info(f"[{partition_name}] Copying index to mock directory")
                    shutil.copy2(index_path, mock_dir / "faiss.index")
                    shutil.copy2(
                        partition_dir / "all_doc_ids.npy",
                        mock_dir / "all_doc_ids.npy",
                    )

                logger.success(f"[{partition_name}] Index built successfully")
                built += 1
            else:
                logger.error(f"[{partition_name}] Index build failed")
                failed += 1

        except Exception as e:
            logger.exception(f"[{partition_name}] Error building index: {e}")
            failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info(f"Summary: {built} built, {skipped} skipped, {failed} failed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
