#!/usr/bin/env python3
"""Generate specialty mapping for MIMIC-IV-Note dataset.

This script classifies all clinical note chunks into medical specialties
and saves the mapping to specialty_mapping.json files for both full and
subset datasets.

Usage:
    python scripts/generate_specialty_mapping.py [--subset-only] [--full-only]

The script will generate:
    - datasets/full/mimic-iv-note/specialty_mapping.json
    - datasets/subsets/mimic-iv-note/specialty_mapping.json
"""

import argparse
import sys
from pathlib import Path

from loguru import logger

from safer_bench.specialty_classifier import generate_specialty_mapping

# Add src to path for imports
src_path = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(src_path))


def main():
    parser = argparse.ArgumentParser(
        description="Generate specialty mapping for MIMIC-IV-Note dataset"
    )
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Only generate mapping for subset dataset",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Only generate mapping for full dataset",
    )
    args = parser.parse_args()

    # Determine project root
    project_root = Path(__file__).parents[1]

    # Define paths
    datasets = []
    if not args.full_only:
        datasets.append(
            ("subsets", project_root / "datasets" / "subsets" / "mimic-iv-note")
        )
    if not args.subset_only:
        datasets.append(("full", project_root / "datasets" / "full" / "mimic-iv-note"))

    for mode, dataset_path in datasets:
        chunks_dir = dataset_path / "private" / "chunk"
        output_path = dataset_path / "specialty_mapping.json"

        if not chunks_dir.exists():
            logger.warning(f"Chunks directory not found: {chunks_dir}")
            logger.warning(f"Skipping {mode} dataset")
            continue

        logger.info(f"Processing {mode} dataset at {chunks_dir}")

        try:
            mapping = generate_specialty_mapping(
                chunks_dir=chunks_dir,
                output_path=output_path,
            )
            logger.success(
                f"Generated specialty mapping for {mode} dataset: "
                f"{len(mapping)} chunks classified"
            )
        except Exception as e:
            logger.error(f"Failed to generate mapping for {mode} dataset: {e}")
            continue

    logger.success("Specialty mapping generation complete!")


if __name__ == "__main__":
    main()
