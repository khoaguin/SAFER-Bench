"""Data partitioning utilities for different distribution strategies.

This module handles splitting, merging, and filtering dataset chunks
for various data distribution strategies (hybrid, topic-based, centralized).

FAISS indexes are NOT built here - they are built on-demand by the FedRAG
retriever running on DOs at runtime.
"""

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from safer_bench.dataset_utils import get_dataset_path


class DataPartitioner:
    """Handles data partitioning for different distribution strategies."""

    def __init__(self, seed: int = 42):
        """Initialize the data partitioner.

        Args:
            seed: Random seed for deterministic sampling
        """
        self.seed = seed
        random.seed(seed)
        logger.debug(f"DataPartitioner initialized with seed={seed}")

    def create_hybrid_partition(
        self,
        datasets: List[str],
        proportions: Dict[str, float],
        output_path: Path,
        use_subset: bool = False,
        project_root_dir: Optional[Path] = None,
    ) -> None:
        """Create a hybrid partition combining portions of multiple datasets.

        Args:
            datasets: List of dataset names (e.g., ["statpearls", "textbooks"])
            proportions: Dict mapping dataset_name to proportion (e.g., {"statpearls": 0.5})
            output_path: Path to output partition directory
            use_subset: Whether to use subset or full datasets
            project_root_dir: Optional root directory
        """
        logger.info(f"Creating hybrid partition with datasets: {datasets}")
        logger.info(f"Proportions: {proportions}")

        # Create output directories
        output_chunk_dir = output_path / "chunk"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

        all_sampled_chunks = []

        for dataset_name in datasets:
            proportion = proportions.get(dataset_name, 0.0)
            if proportion <= 0:
                logger.warning(f"Skipping {dataset_name} (proportion={proportion})")
                continue

            # Get source dataset path
            dataset_path = get_dataset_path(dataset_name, use_subset, project_root_dir)
            source_chunk_dir = dataset_path / "private" / "chunk"

            if not source_chunk_dir.exists():
                raise FileNotFoundError(
                    f"Source chunk directory not found: {source_chunk_dir}"
                )

            # Get all chunk files from this dataset
            chunk_files = sorted(source_chunk_dir.glob("*.jsonl"))
            logger.debug(f"{dataset_name}: Found {len(chunk_files)} chunk files")

            # Sample chunks based on proportion
            num_to_sample = max(1, int(len(chunk_files) * proportion))
            sampled_chunks = self._sample_chunks(chunk_files, num_to_sample)

            logger.info(
                f"{dataset_name}: Sampled {len(sampled_chunks)}/{len(chunk_files)} chunks"
            )
            all_sampled_chunks.extend(sampled_chunks)

        # Copy sampled chunks to output directory
        logger.info(f"Copying {len(all_sampled_chunks)} chunks to {output_chunk_dir}")
        self._copy_chunks(all_sampled_chunks, output_chunk_dir)

        # Create mock directory with README
        self._create_mock_dir(output_path, datasets, "hybrid")

        logger.success(f"Hybrid partition created at {output_path}")

    def create_topic_partition(
        self,
        datasets: List[str],
        topics: List[str],
        output_path: Path,
        use_subset: bool = False,
        project_root_dir: Optional[Path] = None,
    ) -> None:
        """Create a topic-based partition filtering by medical specialties.

        Args:
            datasets: List of dataset names
            topics: List of medical topics/specialties (e.g., ["cardiology", "neurology"])
            output_path: Path to output partition directory
            use_subset: Whether to use subset or full datasets
            project_root_dir: Optional root directory
        """
        logger.info(f"Creating topic-based partition with topics: {topics}")

        # Create output directories
        output_chunk_dir = output_path / "chunk"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

        all_filtered_chunks = []

        for dataset_name in datasets:
            # Get source dataset path
            dataset_path = get_dataset_path(dataset_name, use_subset, project_root_dir)
            source_chunk_dir = dataset_path / "private" / "chunk"

            if not source_chunk_dir.exists():
                raise FileNotFoundError(
                    f"Source chunk directory not found: {source_chunk_dir}"
                )

            # Get all chunk files
            chunk_files = sorted(source_chunk_dir.glob("*.jsonl"))
            logger.debug(f"{dataset_name}: Found {len(chunk_files)} chunk files")

            # Filter chunks by topics
            filtered_chunks = self._filter_chunks_by_topic(chunk_files, topics)

            logger.info(
                f"{dataset_name}: Filtered {len(filtered_chunks)}/{len(chunk_files)} chunks "
                f"matching topics {topics}"
            )
            all_filtered_chunks.extend(filtered_chunks)

        if not all_filtered_chunks:
            logger.warning(f"No chunks found matching topics: {topics}")
            raise ValueError(f"No chunks found matching topics: {topics}")

        # Copy filtered chunks to output directory
        logger.info(f"Copying {len(all_filtered_chunks)} chunks to {output_chunk_dir}")
        self._copy_chunks(all_filtered_chunks, output_chunk_dir)

        # Create mock directory with README
        self._create_mock_dir(output_path, datasets, "topic-based", topics)

        logger.success(f"Topic-based partition created at {output_path}")

    def create_centralized_partition(
        self,
        datasets: List[str],
        output_path: Path,
        use_subset: bool = False,
        project_root_dir: Optional[Path] = None,
    ) -> None:
        """Merge all datasets into a single unified corpus.

        Args:
            datasets: List of all dataset names to merge
            output_path: Path to output partition directory
            use_subset: Whether to use subset or full datasets
            project_root_dir: Optional root directory
        """
        logger.info(f"Creating centralized partition with all datasets: {datasets}")

        # Create output directories
        output_chunk_dir = output_path / "chunk"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = []

        for dataset_name in datasets:
            # Get source dataset path
            dataset_path = get_dataset_path(dataset_name, use_subset, project_root_dir)
            source_chunk_dir = dataset_path / "private" / "chunk"

            if not source_chunk_dir.exists():
                raise FileNotFoundError(
                    f"Source chunk directory not found: {source_chunk_dir}"
                )

            # Get all chunk files from this dataset
            chunk_files = sorted(source_chunk_dir.glob("*.jsonl"))
            logger.info(f"{dataset_name}: Found {len(chunk_files)} chunk files")

            all_chunks.extend(chunk_files)

        # Copy all chunks to output directory
        logger.info(f"Copying {len(all_chunks)} chunks to {output_chunk_dir}")
        self._copy_chunks(all_chunks, output_chunk_dir)

        # Create mock directory with README
        self._create_mock_dir(output_path, datasets, "centralized")

        logger.success(f"Centralized partition created at {output_path}")

    def _sample_chunks(self, chunk_files: List[Path], num_to_sample: int) -> List[Path]:
        """Sample a specified number of chunk files deterministically.

        Args:
            chunk_files: List of chunk file paths
            num_to_sample: Number of chunks to sample

        Returns:
            List of sampled chunk file paths
        """
        if num_to_sample >= len(chunk_files):
            return chunk_files

        # Deterministic sampling using seed
        sampled = random.sample(chunk_files, num_to_sample)
        return sampled

    def _copy_chunks(self, chunk_files: List[Path], output_dir: Path) -> None:
        """Copy chunk files to output directory.

        To avoid naming conflicts when merging chunks from multiple datasets,
        we prefix each chunk with the dataset name.

        Args:
            chunk_files: List of source chunk file paths
            output_dir: Destination directory
        """
        for chunk_file in chunk_files:
            # Get parent directory name (dataset name)
            # Path structure: .../datasets/full/{dataset}/private/chunk/{file}.jsonl
            dataset_name = chunk_file.parents[2].name

            # Create prefixed filename to avoid conflicts
            # e.g., statpearls_article-101404.jsonl
            prefixed_name = f"{dataset_name}_{chunk_file.name}"
            dest_file = output_dir / prefixed_name

            shutil.copy2(chunk_file, dest_file)

    def _filter_chunks_by_topic(
        self, chunk_files: List[Path], topics: List[str]
    ) -> List[Path]:
        """Filter chunks based on topic keywords in title/content.

        Args:
            chunk_files: List of chunk file paths
            topics: List of medical topics/specialties

        Returns:
            List of chunk files matching the topics
        """
        # Get keywords for all topics
        all_keywords = []
        for topic in topics:
            keywords = self._get_topic_keywords(topic)
            all_keywords.extend(keywords)

        logger.debug(f"Filtering with keywords: {all_keywords}")

        filtered_chunks = []

        for chunk_file in chunk_files:
            # Read the chunk file and check if any line contains topic keywords
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()  # Case-insensitive matching

                    # Check if any keyword appears in the file
                    if any(keyword.lower() in content for keyword in all_keywords):
                        filtered_chunks.append(chunk_file)
            except Exception as e:
                logger.warning(f"Error reading {chunk_file}: {e}")
                continue

        return filtered_chunks

    def _get_topic_keywords(self, topic: str) -> List[str]:
        """Map medical specialty to keyword list.

        Args:
            topic: Medical specialty name

        Returns:
            List of keywords associated with the topic
        """
        # Medical specialty keyword mapping
        # This can be expanded based on domain expertise
        topic_keywords = {
            "cardiology": [
                "heart",
                "cardiac",
                "coronary",
                "cardiovascular",
                "myocardial",
            ],
            "pulmonology": [
                "lung",
                "pulmonary",
                "respiratory",
                "breathing",
                "bronch",
                "pneumo",
            ],
            "neurology": [
                "brain",
                "neural",
                "neurol",
                "cognitive",
                "cerebral",
                "nervous system",
            ],
            "gastroenterology": [
                "gastro",
                "digestive",
                "intestinal",
                "stomach",
                "colon",
                "hepatic",
            ],
            "oncology": [
                "cancer",
                "tumor",
                "oncol",
                "malignant",
                "carcinoma",
                "neoplasm",
            ],
            "endocrinology": [
                "hormone",
                "endocrine",
                "diabetes",
                "thyroid",
                "metabolic",
            ],
            "nephrology": ["kidney", "renal", "nephro", "urinary"],
            "hematology": ["blood", "hematol", "anemia", "leukemia", "lymphoma"],
            "rheumatology": ["arthritis", "rheumat", "autoimmune", "joint"],
            "dermatology": ["skin", "dermat", "cutaneous", "rash"],
        }

        keywords = topic_keywords.get(topic.lower(), [])

        if not keywords:
            logger.warning(
                f"No keywords defined for topic '{topic}'. Using topic name as keyword."
            )
            keywords = [topic.lower()]

        return keywords

    def _create_mock_dir(
        self,
        output_path: Path,
        datasets: List[str],
        strategy: str,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Create mock directory with README describing the partition.

        Args:
            output_path: Path to partition directory
            datasets: List of dataset names included
            strategy: Distribution strategy name
            topics: Optional list of topics (for topic-based strategy)
        """
        mock_dir = output_path.parent / "mock"
        mock_dir.mkdir(parents=True, exist_ok=True)

        readme_content = f"""# Partitioned Dataset

**Distribution Strategy**: {strategy}
**Source Datasets**: {', '.join(datasets)}
"""

        if topics:
            readme_content += f"**Topics**: {', '.join(topics)}\n"

        readme_content += """
This is a partitioned dataset created for federated RAG benchmarking.
The FAISS index will be built automatically by the FedRAG app at runtime.

**Note**: This mock directory is for documentation purposes only.
"""

        readme_path = mock_dir / "README.md"
        readme_path.write_text(readme_content)
        logger.debug(f"Created mock README at {readme_path}")
