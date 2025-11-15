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
        datasets: Dict[str, float],
        output_path: Path,
        use_subset: bool = False,
        project_root_dir: Optional[Path] = None,
        partition_id: Optional[int] = None,
    ) -> None:
        """Create a hybrid partition combining portions of multiple datasets.

        Args:
            datasets: Dict mapping dataset_name to proportion (e.g., {"statpearls": 0.5, "textbooks": 0.5})
            output_path: Path to output partition directory
            use_subset: Whether to use subset or full datasets
            project_root_dir: Optional root directory
            partition_id: Unique ID for this partition (used for deterministic sampling)
        """
        logger.info(f"Creating hybrid partition with datasets: {datasets}")

        # Create output directories
        output_chunk_dir = output_path / "chunk"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

        all_sampled_chunks = []

        for dataset_name, proportion in datasets.items():
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
            sampled_chunks = self._sample_chunks(
                chunk_files, num_to_sample, partition_id=partition_id
            )

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
        datasets: Dict[str, float],
        topics: List[str],
        output_path: Path,
        use_subset: bool = False,
        project_root_dir: Optional[Path] = None,
    ) -> None:
        """Create a topic-based partition filtering by medical specialties.

        Args:
            datasets: Dict of datasets to filter from (proportion values ignored for topic strategy)
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

        for dataset_name in datasets.keys():
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
        datasets: Dict[str, float],
        output_path: Path,
        use_subset: bool = False,
        project_root_dir: Optional[Path] = None,
    ) -> None:
        """Merge all datasets into a single unified corpus.

        Args:
            datasets: Dict of all datasets to merge (proportion values should be 1.0 for all)
            output_path: Path to output partition directory
            use_subset: Whether to use subset or full datasets
            project_root_dir: Optional root directory
        """
        logger.info(
            f"Creating centralized partition with all datasets: {list(datasets.keys())}"
        )

        # Create output directories
        output_chunk_dir = output_path / "chunk"
        output_chunk_dir.mkdir(parents=True, exist_ok=True)

        all_chunks = []

        for dataset_name in datasets.keys():
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

    def _sample_chunks(
        self,
        chunk_files: List[Path],
        num_to_sample: int,
        partition_id: Optional[int] = None,
    ) -> List[Path]:
        """Sample a specified number of chunk files deterministically.

        Args:
            chunk_files: List of chunk file paths
            num_to_sample: Number of chunks to sample
            partition_id: Optional partition ID for per-partition random seed

        Returns:
            List of sampled chunk file paths
        """
        if num_to_sample >= len(chunk_files):
            return chunk_files

        # Use per-partition random seed to ensure disjoint sampling across partitions
        if partition_id is not None:
            # Create a local Random instance with partition-specific seed
            # This ensures different partitions sample different chunks
            rng = random.Random(self.seed + partition_id)
            sampled = rng.sample(chunk_files, num_to_sample)
        else:
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
        datasets: Dict[str, float],
        strategy: str,
        topics: Optional[List[str]] = None,
    ) -> None:
        """Create mock directory with README and sample chunks.

        Args:
            output_path: Path to partition directory (private/)
            datasets: Dict of dataset names and proportions
            strategy: Distribution strategy name
            topics: Optional list of topics (for topic-based strategy)
        """
        mock_dir = output_path.parent / "mock"
        mock_chunk_dir = mock_dir / "chunk"
        mock_chunk_dir.mkdir(parents=True, exist_ok=True)

        # Copy a few sample chunks from the private directory
        private_chunk_dir = output_path / "chunk"
        chunk_files = list(private_chunk_dir.glob("*.jsonl"))

        # Copy first 10 chunks as samples for mock
        num_samples = min(10, len(chunk_files))
        for chunk_file in chunk_files[:num_samples]:
            dest_file = mock_chunk_dir / chunk_file.name
            shutil.copy2(chunk_file, dest_file)

        logger.debug(f"Copied {num_samples} sample chunks to mock directory")

        # Create README for both mock and private (Syft requires same file extensions)
        mock_readme_content = f"""# Partitioned Dataset (Mock/Preview)

**Distribution Strategy**: {strategy}
**Source Datasets**: {', '.join(datasets.keys())}
**Total Chunks**: {num_samples}
"""

        private_readme_content = f"""# Partitioned Dataset (Private)

**Distribution Strategy**: {strategy}
**Source Datasets**: {', '.join(datasets.keys())}
**Total Chunks**: {len(chunk_files)}
"""

        if topics:
            mock_readme_content += f"**Topics**: {', '.join(topics)}\n"
            private_readme_content += f"**Topics**: {', '.join(topics)}\n"

        mock_readme_content += """
This is a partitioned dataset created for federated RAG benchmarking.

**Note**: This mock directory contains sample chunks for preview purposes.
The full dataset is in the private/ directory.
"""

        private_readme_content += """
This is a partitioned dataset created for federated RAG benchmarking.
The FAISS index will be built automatically by the FedRAG app at runtime.
"""

        # Write both READMEs
        mock_readme_path = mock_dir / "README.md"
        mock_readme_path.write_text(mock_readme_content)

        # Private README goes in the private directory itself (output_path is already private/)
        private_readme_path = output_path / "README.md"
        private_readme_path.write_text(private_readme_content)

        logger.debug(
            f"Created README files: mock={mock_readme_path}, private={private_readme_path}"
        )
