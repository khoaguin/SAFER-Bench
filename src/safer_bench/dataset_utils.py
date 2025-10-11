"""Dataset utilities for SaferBench."""

from pathlib import Path
from typing_extensions import Optional
from loguru import logger


def get_dataset_path(
    dataset_name: str, use_subset: bool = False, project_root_dir: Optional[Path] = None
) -> Path:
    """Get dataset path based on subset configuration.

    Args:
        dataset_name: Name of the dataset (e.g., 'statpearls', 'textbooks')
        use_subset: Flag indicating whether to use subset or full dataset
        project_root_dir: Optional root directory (defaults to project root)

    Returns:
        Path to the dataset directory
    """
    if project_root_dir is None:
        # Use project root directory (safer-bench/)
        project_root_dir = Path(__file__).parents[2]

    base_dir = project_root_dir / "datasets"

    if use_subset:
        path = base_dir / "subsets" / dataset_name
        logger.debug(f"Using subset datasets. Looking for {dataset_name} in {path}.")
        return path
    else:
        path = base_dir / "full" / dataset_name
        logger.debug(f"Using full datasets. Looking for {dataset_name} in {path}.")
        return path


def validate_dataset_exists(
    dataset_name: str, use_subset: bool = False, project_root_dir: Optional[Path] = None
) -> bool:
    """Check if a dataset exists at the expected path.

    Args:
        dataset_name: Name of the dataset
        use_subset: Flag indicating whether to use subset or full dataset
        project_root_dir: Optional root directory

    Returns:
        True if dataset directory exists, False otherwise
    """
    dataset_path = get_dataset_path(dataset_name, use_subset, project_root_dir)
    return dataset_path.exists() and dataset_path.is_dir()


def list_available_datasets(
    use_subset: bool = False, project_root_dir: Optional[Path] = None
) -> list[str]:
    """List all available datasets in the current mode (subset/full).

    Args:
        use_subset: Flag indicating whether to use subset or full dataset
        project_root_dir: Optional root directory

    Returns:
        List of dataset names
    """
    if project_root_dir is None:
        project_root_dir = Path(__file__).parents[2]

    base_dir = project_root_dir / "datasets"
    mode_dir = base_dir / ("subsets" if use_subset else "full")

    if not mode_dir.exists():
        return []

    return [d.name for d in mode_dir.iterdir() if d.is_dir()]
