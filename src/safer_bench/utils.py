"""Utility functions for SAFER-Bench."""

import torch

from safer_bench.logging import logger


def get_device(device_config: str = "auto") -> str:
    """
    Get the best available device for computation.

    Args:
        device_config: Device configuration from config file.
                    Can be "auto", "cuda", "mps", "cpu", etc.

    Returns:
        Device string to use with PyTorch/Transformers
    """
    if device_config != "auto":
        # User specified a specific device
        logger.debug(f"Using specified device: {device_config}")
        return device_config

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("🎮 Using CUDA GPU for acceleration")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        logger.info("💻 Using CPU (no GPU acceleration available)")

    return device
