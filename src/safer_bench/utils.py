"""Utility functions for SAFER-Bench."""

import torch
from omegaconf import DictConfig, OmegaConf

from loguru import logger


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


def display_config(cfg: DictConfig):
    """Display the loaded configuration in a clean, robust format."""
    logger.info("=" * 80)

    # Highlight dataset mode prominently
    mode = (
        "🚀 SUBSET MODE (Fast)" if cfg.dataset.use_subset else "🔬 FULL MODE (Complete)"
    )
    logger.critical(f"SAFERBENCH CONFIGURATION - {mode}")
    logger.info("=" * 80)

    # Convert to YAML string for clean display
    config_yaml = OmegaConf.to_yaml(cfg, resolve=True)

    # Add emoji sections for better readability
    formatted_config = config_yaml
    formatted_config = formatted_config.replace("dataset:", "💾 dataset:")
    formatted_config = formatted_config.replace("federation:", "🌐 federation:")
    formatted_config = formatted_config.replace("retriever:", "🔍 retriever:")
    formatted_config = formatted_config.replace("merger:", "🔗 merger:")
    formatted_config = formatted_config.replace("llm:", "🤖 llm:")
    formatted_config = formatted_config.replace("privacy:", "🔒 privacy:")
    formatted_config = formatted_config.replace("retrieval:", "📏 retrieval:")
    formatted_config = formatted_config.replace("qa:", "❓ qa:")
    formatted_config = formatted_config.replace("evaluation:", "📊 evaluation:")
    formatted_config = formatted_config.replace("runtime:", "⚙️  runtime:")
    formatted_config = formatted_config.replace("hydra:", "⚙️  hydra:")

    # Log the formatted config
    for line in formatted_config.split("\n"):
        if line.strip():  # Skip empty lines
            logger.info(line)

    logger.info("=" * 80)
