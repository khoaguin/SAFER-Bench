#!/usr/bin/env python3
"""Main entry point for SaferBench federated RAG benchmarking."""

import asyncio
from pathlib import Path

import hydra
from omegaconf import DictConfig
from loguru import logger

from safer_bench.benchmark_runner import BenchmarkRunner

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = str(PROJECT_ROOT / "configs")


def display_config(cfg: DictConfig):
    """Display the loaded configuration in a clean, robust format."""
    from omegaconf import OmegaConf

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
    formatted_config = formatted_config.replace("hydra:", "⚙️  hydra:")

    # Log the formatted config
    for line in formatted_config.split("\n"):
        if line.strip():  # Skip empty lines
            logger.info(line)

    logger.info("=" * 80)


async def run_benchmark(cfg: DictConfig):
    """Run the SaferBench federated RAG benchmark."""
    logger.info("🚀 Starting SaferBench federated RAG benchmark")

    try:
        # Display configuration
        display_config(cfg)

        # Create and run benchmark
        runner = BenchmarkRunner(cfg)
        metrics = await runner.run()

        logger.success("✨ Benchmark completed successfully!")
        logger.info(
            f"📊 Results: {metrics.get('benchmark_metadata', {}).get('benchmark_id', 'N/A')}"
        )

        return metrics

    except Exception as e:
        logger.exception(f"❌ Benchmark failed: {e}")
        raise


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra."""
    # Add file logging to capture all logs in the output directory
    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    # Add file handler to loguru
    logger.add(
        output_dir / "saferbench.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    # Run the async benchmark
    asyncio.run(run_benchmark(cfg))


if __name__ == "__main__":
    main()
