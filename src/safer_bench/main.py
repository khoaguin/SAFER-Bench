#!/usr/bin/env python3
"""Main entry point for SaferBench federated RAG benchmarking."""

import asyncio
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from loguru import logger

from safer_bench.benchmark_runner import BenchmarkRunner
from safer_bench.utils import display_config

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = str(PROJECT_ROOT / "configs")


def setup_logging(output_dir: Path):
    """Configure logging for SaferBench.

    Args:
        output_dir: Directory for log files
    """
    # Add file handler to loguru
    logger.add(
        output_dir / "saferbench.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}",
        level="DEBUG",
    )

    # Disable verbose logs from syft libraries
    logger.disable("syft_rds.client.rds_client")
    logger.disable("syft_rds.client.rds_clients.runtime")
    logger.disable("syft_event.handlers")
    logger.disable("syft_rds.orchestra")
    logger.disable("syft_event.cleanup")


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
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)

    # Setup logging
    setup_logging(output_dir)

    # Run the async benchmark
    asyncio.run(run_benchmark(cfg))


if __name__ == "__main__":
    main()
