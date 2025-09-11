#!/usr/bin/env python3
"""Main entry point for SAFER-Bench pipeline."""

import argparse
from pathlib import Path

from safer_bench.logging import logger, setup_logging
from safer_bench.config.benchmark_config import BenchmarkConfig


def display_config(config: BenchmarkConfig):
    """Display the loaded configuration."""
    logger.info("=" * 80)
    logger.info("SAFER-BENCH CONFIGURATION")
    logger.info("=" * 80)

    logger.info("📋 BASELINE CONFIGURATION:")
    logger.info("-" * 40)
    for field, value in config.baseline.model_dump().items():
        logger.info(f"  {field:<20}: {value}")

    logger.info("🔄 DIMENSIONS TO VARY:")
    logger.info("-" * 40)
    for dimension, values in config.dimensions_to_vary.items():
        logger.info(f"  {dimension:<20}: {values}")

    logger.info("=" * 80)


def run_pipeline(config_path: Path):
    """Run the SAFER-Bench pipeline."""
    setup_logging()

    # Add context for this pipeline run
    with logger.contextualize(config_file=str(config_path)):
        logger.info("🚀 Starting SAFER-Bench pipeline")
        logger.debug(f"Configuration path: {config_path.absolute()}")

        try:
            # Load configuration
            logger.info("📁 Loading configuration...")
            config = BenchmarkConfig.from_yaml(config_path)
            logger.success("✅ Configuration loaded successfully")

            # Display configuration
            display_config(config)

            # TODO: Add actual pipeline execution here
            logger.info("🔨 Pipeline execution would happen here...")
            logger.info("   - Initialize retrievers")
            logger.info("   - Setup data owners")
            logger.info("   - Run experiments")
            logger.info("   - Collect metrics")

            logger.success("✨ Pipeline completed successfully!")

        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {config_path}")
            raise
        except Exception:
            logger.exception("❌ Pipeline failed with unexpected error")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SAFER-Bench Pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline.yaml"),
        help="Path to configuration file (default: configs/baseline.yaml)",
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"❌ Configuration file not found: {args.config}")
        return 1

    run_pipeline(args.config)
    return 0


if __name__ == "__main__":
    exit(main())
