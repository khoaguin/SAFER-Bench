#!/usr/bin/env python3
"""Main entry point for SAFER-Bench pipeline."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from safer_bench.logging import logger, setup_logging
from safer_bench.utils import get_device

# Get the project root directory
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = str(PROJECT_ROOT / "configs")


def display_config(cfg: DictConfig):
    """Display the loaded configuration."""
    logger.info("=" * 80)
    logger.info("SAFER-BENCH CONFIGURATION")
    logger.info("=" * 80)

    logger.info("🌐 FEDERATION:")
    logger.info(f"  Mode: {cfg.federation.setup.mode}")
    logger.info(f"  Aggregator: {cfg.federation.aggregator}")
    logger.info(f"  Data Owners: {', '.join(cfg.federation.datasites)}")
    logger.info(f"  Corpora: {cfg.federation.corpus_names}")

    logger.info("🔍 RETRIEVER:")
    logger.info(f"  Type: {cfg.retriever.type}")
    logger.info(f"  Embedding: {cfg.retriever.embedding_model}")
    logger.info(f"  k-NN: {cfg.retrieval.k_nn}")

    logger.info("🤖 LLM:")
    logger.info(f"  Model: {cfg.llm.model}")
    actual_device = get_device(cfg.llm.device)
    logger.info(f"  Device config: {cfg.llm.device} → Using: {actual_device}")

    logger.info("🔒 PRIVACY:")
    logger.info(f"  Type: {cfg.privacy.type}")

    logger.info("=" * 80)


def setup_local_federation(cfg: DictConfig):
    """Set up local federation with RDS."""
    from pathlib import Path
    from syft_rds.orchestra import setup_rds_server

    logger.info("🚀 Setting up local federation...")

    # Initialize DS server
    ds_stack = setup_rds_server(
        email=cfg.federation.aggregator,
        root_dir=Path(cfg.federation.setup.root_dir),
        key=cfg.federation.setup.network_key,
    )

    # Connect to DOs as guest
    do_clients = []
    for do_email in cfg.federation.datasites:
        client = ds_stack.init_session(host=do_email)
        do_clients.append(client)
        logger.info(f"✅ Connected to {do_email}")

    return ds_stack, do_clients


def run_pipeline(cfg: DictConfig):
    """Run the SAFER-Bench pipeline."""
    setup_logging()

    logger.info("🚀 Starting SAFER-Bench pipeline")

    try:
        # Display configuration
        display_config(cfg)

        if cfg.federation.setup.mode == "local":
            # Set up local federation
            # ds_stack, do_clients = setup_local_federation(cfg)

            # TODO: Add actual pipeline execution here
            logger.info("🔨 Pipeline execution would happen here...")
            logger.info("   - Initialize retrievers")
            logger.info("   - Submit jobs to data owners")
            logger.info("   - Run federated experiments")
            logger.info("   - Collect metrics")
        else:
            logger.info("🌍 Distributed federation mode not yet implemented")

        logger.success("✨ Pipeline completed successfully!")

    except Exception:
        logger.exception("❌ Pipeline failed with unexpected error")
        raise


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra."""
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
