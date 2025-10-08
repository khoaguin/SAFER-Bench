"""
FedRAGProjectAdapter: Prepares FedRAG project with injected benchmark parameters.
Handles configuration injection and project preparation for job submission.
"""

import shutil
import yaml
import tomli
import tomli_w
from pathlib import Path
from typing_extensions import Dict, List, Optional

from omegaconf import DictConfig
from loguru import logger
from syft_flwr.bootstrap import bootstrap as syft_flwr_bootstrap

from safer_bench.models import FederationInfo


class FedRAGProjectAdapter:
    """Adapter for preparing FedRAG projects with benchmark-specific parameters.
    - Copies FedRAG project to a temporary directory
    - Injects retriever configuration (embedding model, dimensions)
    - Updates pyproject.toml with benchmark parameters
    - Configures federation information (DO emails, aggregator)
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the FedRAG adapter.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        # Source FedRAG project in assets
        self.source_fedrag_path = Path(__file__).parent / "assets" / "fedrag_v1"
        if not self.source_fedrag_path.exists():
            raise FileNotFoundError(
                f"FedRAG project not found at {self.source_fedrag_path}"
            )
        # Temporary directory for prepared project
        self.temp_dir = None
        self.prepared_project_path = None
        # Benchmark ID will be set by BenchmarkRunner
        self.benchmark_id = None

    async def prepare_project(
        self, federation_info: Optional[FederationInfo] = None
    ) -> Path:
        """Prepare FedRAG project with injected benchmark parameters.

        Args:
            federation_info: Optional federation information with DO emails

        Returns:
            Path to the prepared FedRAG project
        """
        logger.info("🔧 Preparing FedRAG project with benchmark parameters")

        try:
            # Copy project to temporary directory
            self.prepared_project_path = self._copy_project_to_temp()
            logger.debug(f"Copied FedRAG project to: {self.prepared_project_path}")

            # Update retriever configuration
            self._update_retriever_config(self.prepared_project_path)
            logger.debug("Updated retriever configuration")

            # Update pyproject.toml with benchmark parameters
            self._update_pyproject_config(self.prepared_project_path, federation_info)
            logger.debug("Updated pyproject.toml with benchmark parameters")

            logger.success(
                f"✅ FedRAG project prepared at: {self.prepared_project_path}"
            )
            return self.prepared_project_path

        except Exception as e:
            logger.error(f"Failed to prepare FedRAG project: {e}")
            if self.temp_dir:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise

    def _copy_project_to_temp(self) -> Path:
        """Copy FedRAG project to the output directory.

        Returns:
            Path to the prepared project directory
        """
        # Use Hydra's output directory instead of system temp
        from hydra.core.hydra_config import HydraConfig

        # Get the output directory from Hydra runtime
        hydra_cfg = HydraConfig.get()
        output_dir = Path(hydra_cfg.runtime.output_dir)

        # Create fedrag subdirectory in output directory
        fedrag_project_dir = output_dir / "fedrag_project"

        # Copy entire project
        shutil.copytree(self.source_fedrag_path, fedrag_project_dir)
        logger.debug(f"Created FedRAG project at: {fedrag_project_dir}")

        # Store the directory path for cleanup (though we may not want to clean this up)
        self.temp_dir = str(output_dir)

        return fedrag_project_dir

    def _update_retriever_config(self, fedrag_path: Path) -> None:
        """Update retriever.yaml with benchmark-specific embedding configuration.

        Args:
            fedrag_path: Path to the FedRAG project
        """
        retriever_yaml_path = fedrag_path / "fedrag" / "retriever.yaml"
        if not retriever_yaml_path.exists():
            raise FileNotFoundError(
                f"Retriever config not found at {retriever_yaml_path}"
            )

        # Prepare retriever configuration
        retriever_config = {
            "embedding_model": self.cfg.retriever.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "embedding_dimension": self.cfg.retriever.get("embedding_dimension", 384),
        }

        # Add additional retriever parameters if configured
        if "type" in self.cfg.retriever:
            retriever_config["retriever_type"] = self.cfg.retriever.type
        if "similarity_metric" in self.cfg.retrieval:
            retriever_config["similarity_metric"] = self.cfg.retrieval.similarity_metric

        # Write updated configuration
        with open(retriever_yaml_path, "w") as f:
            yaml.dump(retriever_config, f, default_flow_style=False)

        logger.debug(f"Updated retriever config: {retriever_config}")
        logger.debug(f"Updated retriever config path: {retriever_yaml_path}")

    def _update_pyproject_config(
        self, fedrag_path: Path, federation_info: Optional[Dict]
    ):
        """Update pyproject.toml with benchmark parameters and federation info.

        Args:
            fedrag_path: Path to the FedRAG project
            federation_info: Optional federation information with DO emails
        """
        pyproject_path = fedrag_path / "pyproject.toml"

        # Load existing configuration
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)

        # Update [tool.flwr.app.config] section with benchmark parameters
        self._update_flwr_app_config(pyproject, federation_info)

        # Update number of supernodes based on data owners
        if federation_info and "data_owners" in federation_info:
            federations = (
                pyproject.setdefault("tool", {})
                .setdefault("flwr", {})
                .setdefault("federations", {})
            )
            num_dos = len(federation_info.data_owners)
            federations.setdefault("local-simulation", {}).setdefault("options", {})[
                "num-supernodes"
            ] = num_dos

        # Save updated configuration (without syft_flwr section - that will be handled by bootstrap)
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(pyproject, f)

        # Use syft-flwr's bootstrap to properly configure the [tool.syft_flwr] section
        if federation_info:
            self._bootstrap_syft_flwr(fedrag_path, federation_info)

    def _update_flwr_app_config(
        self, pyproject: Dict, federation_info: Optional[FederationInfo]
    ):
        """Update [tool.flwr.app.config] section with benchmark parameters.

        Args:
            pyproject: Loaded pyproject.toml configuration dictionary
            federation_info: Optional federation information with DO emails
        """
        app_config = (
            pyproject.setdefault("tool", {})
            .setdefault("flwr", {})
            .setdefault("app", {})
            .setdefault("config", {})
        )

        # QA configuration
        app_config["server-qa-datasets"] = self.cfg.qa.get(
            "datasets", "pubmedqa|bioasq"
        )
        app_config["server-qa-num"] = self.cfg.qa.get("num_questions", 10)

        # Retrieval configuration
        app_config["k-nn"] = self.cfg.retrieval.get("k_nn", 8)
        app_config["chunk-size"] = self.cfg.retrieval.get("chunk_size", 512)
        app_config["chunk-overlap"] = self.cfg.retrieval.get("chunk_overlap", 50)

        # Corpus configuration (from federation)
        if federation_info:
            corpus_names = "|".join([do.dataset for do in federation_info.data_owners])
            app_config["clients-corpus-names"] = corpus_names

        # Merger configuration
        merger_type = self.cfg.merger.get("type", "rrf")
        if merger_type == "rrf":
            app_config["k-rrf"] = self.cfg.merger.get("k_rrf", 60)
        app_config["merger-type"] = merger_type

        # LLM configuration
        llm_config = self.cfg.get("llm", {})
        if "model" in llm_config:
            # Map from config model name to HuggingFace path
            llm_model_map = {
                "qwen_small": "Qwen/Qwen2.5-0.5B-Instruct",
                "smollm_360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "biomistral_7b": "BioMistral/BioMistral-7B",
                "llama_1b": "meta-llama/Llama-3.2-1B-Instruct",
            }
            model_name = llm_config.get("model", "qwen_small")
            app_config["server-llm-hfpath"] = llm_model_map.get(
                model_name,
                llm_config.get("model_path", "HuggingFaceTB/SmolLM2-360M-Instruct"),
            )

        app_config["server-llm-use-gpu"] = str(llm_config.get("use_gpu", False)).lower()

    def _bootstrap_syft_flwr(
        self,
        fedrag_path: Path,
        federation_info: FederationInfo,
        datasites: List[str] = None,
    ):
        """Use syft-flwr's bootstrap to configure federation information.
        This will create the [tool.syft_flwr] section in pyproject.toml.

        Args:
            fedrag_path: Path to the FedRAG project
            federation_info: Federation information with aggregator and data owners
            datasites: Optional list of datasite emails to use. If not provided, uses all DOs from federation_info.
        """
        try:
            # Extract aggregator and datasites from federation_info
            aggregator = federation_info.aggregator
            if datasites is None:
                datasites = [do.email for do in federation_info.data_owners]

            if not aggregator or not datasites:
                logger.warning(
                    "Missing aggregator or datasites for syft-flwr bootstrap"
                )
                return

            # Remove main.py if it exists (bootstrap will recreate it)
            main_py_path = fedrag_path / "main.py"
            if main_py_path.exists():
                main_py_path.unlink()
                logger.debug("Removed existing main.py for bootstrap")

            # Use syft-flwr bootstrap to properly configure the project
            syft_flwr_bootstrap(
                flwr_project_dir=fedrag_path, aggregator=aggregator, datasites=datasites
            )

            logger.debug(
                f"Successfully bootstrapped syft-flwr with aggregator: {aggregator}, datasites: {datasites}"
            )

        except Exception as e:
            logger.error(f"Failed to bootstrap syft-flwr: {e}")
            # Continue without bootstrap - fallback to manual configuration if needed
            pass

    def update_datasites_for_approved_jobs(
        self, fedrag_path: Path, approved_do_emails: List[str]
    ):
        """Update pyproject.toml datasites list with only approved DOs.

        This manually edits the [tool.syft_flwr].datasites list without re-bootstrapping,
        which preserves the app_name and keeps job UIDs valid.

        Args:
            fedrag_path: Path to the FedRAG project
            approved_do_emails: List of approved data owner emails
        """
        logger.info(
            f"Updating pyproject.toml with {len(approved_do_emails)} approved DOs"
        )

        pyproject_path = fedrag_path / "pyproject.toml"
        if not pyproject_path.exists():
            logger.error(f"pyproject.toml not found at {pyproject_path}")
            return

        try:
            # Read existing pyproject.toml
            with open(pyproject_path, "rb") as f:
                pyproject = tomli.load(f)

            # Update datasites list in [tool.syft_flwr] section
            if "tool" in pyproject and "syft_flwr" in pyproject["tool"]:
                pyproject["tool"]["syft_flwr"]["datasites"] = approved_do_emails
                logger.debug(f"Updated datasites: {approved_do_emails}")
            else:
                logger.warning("[tool.syft_flwr] section not found in pyproject.toml")
                return

            # Write back to file
            with open(pyproject_path, "wb") as f:
                tomli_w.dump(pyproject, f)

            logger.success("✅ Updated pyproject.toml with approved DOs")

        except Exception as e:
            logger.error(f"Failed to update pyproject.toml: {e}")
            raise

    def cleanup(self):
        """Clean up temporary directories."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
