"""
FederationManager: Manages the setup and lifecycle of the federated environment.
Handles SyftBox network initialization and data owner registration.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional

from omegaconf import DictConfig
from loguru import logger
from pydantic import BaseModel, Field, EmailStr, field_validator

from syft_rds.orchestra import setup_rds_server, remove_rds_stack_dir, SingleRDSStack
from syft_core import Client as SyftBoxClient


SAFER_BENCH_SYFTBOX_NETWORK = "safer_bench_network"


class DataOwnerInfo(BaseModel):
    """Information about a data owner in the federation."""

    email: EmailStr
    dataset: str = Field(..., min_length=1, description="Dataset name")
    data_fraction: float = Field(
        ..., gt=0.0, le=1.0, description="Fraction of dataset (0-1]"
    )

    @field_validator("dataset")
    @classmethod
    def validate_dataset(cls, v: str) -> str:
        """Ensure dataset name is valid."""
        valid_datasets = ["statpearls", "textbooks", "mimic-iv-note", "mimic-iv-bhc"]
        if v not in valid_datasets:
            raise ValueError(
                f"Invalid dataset name: {v}. Must be one of {valid_datasets}."
            )
        return v


class FederationManager:
    """Manages federation setup for SaferBench."""

    def __init__(self, cfg: DictConfig):
        """Initialize the federation manager.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.network_key = SAFER_BENCH_SYFTBOX_NETWORK
        # Use project root directory (safer-bench/) regardless of where script is run from
        self.root_dir = Path(__file__).parents[2]

        # Parse federation configuration
        self.aggregator_email = cfg.federation.aggregator
        self.data_owners = self._parse_data_owners()  # List[DataOwnerInfo]
        self.use_subset = cfg.dataset.use_subset  # Cache for easy access

        # Runtime state tracked separately
        self.ds_stack: Optional[SingleRDSStack] = None
        self.ds_client: Optional[SyftBoxClient] = None
        self.do_stacks: Dict[str, SingleRDSStack] = {}  # email -> stack
        self.do_clients: Dict[str, SyftBoxClient] = {}  # email -> client
        self.is_setup = False

    # Public Methods
    async def setup(self) -> Dict[str, Any]:
        """Setup the federated environment with SyftBox.

        Returns:
            Dictionary containing federation setup information
        """
        logger.info(
            f"🔧 Setting up federation with {len(self.data_owners)} data owners"
        )

        try:
            # Clean up any previous runs
            await self.cleanup()

            # Setup Data Scientist (aggregator)
            logger.info(f"Setting up aggregator: {self.aggregator_email}")
            self.ds_stack = await self._setup_aggregator()

            # Setup Data Owners
            logger.info(f"Setting up {len(self.data_owners)} data owners")
            await self._setup_data_owners()

            # Verify connections
            await self._verify_federation()

            self.is_setup = True

            federation_info = {
                "ds": self.ds_stack,
                "ds_client": self.ds_client,
                "dos": list(self.do_stacks.values()),
                "clients": list(self.do_clients.values()),
                "data_owners": self.data_owners,
                "aggregator": self.aggregator_email,
                "network_key": self.network_key,
            }

            logger.success("✅ Federation setup complete")
            self._log_federation_summary(federation_info)

            return federation_info

        except Exception as e:
            logger.error(f"Failed to setup federation: {e}")
            await self.cleanup()
            raise

    async def cleanup(self):
        """Clean up federation resources and reset state.

        Safe to call multiple times - will clean up any existing resources
        and reset the federation state.
        """
        logger.debug("🧹 Cleaning up federation resources")

        try:
            # Clean up RDS stacks (safe to call even if none exist)
            try:
                remove_rds_stack_dir(root_dir=self.root_dir, key=self.network_key)
                logger.debug("Cleaned up SyftBox directories")
            except Exception as e:
                logger.debug(f"No SyftBox directories to clean: {e}")

            # Clean up .syftbox directory
            syftbox_dir = self.root_dir / ".syftbox"
            if syftbox_dir.exists():
                import shutil

                shutil.rmtree(syftbox_dir, ignore_errors=True)
                logger.debug("Cleaned up .syftbox directory")

            # Reset runtime state
            self.ds_stack = None
            self.ds_client = None
            self.do_stacks.clear()
            self.do_clients.clear()
            self.is_setup = False

            logger.debug("Federation cleanup complete")

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def get_guest_clients(self) -> List[SyftBoxClient]:
        """Get guest client connections from DS to all DOs.

        Returns:
            List of guest client connections
        """
        if not self.is_setup:
            raise RuntimeError("Federation not setup. Call setup() first.")

        guest_clients = []
        for do_info in self.data_owners:
            guest_client = self.ds_stack.init_session(host=do_info.email)
            guest_clients.append(guest_client)

        return guest_clients

    async def wait_for_jobs(self, timeout: int = 60):
        """Wait for all data owners to be ready to receive jobs.

        Args:
            timeout: Maximum time to wait in seconds
        """
        logger.info("Waiting for data owners to be ready...")
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            all_ready = True

            for do_info in self.data_owners:
                try:
                    # Check if DO is responsive
                    do_client = self.do_clients[do_info.email]
                    do_client.dataset.get_all()
                    logger.debug(f"✅ {do_info.email} is ready")
                except Exception:
                    all_ready = False
                    logger.debug(f"⏳ Waiting for {do_info.email}")
                    break

            if all_ready:
                logger.success("All data owners are ready")
                return

            await asyncio.sleep(2)

        raise TimeoutError(f"Data owners not ready after {timeout} seconds")

    # Private Methods
    def _parse_data_owners(self) -> List[DataOwnerInfo]:
        """Parse data owner configurations from config.

        Returns:
            List of DataOwnerInfo objects
        """
        data_owners = []

        # Dynamically get all DO configurations (do1, do2, do3, etc.)
        for i in range(1, self.cfg.federation.num_data_owners + 1):
            do_config = getattr(self.cfg.federation, f"do{i}")
            data_owners.append(
                DataOwnerInfo(
                    email=do_config.email,
                    dataset=do_config.dataset,
                    data_fraction=do_config.data_fraction,
                )
            )

        return data_owners

    async def _setup_aggregator(self) -> SingleRDSStack:
        """Setup the data scientist/aggregator node.

        Returns:
            SingleRDSStack for the aggregator
        """
        # Setup RDS server for aggregator
        ds_stack = setup_rds_server(
            email=self.aggregator_email, root_dir=self.root_dir, key=self.network_key
        )

        # Initialize client session
        self.ds_client = ds_stack.init_session(host=self.aggregator_email)

        # Set environment variable for SyftBox client config
        os.environ["SYFTBOX_CLIENT_CONFIG_PATH"] = str(ds_stack.client.config_path)

        logger.debug(f"Aggregator setup complete: {self.aggregator_email}")
        return ds_stack

    async def _setup_data_owners(self):
        """Setup all data owner nodes."""
        tasks = []

        for do_info in self.data_owners:
            task = self._setup_single_do(do_info)
            tasks.append(task)

        # Setup all DOs in parallel
        await asyncio.gather(*tasks)

        logger.debug(f"All {len(self.data_owners)} data owners setup complete")

    async def _setup_single_do(self, do_info: DataOwnerInfo):
        """Setup a single data owner node.

        Args:
            do_info: DataOwnerInfo object for the data owner
        """
        # Setup RDS server for DO
        do_stack = setup_rds_server(
            email=do_info.email, root_dir=self.root_dir, key=self.network_key
        )

        # Initialize client session
        do_client = do_stack.init_session(host=do_info.email)

        # Store stack and client in dictionaries
        self.do_stacks[do_info.email] = do_stack
        self.do_clients[do_info.email] = do_client

        logger.debug(
            f"Data owner setup complete: {do_info.email} with dataset {do_info.dataset}"
        )

    async def _verify_federation(self):
        """Verify that all nodes can communicate."""
        logger.debug("Verifying federation connectivity")

        # DS should be able to connect to all DOs as guest
        for do_info in self.data_owners:
            try:
                # Try to connect from DS to DO as guest
                guest_client = self.ds_stack.init_session(host=do_info.email)
                assert not guest_client.is_admin, f"Should be guest on {do_info.email}"
                logger.debug(f"✅ DS can connect to {do_info.email} as guest")
            except Exception as e:
                logger.warning(f"⚠️ DS cannot connect to {do_info.email}: {e}")

        # Each DO should be admin on their own datasite
        for do_info in self.data_owners:
            do_client = self.do_clients[do_info.email]
            assert (
                do_client.is_admin
            ), f"{do_info.email} should be admin on their datasite"
            logger.debug(f"✅ {do_info.email} is admin on their datasite")

    def _log_federation_summary(self, federation_info: Dict):
        """Log a summary of the federation setup.

        Args:
            federation_info: Federation setup information
        """
        logger.info("=" * 60)
        logger.info("Federation Summary:")
        logger.info(f"  Aggregator: {self.aggregator_email}")
        logger.info(f"  Network Key: {self.network_key}")
        logger.info(f"  Network Directory: {self.root_dir / self.network_key}")
        logger.info(f"  Data Owners ({len(self.data_owners)}):")

        for i, do_info in enumerate(self.data_owners, 1):
            logger.info(f"    DO{i}: {do_info.email}")
            logger.info(f"         Dataset: {do_info.dataset}")
            logger.info(f"         Data Fraction: {do_info.data_fraction * 100}%")

        logger.info("=" * 60)
