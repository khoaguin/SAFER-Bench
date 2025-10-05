"""
FederationManager: Manages the setup and lifecycle of the federated environment.
Handles SyftBox network initialization and data owner registration.
"""

import os
import asyncio
import random
import concurrent.futures
from pathlib import Path
from typing_extensions import Dict, List, Any, Optional

from omegaconf import DictConfig
from loguru import logger

from syft_core import Client as SyftBoxClient
from syft_rds.orchestra import setup_rds_server, remove_rds_stack_dir, SingleRDSStack
from syft_rds.client.rds_client import RDSClient

from safer_bench.dataset_utils import get_dataset_path, validate_dataset_exists
from safer_bench.models import (
    DataOwnerInfo,
    JobInfo,
    JobProcessingStatus,
    JobProcessingResult,
    DatasetUploadStatus,
    DatasetUploadInfo,
    DatasetUploadResult,
    FederationInfo,
    DSServerResult,
    FedRAGExecutionResult,
)


SAFER_BENCH_SYFTBOX_NETWORK = "safer_bench_network"


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
        self.aggregator_email: str = cfg.federation.aggregator
        self.data_owners: List[DataOwnerInfo] = self._parse_data_owners()
        self.use_subset: bool = cfg.dataset.use_subset
        self.benchmark_id: Optional[str] = None  # Set by BenchmarkRunner

        # Runtime state tracked separately
        self.ds_stack: Optional[SingleRDSStack] = None
        self.ds_client: Optional[RDSClient] = None
        self.do_stacks: Dict[
            str, SingleRDSStack
        ] = {}  # { client_email: SingleRDSStack }
        self.do_clients: Dict[str, RDSClient] = {}  # { client_email: RDSClient }
        self.is_setup = False

    # Public Methods
    async def setup_federation(self) -> FederationInfo:
        """Setup the federated environment with SyftBox, including:
        DOs who owns the datasets and DS who wants to run FedRAG jobs on them.

        Returns:
            FederationInfo containing federation setup information
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

            federation_info = FederationInfo(
                benchmark_id=self.benchmark_id or "unknown",
                data_owners=self.data_owners,
                aggregator=self.aggregator_email,
                network_key=self.network_key,
                num_data_owners=len(self.data_owners),
            )

            logger.success("✅ Federation setup complete")
            logger.debug(
                f"Federation info: {federation_info.model_dump_json(indent=2)}"
            )

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

    async def dos_upload_datasets(self) -> DatasetUploadResult:
        """Upload datasets to all data owners concurrently.

        Returns:
            DatasetUploadResult containing upload results with successful and failed uploads
        """
        if not self.is_setup:
            raise RuntimeError("Federation not setup. Call setup() first.")

        logger.info(
            f"📤 Starting concurrent dataset uploads for {len(self.data_owners)} data owners"
        )

        # Create upload tasks for all data owners
        tasks = []
        for do_info in self.data_owners:
            task = self._upload_single_dataset(do_info)
            tasks.append(task)

        # Execute all uploads concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process and categorize results
        return self._process_upload_results(results)

    async def ds_submits_jobs(
        self, fedrag_project_path: Path, benchmark_id: str
    ) -> List[JobInfo]:
        """Submit FedRAG jobs to all data owners.

        Args:
            fedrag_project_path: Path to the prepared FedRAG project
            benchmark_id: Unique identifier for this benchmark run

        Returns:
            List of job submission results with job objects and metadata
        """
        if not self.is_setup:
            raise RuntimeError("Federation not setup. Call setup() first.")

        logger.info(f"📨 Submitting FedRAG jobs to {len(self.data_owners)} data owners")

        jobs_info: List[JobInfo] = []
        for do_info in self.data_owners:
            try:
                # Get DO client (as guest from DS perspective)
                guest_client = self.ds_stack.init_session(host=do_info.email)

                logger.info(
                    f"Submitting job to {do_info.email} for dataset {do_info.dataset}"
                )

                # Submit job using syft_rds client
                job = guest_client.job.submit(
                    name=f"fedrag_benchmark_{benchmark_id}",
                    user_code_path=fedrag_project_path,
                    dataset_name=do_info.dataset,
                    entrypoint="main.py",
                )

                job_info = JobInfo(
                    job=job,
                    do_email=do_info.email,
                    dataset=do_info.dataset,
                    client=guest_client,
                    status=JobProcessingStatus.submission_succeeded,  # Jobs start pending review
                    data_fraction=do_info.data_fraction,
                    benchmark_id=benchmark_id,
                )
                jobs_info.append(job_info)

                logger.success(f"✅ Job submitted to {do_info.email}")

            except Exception as e:
                logger.error(f"❌ Failed to submit job to {do_info.email}: {e}")
                job_info = JobInfo(
                    job=None,
                    do_email=do_info.email,
                    dataset=do_info.dataset,
                    client=None,
                    status=JobProcessingStatus.submission_failed,  # Job submission failed
                    error=str(e),
                    benchmark_id=benchmark_id,
                )
                jobs_info.append(job_info)

        # Log submission summary
        submitted_count = sum(
            1 for j in jobs_info if j.status == JobProcessingStatus.submission_succeeded
        )
        logger.info(
            f"📊 Job submission complete: {submitted_count}/{len(jobs_info)} successful"
        )

        return jobs_info

    async def dos_process_jobs(
        self, jobs_info: List[JobInfo], approval_rate: float
    ) -> JobProcessingResult:
        """DOs approve / reject jobs based on configured approval rate.

        Args:
            jobs_info: List of submitted JobInfo objects
            approval_rate: Percentage of jobs to approve (0.0 to 1.0)

        Returns:
            JobProcessingResult with approval results and statistics
        """
        if not self.is_setup:
            raise RuntimeError("Federation not setup. Call setup() first.")

        logger.info(
            f"📋 Processing job approvals with {approval_rate*100}% approval rate"
        )

        approved_jobs_info: List[JobInfo] = []
        rejected_jobs_info: List[JobInfo] = []
        processing_failed_jobs_info: List[JobInfo] = []

        # Deterministic but fair approval using seeded shuffle
        # Use benchmark_id as seed for reproducibility
        if jobs_info and jobs_info[0].benchmark_id:
            # Convert benchmark_id to integer seed for reproducibility
            seed = hash(jobs_info[0].benchmark_id) % (2**32)
            random.seed(seed)
            logger.debug(
                f"Using seed {seed} (from benchmark_id) for job approval shuffle"
            )

        # Shuffle jobs to ensure fair distribution across DOs
        shuffled_jobs = jobs_info.copy()
        random.shuffle(shuffled_jobs)

        # Approve first N jobs from shuffled list
        num_to_approve = int(len(shuffled_jobs) * approval_rate)

        for idx, job_info in enumerate(shuffled_jobs):
            do_email = job_info.do_email

            try:
                if idx < num_to_approve:
                    # Approve first N jobs from shuffled list
                    logger.info(f"✅ Approving job for {do_email}")
                    job_info.status = JobProcessingStatus.approved
                    approved_jobs_info.append(job_info)
                else:
                    # Reject remaining jobs
                    logger.info(f"❌ Rejecting job for {do_email}")
                    job_info.status = JobProcessingStatus.rejected
                    rejected_jobs_info.append(job_info)

            except Exception as e:
                logger.error(f"Failed to process job for {do_email}: {e}")
                job_info.status = JobProcessingStatus.processing_failed
                job_info.error = str(e)
                processing_failed_jobs_info.append(job_info)

        # Log summary
        total_jobs = len(jobs_info)
        approved_count = len(approved_jobs_info)
        rejected_count = len(rejected_jobs_info)
        processing_failed_count = len(processing_failed_jobs_info)

        logger.info(
            f"📊 Job approval summary: {approved_count}/{total_jobs} approved, {rejected_count}/{total_jobs} rejected, {processing_failed_count}/{total_jobs} processing_failed"
        )

        return JobProcessingResult(
            total=total_jobs,
            num_approved=approved_count,
            num_rejected=rejected_count,
            approved_jobs=approved_jobs_info,
            rejected_jobs=rejected_jobs_info,
            processing_failed_jobs=processing_failed_jobs_info,
            approval_rate=approval_rate,
        )

    async def run_fedrag_jobs(
        self,
        jobs_processing_results: JobProcessingResult,
        fedrag_project_path: Path,
    ) -> FedRAGExecutionResult:
        """Run FedRAG jobs on all approved data owners in parallel, with DS server running concurrently.

        Args:
            jobs_processing_results: JobProcessingResult with approved jobs
            fedrag_project_path: Path to the prepared FedRAG project

        Returns:
            FedRAGExecutionResult with execution metrics and results
        """
        if not self.is_setup:
            raise RuntimeError("Federation not setup. Call setup() first.")

        logger.info(
            f"🚀 Running FedRAG jobs on {len(jobs_processing_results.approved_jobs)} approved data owners"
        )

        # Create process pool for DO jobs (one process per DO)
        num_approved_jobs = len(jobs_processing_results.approved_jobs)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_approved_jobs
        ) as process_pool:
            # Start DS server task (will run until completion)
            logger.info("Starting DS aggregator server...")
            ds_task = asyncio.create_task(
                self._run_ds_aggregator_server(fedrag_project_path)
            )

            # Start DO client tasks (will run in passive mode, listening for queries)
            logger.info("Starting DO clients...")
            do_tasks = []
            for job_info in jobs_processing_results.approved_jobs:
                do_tasks.append(
                    asyncio.create_task(self._run_single_do_job(job_info, process_pool))
                )

            # Wait for DS server to complete
            ds_result = await ds_task

            # Cancel all DO tasks when DS completes (following syft_flwr pattern)
            logger.debug("DS server completed, cancelling DO client tasks...")
            for do_task in do_tasks:
                if not do_task.done():
                    do_task.cancel()

            # Gather DO results (with exceptions handled)
            do_results = await asyncio.gather(*do_tasks, return_exceptions=True)

        # Process DO results
        updated_jobs = []
        for idx, result in enumerate(do_results):
            if isinstance(result, asyncio.CancelledError):
                # DO was cancelled after DS completed - this is expected
                job_info = jobs_processing_results.approved_jobs[idx]
                job_info.status = JobProcessingStatus.approved
                updated_jobs.append(job_info)
                logger.debug(
                    f"DO client {job_info.do_email} cancelled after DS completion"
                )
            elif isinstance(result, Exception):
                job_info = jobs_processing_results.approved_jobs[idx]
                job_info.status = JobProcessingStatus.processing_failed
                job_info.error = str(result)
                updated_jobs.append(job_info)
            else:
                updated_jobs.append(result)

        successful_count = len(
            [j for j in updated_jobs if j.status == JobProcessingStatus.approved]
        )
        failed_count = len(updated_jobs) - successful_count
        logger.success(f"✅ Completed {successful_count}/{len(updated_jobs)} DO jobs")

        # Convert ds_result dict to DSServerResult model
        if isinstance(ds_result, Exception):
            ds_server_result = DSServerResult(
                status="error",
                error=str(ds_result),
            )
        else:
            ds_server_result = DSServerResult(**ds_result)

        # Create and return FedRAGExecutionResult
        return FedRAGExecutionResult(
            total_jobs=len(updated_jobs),
            successful_jobs=successful_count,
            failed_jobs=failed_count,
            job_results=updated_jobs,
            ds_server_result=ds_server_result,
            success_rate=successful_count / len(updated_jobs) if updated_jobs else 0.0,
        )

    async def _run_single_do_job(
        self, job_info: JobInfo, process_pool: concurrent.futures.ProcessPoolExecutor
    ) -> JobInfo:
        """Run a single DO job on private data using run_private() in a separate process.

        Args:
            job_info: JobInfo for the job to run
            process_pool: ProcessPoolExecutor for running job in separate process

        Returns:
            Updated JobInfo with execution results
        """
        try:
            logger.info(f"Running DO job for {job_info.do_email}...")

            # Get DO's config path from stack
            do_stack = self.do_stacks[job_info.do_email]
            config_path = str(do_stack.client.config_path)

            # Run in process pool executor (each process gets its own environment)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                process_pool,
                _run_do_job_in_process,
                job_info.do_email,
                str(job_info.job.uid),
                config_path,
            )

            # Update job info
            job_info.status = JobProcessingStatus.approved
            logger.success(f"✅ DO job completed for {job_info.do_email}")
            return job_info

        except asyncio.CancelledError:
            logger.debug(f"DO job for {job_info.do_email} cancelled (DS completed)")
            job_info.status = JobProcessingStatus.approved
            return job_info
        except Exception as e:
            logger.error(f"❌ Failed to run DO job for {job_info.do_email}: {e}")
            job_info.status = JobProcessingStatus.processing_failed
            job_info.error = str(e)
            return job_info

    def _log_ds_server_output(
        self, stdout: str, stderr: str, success: bool = True
    ) -> None:
        """Log DS server stdout/stderr output.

        Args:
            stdout: Server stdout output
            stderr: Server stderr output
            success: Whether server execution was successful
        """
        if success:
            # Log stdout results at info level (contains FedRAG metrics)
            if stdout.strip():
                logger.info("📊 DS Server Results:")
                for line in stdout.strip().split("\n"):
                    if line.strip():
                        logger.info(f"   {line}")

            # Log stderr at debug level if present
            if stderr.strip():
                logger.debug("DS server stderr:")
                for line in stderr.strip().split("\n"):
                    if line.strip():
                        logger.debug(f"   {line}")
        else:
            # On failure, log stderr at error level for diagnostics
            if stderr.strip():
                logger.error("DS server stderr:")
                for line in stderr.strip().split("\n"):
                    if line.strip():
                        logger.error(f"   {line}")

            # Log stdout at debug level (may contain partial results)
            if stdout.strip():
                logger.debug("DS server stdout:")
                for line in stdout.strip().split("\n"):
                    if line.strip():
                        logger.debug(f"   {line}")

    async def _run_ds_aggregator_server(
        self, fedrag_project_path: Path
    ) -> Dict[str, Any]:
        """Run DS aggregator FL server.

        Args:
            fedrag_project_path: Path to the FedRAG project

        Returns:
            Dictionary with server execution results
        """
        try:
            logger.info("Starting DS aggregator FL server...")

            main_py_path = fedrag_project_path / "main.py"
            if not main_py_path.exists():
                raise FileNotFoundError(f"main.py not found at {main_py_path}")

            process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                str(main_py_path),
                "--active",  # Use python directly from the parent project's venv
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=fedrag_project_path,
            )

            stdout, stderr = await process.communicate()
            stdout_str = stdout.decode()
            stderr_str = stderr.decode()

            if process.returncode == 0:
                logger.success("✅ DS aggregator server completed successfully")
                self._log_ds_server_output(stdout_str, stderr_str, success=True)

                return {
                    "status": "success",
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }
            else:
                logger.error(
                    f"❌ DS aggregator server failed with code {process.returncode}"
                )
                self._log_ds_server_output(stdout_str, stderr_str, success=False)

                return {
                    "status": "failed",
                    "returncode": process.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                }

        except Exception as e:
            logger.error(f"❌ Failed to run DS aggregator server: {e}")
            return {"status": "error", "error": str(e)}

    async def _upload_single_dataset(self, do_info: DataOwnerInfo) -> DatasetUploadInfo:
        """Upload dataset for a single data owner.

        Args:
            do_info: DataOwnerInfo object for the data owner

        Returns:
            DatasetUploadInfo with upload result information
        """
        try:
            logger.debug(f"📤 Starting dataset upload for {do_info.email}")

            # Validate dataset exists before attempting upload
            if not validate_dataset_exists(
                do_info.dataset, self.use_subset, self.root_dir
            ):
                raise FileNotFoundError(
                    f"Dataset {do_info.dataset} not found in {'subset' if self.use_subset else 'full'} mode"
                )

            # Get dataset paths
            corpus_path: Path = get_dataset_path(
                do_info.dataset, self.use_subset, self.root_dir
            )
            dataset_name = do_info.dataset

            # Get DO client
            do_client = self.do_clients[do_info.email]

            # Create Syft dataset
            logger.debug(f"Creating Syft dataset '{dataset_name}' for {do_info.email}")
            private_path = corpus_path / "private"
            mock_path = corpus_path / "mock"
            if not private_path.exists() or not mock_path.exists():
                raise FileNotFoundError(
                    f"Dataset paths not found for {do_info.dataset}"
                )

            dataset = do_client.dataset.create(
                name=dataset_name,
                path=private_path,
                mock_path=mock_path,
                description_path=(mock_path / "README.md")
                if (mock_path / "README.md").exists()
                else None,
            )

            logger.success(
                f"✅ Dataset upload successful for {do_info.email}: {dataset_name}"
            )

            return DatasetUploadInfo(
                do_email=do_info.email,
                dataset_name=do_info.dataset,
                syft_dataset_name=dataset_name,
                dataset_object=dataset,
                corpus_path=str(corpus_path),
                status=DatasetUploadStatus.success,
                data_fraction=do_info.data_fraction,
            )

        except Exception as e:
            logger.error(f"❌ Dataset upload failed for {do_info.email}: {e}")
            return DatasetUploadInfo(
                do_email=do_info.email,
                dataset_name=do_info.dataset,
                status=DatasetUploadStatus.failed,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _process_upload_results(self, results: List[Any]) -> DatasetUploadResult:
        """Process and categorize upload results.

        Args:
            results: List of upload results or exceptions

        Returns:
            DatasetUploadResult with categorized results and summary statistics
        """
        successful = []
        failed = []

        for result in results:
            # Handle exceptions that occurred during asyncio.gather
            if isinstance(result, Exception):
                failed.append(
                    DatasetUploadInfo(
                        do_email="unknown",
                        dataset_name="unknown",
                        status=DatasetUploadStatus.failed,
                        error=str(result),
                        error_type=type(result).__name__,
                    )
                )
            elif result.status == DatasetUploadStatus.success:
                successful.append(result)
            else:
                failed.append(result)

        # Log summary
        total = len(results)
        logger.info(
            f"📊 Dataset upload summary: {len(successful)}/{total} successful, {len(failed)}/{total} failed"
        )

        if failed:
            logger.warning("Failed uploads:")
            for fail in failed:
                logger.warning(f"  ❌ {fail.do_email}: {fail.error or 'Unknown error'}")

        return DatasetUploadResult(
            total=total,
            successful=successful,
            failed=failed,
            success_count=len(successful),
            failure_count=len(failed),
            success_rate=len(successful) / total if total > 0 else 0.0,
        )

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


def _run_do_job_in_process(do_email: str, job_uid: str, config_path: str) -> None:
    """Run DO job in a separate process.

    This function must be at module level to be picklable by ProcessPoolExecutor.
    Each process gets its own environment and reconstructs the DO client.

    Args:
        do_email: Data owner email
        job_uid: Job UID to execute
        config_path: Path to SyftBox client config file
    """
    # Disable verbose syft library logs in this process
    logger.disable("syft_event")
    logger.disable("syft_rds")
    logger.disable("syft_crypto")
    logger.disable("syft_flwr")

    # Set environment for this process
    os.environ["SYFTBOX_CLIENT_CONFIG_PATH"] = config_path

    # Load SyftBox client from config
    syftbox_client = SyftBoxClient.load(Path(config_path))

    # Create RDS stack with the loaded client
    do_stack = SingleRDSStack(client=syftbox_client)

    # Initialize session as admin on own datasite
    do_client = do_stack.init_session(host=do_email)

    # Get and run the job
    job = do_client.job.get(job_uid)
    do_client.run_private(job)
