"""
FederationManager: Manages the setup and lifecycle of the federated environment.
Handles SyftBox network initialization and data owner registration.
"""

import os
import asyncio
import random
import hashlib
import concurrent.futures
from pathlib import Path
from typing_extensions import Dict, List, Any, Optional
import shutil
import time

from omegaconf import DictConfig
from loguru import logger

from syft_core import Client as SyftBoxClient
from syft_rds.client.rds_client import RDSClient, init_session
from syft_rds.models import Job

from safer_bench.dataset_utils import get_dataset_path, validate_dataset_exists
from safer_bench.data_partitioner import DataPartitioner
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


LOCAL_SYFTBOX_NETWORK = "local_syftbox_network"
SAFER_BENCH_SYFTBOX_NETWORK = "safer_bench_network"


class FederationConfig:
    """Configuration constants for federation management."""

    # Timeouts
    DEFAULT_JOB_READY_TIMEOUT = 60  # seconds
    DATASET_UPLOAD_TIMEOUT = 60  # seconds
    ASYNC_GATHER_TIMEOUT = 600  # seconds for general async.gather operations
    PROCESS_CANCEL_TIMEOUT = 5.0  # seconds

    # Intervals
    JOB_READY_CHECK_INTERVAL = 2  # seconds

    # Retries
    DIRECTORY_CLEANUP_MAX_RETRIES = 3
    DIRECTORY_CLEANUP_RETRY_DELAY = 0.5  # seconds


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
        self.syftbox_network_dir = (
            self.root_dir / LOCAL_SYFTBOX_NETWORK / self.network_key
        )

        # Parse federation configuration
        self.federation_name: str = (
            cfg.federation.name
        )  # e.g., "hybrid_2do", "centralized_1do"
        self.aggregator_email: str = cfg.federation.aggregator
        self.data_owners: List[DataOwnerInfo] = self._parse_data_owners()
        self.use_subset: bool = cfg.dataset.use_subset
        self.benchmark_id: Optional[str] = None  # Set by BenchmarkRunner

        # Runtime state tracked separately
        self.ds_client: Optional[RDSClient] = None
        self.do_clients: Dict[str, RDSClient] = {}  # { client_email: RDSClient }
        self._guest_clients: List[RDSClient] = []  # Track guest clients for cleanup
        self.is_setup = False

        # Lock for protecting shared state access
        self._state_lock = asyncio.Lock()

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

            # Prepare data partitions (if needed)
            logger.info("🔧 Preparing data partitions")
            self._prepare_data_partitions()

            # Setup Data Scientist (aggregator)
            logger.info(f"Setting up aggregator: {self.aggregator_email}")
            self.ds_client = await self._setup_aggregator()

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
            logger.exception(f"Failed to setup federation: {e}")
            await self.cleanup()
            raise RuntimeError(f"Federation setup failed: {e}") from e

    async def cleanup(self) -> None:
        """Clean up federation resources and reset state.

        Safe to call multiple times - will clean up any existing resources
        and reset the federation state.
        """
        logger.debug("🧹 Cleaning up federation resources")

        try:
            # Stop all RDS servers before cleanup to release file handles
            if self.ds_client:
                try:
                    self.ds_client.stop_server()
                    logger.debug("Stopped DS RDS server")
                except Exception as e:
                    logger.debug(f"Failed to stop DS server: {e}")

            # Protect access to shared state with lock
            async with self._state_lock:
                for email, client in self.do_clients.items():
                    try:
                        client.stop_server()
                        logger.debug(f"Stopped DO's SyftEvent server for {email}")
                    except Exception as e:
                        logger.debug(
                            f"Failed to stop DO's SyftEvent server {email}: {e}"
                        )

                # Clean up guest clients
                for guest_client in self._guest_clients:
                    try:
                        guest_client.stop_server()
                        logger.debug("Stopped guest client server")
                    except Exception as e:
                        logger.debug(f"Failed to stop guest client server: {e}")

            # Clean up federation network directory
            self._cleanup_directory(
                directory=self.syftbox_network_dir, description=self.network_key
            )

            # Clean up .syftbox directory
            self._cleanup_directory(
                directory=self.root_dir / LOCAL_SYFTBOX_NETWORK / ".syftbox",
                description=".syftbox",
            )

            # Reset runtime state - protect with lock
            async with self._state_lock:
                self.ds_client = None
                self.do_clients.clear()
                self._guest_clients.clear()
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
            # Create guest connection from DS to DO's datasite
            guest_client = init_session(
                host=do_info.email,
                email=self.aggregator_email,
                syftbox_dir=self.syftbox_network_dir,
                reset=False,
                start_syft_event_server=False,  # Don't start server for guest connections
            )
            guest_clients.append(guest_client)

            # Track for cleanup
            async with self._state_lock:
                self._guest_clients.append(guest_client)

        return guest_clients

    async def wait_for_jobs(self, timeout: int = 60) -> None:
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
                    # Check if DO is responsive - protect with lock
                    async with self._state_lock:
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

        # Execute all uploads concurrently with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=FederationConfig.DATASET_UPLOAD_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Dataset upload timed out after {FederationConfig.DATASET_UPLOAD_TIMEOUT}s"
            )
            raise TimeoutError(
                f"Dataset uploads failed to complete within {FederationConfig.DATASET_UPLOAD_TIMEOUT}s"
            )

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
                guest_client = init_session(
                    host=do_info.email,
                    email=self.aggregator_email,
                    syftbox_dir=self.syftbox_network_dir,
                    reset=False,
                    start_syft_event_server=False,
                )

                # Track guest client for cleanup
                async with self._state_lock:
                    self._guest_clients.append(guest_client)

                dataset_name = do_info.get_dataset_name()
                logger.info(
                    f"Submitting job to {do_info.email} for dataset {dataset_name}"
                )

                # Submit job using syft_rds client
                job = guest_client.job.submit(
                    name=f"fedrag_benchmark_{benchmark_id}",
                    user_code_path=fedrag_project_path,
                    dataset_name=dataset_name,
                    entrypoint="main.py",
                )

                job_info = JobInfo(
                    job=job,
                    do_email=do_info.email,
                    dataset=dataset_name,
                    client=guest_client,
                    status=JobProcessingStatus.submission_succeeded,  # Jobs start pending review
                    data_fraction=1.0,
                    benchmark_id=benchmark_id,
                )
                jobs_info.append(job_info)

                logger.success(f"✅ Job submitted to {do_info.email}")

            except Exception as e:
                logger.error(f"❌ Failed to submit job to {do_info.email}: {e}")
                job_info = JobInfo(
                    job=None,
                    do_email=do_info.email,
                    dataset=do_info.get_dataset_name(),
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
            # Convert benchmark_id to integer seed using deterministic hashlib
            # (hash() is non-deterministic across Python sessions)
            benchmark_id_bytes = jobs_info[0].benchmark_id.encode("utf-8")
            hash_digest = hashlib.sha256(benchmark_id_bytes).hexdigest()
            seed = int(hash_digest[:8], 16)  # Use first 8 hex chars as seed
            random.seed(seed)
            logger.debug(
                f"Using deterministic seed {seed} (from benchmark_id) for job approval shuffle"
            )

        # Shuffle jobs to ensure fair distribution across DOs
        shuffled_jobs = jobs_info.copy()
        random.shuffle(shuffled_jobs)

        # Approve first N jobs from shuffled list
        # Use round() to properly round fractional approvals (e.g., 0.75 -> 1)
        num_to_approve = round(len(shuffled_jobs) * approval_rate)

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

        # Validate approved jobs list is non-empty
        if not jobs_processing_results.approved_jobs:
            raise ValueError(
                "No approved jobs to run. Cannot execute FedRAG with empty job list."
            )

        logger.info(
            f"🚀 Running FedRAG: DS aggregator server + {len(jobs_processing_results.approved_jobs)} DO clients"
        )

        # Create process pool for both DS and DO jobs (DS + num_DOs)
        num_approved_jobs = len(jobs_processing_results.approved_jobs)
        max_workers = num_approved_jobs + 1  # +1 for DS

        # Create process pool with explicit lifecycle management
        process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

        try:
            # Start DO clients and DS server concurrently
            do_task = asyncio.create_task(
                self._run_dos(jobs_processing_results.approved_jobs, process_pool)
            )
            ds_task = asyncio.create_task(
                self._run_ds(fedrag_project_path, process_pool)
            )

            # Wait for DS server to complete
            job = await ds_task

            # Get DS job logs
            ds_logs: Dict = self.ds_client.job.get_logs(job)

            ds_result = {
                "status": "success",
                "stdout": ds_logs.get("stdout", ""),
                "stderr": ds_logs.get("stderr", ""),
                "logs_dir": ds_logs.get("logs_dir", ""),
                "job": job,
            }

            # Cancel DO tasks when DS completes (following syft_flwr pattern)
            logger.debug("DS server completed, cancelling DO client tasks...")
            do_task.cancel()

            # Gather DO results (with exceptions handled)
            try:
                do_results = await do_task
            except asyncio.CancelledError:
                # DO tasks were cancelled - mark all as successful
                do_results = jobs_processing_results.approved_jobs
                logger.debug("DO tasks cancelled after DS completion")

            # Process results and create return value
            # Copy FAISS indexes back to partition directories
            self._copy_indexes_to_partitions(do_results)

            return self._build_execution_result(do_results, ds_result)

        finally:
            # Explicitly shutdown process pool to prevent zombie processes
            logger.debug("Shutting down process pool...")
            process_pool.shutdown(wait=True, cancel_futures=False)
            logger.debug("Process pool shutdown complete")

    def _copy_indexes_to_partitions(self, do_results: List) -> None:
        """Copy built FAISS indexes from SyftBox runtime back to partition directories.

        Args:
            do_results: List of JobInfo objects with dataset information
        """
        logger.info("📦 Copying FAISS indexes back to partition directories...")

        for job_info in do_results:
            try:
                do_email = job_info.do_email
                dataset_name = job_info.dataset

                if not dataset_name:
                    logger.warning(
                        f"No dataset_name for {do_email}, skipping index copy"
                    )
                    continue

                # Get the SyftBox runtime dataset path (where indexes were built)
                syftbox_dataset_path = (
                    self.root_dir
                    / LOCAL_SYFTBOX_NETWORK
                    / ".syftbox"
                    / "private_datasets"
                    / do_email
                    / dataset_name
                )

                # Get the partition base path: partitions/federation_name/do_email/
                do_email_prefix = do_email.split("@")[0]
                partition_base_path = (
                    self.root_dir
                    / "datasets"
                    / ("subsets" if self.use_subset else "full")
                    / "partitions"
                    / self.federation_name
                    / do_email_prefix
                )
                partition_private_path = partition_base_path / "private"
                partition_mock_path = partition_base_path / "mock"

                # Check if indexes exist in runtime location
                index_file = syftbox_dataset_path / "faiss.index"
                doc_ids_file = syftbox_dataset_path / "all_doc_ids.npy"

                logger.debug(f"Looking for indexes at: {syftbox_dataset_path}")
                logger.debug(
                    f"Index file exists: {index_file.exists()}, path: {index_file}"
                )
                logger.debug(
                    f"Doc IDs file exists: {doc_ids_file.exists()}, path: {doc_ids_file}"
                )

                if not index_file.exists() or not doc_ids_file.exists():
                    logger.warning(
                        f"Indexes not found for {do_email}/{dataset_name} at {syftbox_dataset_path}"
                    )
                    continue

                # Ensure partition directories exist
                if not partition_private_path.exists():
                    logger.warning(
                        f"Partition directory not found: {partition_private_path}, skipping index copy"
                    )
                    continue

                # Copy index files to private directory
                shutil.copy2(index_file, partition_private_path / "faiss.index")
                shutil.copy2(doc_ids_file, partition_private_path / "all_doc_ids.npy")

                # QUICK FIX: Also copy to mock directory if it exists (to keep extensions in sync)
                # Syft RDS requires private and mock directories to have matching file extensions.
                # Without this, dataset uploads fail with: "Directories contain different file extensions"
                # The mock directory is used by Syft RDS for validation/testing purposes.
                if partition_mock_path.exists():
                    shutil.copy2(index_file, partition_mock_path / "faiss.index")
                    shutil.copy2(doc_ids_file, partition_mock_path / "all_doc_ids.npy")
                    logger.debug(
                        f"Copied indexes to mock directory: {partition_mock_path}"
                    )

                logger.success(
                    f"✅ Copied FAISS indexes for {dataset_name} to {partition_private_path}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to copy indexes for {getattr(job_info, 'do_email', 'unknown')}: {e}"
                )

        logger.info("📦 Index copy complete")

    def _cleanup_directory(self, directory: Path, description: str) -> None:
        """Helper method to clean up a directory with retries.

        Args:
            directory: Path to directory to remove
            description: Human-readable description for logging
        """
        if not directory.exists():
            logger.debug(f"Directory {description} doesn't exist, skipping")
            return

        max_retries = FederationConfig.DIRECTORY_CLEANUP_MAX_RETRIES
        for attempt in range(max_retries):
            try:
                shutil.rmtree(directory, ignore_errors=False)
                logger.debug(f"✅ Cleaned up {description} directory: {directory}")
                return
            except OSError as e:
                if attempt < max_retries - 1:
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries} for {description}: {e}"
                    )
                    time.sleep(FederationConfig.DIRECTORY_CLEANUP_RETRY_DELAY)
                else:
                    # Last attempt: try with ignore_errors=True
                    logger.warning(
                        f"Failed to cleanup {description} after {max_retries} attempts: {e}"
                    )
                    try:
                        shutil.rmtree(directory, ignore_errors=True)
                    except Exception as cleanup_error:
                        logger.debug(
                            f"Final cleanup attempt failed for {description}: {cleanup_error}"
                        )

    async def _run_ds(
        self,
        fedrag_project_path: Path,
        process_pool: concurrent.futures.ProcessPoolExecutor,
    ) -> Job:
        """Run DS aggregator server using syft-rds job submission workflow in a separate process.

        Args:
            fedrag_project_path: Path to the FedRAG project
            process_pool: ProcessPoolExecutor for running DS job in separate process

        Returns:
            Job object representing the DS job
        """
        try:
            logger.debug("Starting DS aggregator server via syft-rds job workflow...")

            # DS submits job to themselves (no dataset_name needed for aggregator)
            job = self.ds_client.job.submit(
                name=f"fedrag_aggregator_{self.benchmark_id}",
                user_code_path=fedrag_project_path,
                entrypoint="main.py",
            )
            logger.debug(f"DS job submitted: {job.uid}")

            # DS approves their own job
            self.ds_client.job.approve(job)
            logger.debug(f"DS job approved: {job.uid}")

            # Run the job in blocking mode in a separate process
            logger.debug("Running DS aggregator job in separate process...")

            # Get DS config path
            ds_config_path = str(self.ds_client.local_store.syftbox_client.config_path)

            # Run DS job in process pool to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                process_pool,
                _run_ds_job_in_process,
                self.aggregator_email,
                str(job.uid),
                ds_config_path,
            )

            return job

        except Exception as e:
            logger.exception(f"Failed to run DS aggregator server: {e}")
            raise RuntimeError(f"DS aggregator server execution failed: {e}") from e

    async def _run_dos(
        self,
        approved_jobs: List[JobInfo],
        process_pool: concurrent.futures.ProcessPoolExecutor,
    ) -> List[JobInfo]:
        """Run all DO client jobs in parallel.

        Args:
            approved_jobs: List of approved JobInfo objects
            process_pool: ProcessPoolExecutor for running jobs in separate processes

        Returns:
            List of updated JobInfo objects with execution results
        """
        logger.info(f"🚀 Launching {len(approved_jobs)} DO clients...")

        # Start DO client tasks (will run in passive mode, listening for queries)
        do_tasks = []
        for job_info in approved_jobs:
            do_tasks.append(
                asyncio.create_task(self._run_single_do_job(job_info, process_pool))
            )

        # Gather DO results (with exceptions handled) with timeout
        try:
            do_results = await asyncio.wait_for(
                asyncio.gather(*do_tasks, return_exceptions=True),
                timeout=FederationConfig.ASYNC_GATHER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"DO client tasks timed out after {FederationConfig.ASYNC_GATHER_TIMEOUT}s"
            )
            # Cancel all tasks on timeout
            for task in do_tasks:
                task.cancel()
            # Return approved jobs with timeout status
            do_results = [
                asyncio.TimeoutError(
                    f"DO job timed out after {FederationConfig.ASYNC_GATHER_TIMEOUT}s"
                )
                for _ in approved_jobs
            ]

        # Process DO results
        updated_jobs = []
        for idx, result in enumerate(do_results):
            if isinstance(result, asyncio.CancelledError):
                # DO was cancelled after DS completed - this is expected
                job_info = approved_jobs[idx]
                job_info.status = JobProcessingStatus.approved
                updated_jobs.append(job_info)
                logger.debug(
                    f"DO client {job_info.do_email} cancelled after DS completion"
                )
            elif isinstance(result, Exception):
                job_info = approved_jobs[idx]
                job_info.status = JobProcessingStatus.processing_failed
                job_info.error = str(result)
                updated_jobs.append(job_info)
            else:
                updated_jobs.append(result)

        successful_count = len(
            [j for j in updated_jobs if j.status == JobProcessingStatus.approved]
        )
        logger.success(
            f"✅ {successful_count}/{len(updated_jobs)} DO clients listening for queries"
        )

        return updated_jobs

    def _build_execution_result(
        self,
        do_results: List[JobInfo],
        ds_result: Dict[str, Any],
    ) -> FedRAGExecutionResult:
        """Build FedRAGExecutionResult from execution results.

        Args:
            do_results: List of JobInfo with execution results from DOs
            ds_result: Dictionary with DS server execution results

        Returns:
            FedRAGExecutionResult with complete execution metrics
        """
        # Convert ds_result dict to DSServerResult model
        if isinstance(ds_result, Exception):
            ds_server_result = DSServerResult(
                status="error",
                error=str(ds_result),
            )
        else:
            ds_server_result = DSServerResult(**ds_result)

        # Calculate success metrics
        successful_count = len(
            [j for j in do_results if j.status == JobProcessingStatus.approved]
        )
        failed_count = len(do_results) - successful_count

        # Create and return FedRAGExecutionResult
        return FedRAGExecutionResult(
            total_jobs=len(do_results),
            successful_jobs=successful_count,
            failed_jobs=failed_count,
            job_results=do_results,
            ds_server_result=ds_server_result,
            success_rate=successful_count / len(do_results) if do_results else 0.0,
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
            logger.debug(
                f"Starting DO client: {job_info.do_email} in a separate process..."
            )

            # Get DO's config path from client - protect with lock
            async with self._state_lock:
                do_client = self.do_clients[job_info.do_email]
            config_path = str(do_client.local_store.syftbox_client.config_path)

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
            logger.debug(f"DO client listening: {job_info.do_email}")
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

    async def _upload_single_dataset(self, do_info: DataOwnerInfo) -> DatasetUploadInfo:
        """Upload dataset for a single data owner.

        Args:
            do_info: DataOwnerInfo object for the data owner

        Returns:
            DatasetUploadInfo with upload result information
        """
        try:
            logger.debug(f"📤 Starting dataset upload for {do_info.email}")

            # Determine dataset path based on distribution strategy
            if do_info.partition_name:
                # Use partitioned dataset
                mode_dir = "subsets" if self.use_subset else "full"
                corpus_path = (
                    self.root_dir
                    / "datasets"
                    / mode_dir
                    / "partitions"
                    / do_info.partition_name
                )
                dataset_name = do_info.partition_name
                logger.debug(f"Using partitioned dataset: {dataset_name}")
            else:
                # Use single original dataset
                dataset_name = list(do_info.datasets.keys())[
                    0
                ]  # Get first (and only) dataset

                # Validate dataset exists
                if not validate_dataset_exists(
                    dataset_name, self.use_subset, self.root_dir
                ):
                    raise FileNotFoundError(
                        f"Dataset {dataset_name} not found in {'subset' if self.use_subset else 'full'} mode"
                    )

                corpus_path = get_dataset_path(
                    dataset_name, self.use_subset, self.root_dir
                )
                logger.debug(f"Using original dataset: {dataset_name}")

            # Get DO client - protect with lock
            async with self._state_lock:
                do_client = self.do_clients[do_info.email]

            # Create Syft dataset
            logger.debug(f"Creating Syft dataset '{dataset_name}' for {do_info.email}")
            private_path = corpus_path / "private"
            mock_path = corpus_path / "mock"
            if not private_path.exists() or not mock_path.exists():
                raise FileNotFoundError(
                    f"Dataset paths not found for {dataset_name}: private={private_path}, mock={mock_path}"
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
                dataset_name=dataset_name,
                syft_dataset_name=dataset_name,
                dataset_object=dataset,
                corpus_path=str(corpus_path),
                status=DatasetUploadStatus.success,
                data_fraction=1.0,  # Always 1.0 since we upload the entire partition/dataset
            )

        except Exception as e:
            logger.error(f"❌ Dataset upload failed for {do_info.email}: {e}")
            return DatasetUploadInfo(
                do_email=do_info.email,
                dataset_name=dataset_name if "dataset_name" in locals() else "unknown",
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
                    datasets=dict(do_config.datasets),  # Convert OmegaConf to dict
                    distribution_strategy=do_config.get(
                        "distribution_strategy", "single"
                    ),
                    topics=list(do_config.topics)
                    if hasattr(do_config, "topics")
                    else None,
                )
            )

        return data_owners

    def _prepare_data_partitions(self) -> None:
        """Prepare custom data partitions based on DO distribution strategies.

        For 'hybrid' and 'centralized' strategies, creates partitioned datasets
        by sampling/merging chunks. Partitions are created in datasets/{mode}/partitions/.
        """
        # Check if any DO needs partitioning
        needs_partitioning = any(
            do.distribution_strategy in ["hybrid", "centralized"]
            for do in self.data_owners
        )

        if not needs_partitioning:
            logger.debug("No partitioning needed (all DOs using 'single' strategy)")
            return

        # Initialize partitioner
        partitioner = DataPartitioner(seed=42)

        # Create partitions for each DO that needs one
        for idx, do_info in enumerate(self.data_owners):
            if do_info.distribution_strategy == "single":
                logger.debug(
                    f"No partitioning needed for {do_info.email}, will use original dataset"
                )
                continue

            # Create partition directory
            mode_dir = "subsets" if self.use_subset else "full"
            partitions_base = self.root_dir / "datasets" / mode_dir / "partitions"
            partitions_base.mkdir(parents=True, exist_ok=True)

            # Create unique partition name: federation_name/do_email
            # This ensures each federation config has separate partitions
            do_email_prefix = do_info.email.split("@")[0]
            partition_path = (
                partitions_base / self.federation_name / do_email_prefix / "private"
            )

            logger.info(
                f"Creating {do_info.distribution_strategy} partition for {do_info.email}"
            )

            if do_info.distribution_strategy == "hybrid":
                partitioner.create_hybrid_partition(
                    datasets=do_info.datasets,
                    output_path=partition_path,
                    use_subset=self.use_subset,
                    project_root_dir=self.root_dir,
                    partition_id=idx,  # Pass partition index for deterministic slicing
                    total_partitions=len(
                        self.data_owners
                    ),  # Total number of DOs for slicing
                )
            elif do_info.distribution_strategy == "centralized":
                partitioner.create_centralized_partition(
                    datasets=do_info.datasets,
                    output_path=partition_path,
                    use_subset=self.use_subset,
                    project_root_dir=self.root_dir,
                )

            # Store partition name in DO info for later use
            # Format: federation_name/do_email (e.g., "hybrid_2do/do1")
            partition_name = f"{self.federation_name}/{do_email_prefix}"
            do_info.partition_name = partition_name
            logger.success(f"✅ Partition created: {partition_name}")

    async def _setup_aggregator(self) -> RDSClient:
        """Setup the data scientist/aggregator node.

        Returns:
            RDSClient for the aggregator
        """
        # Setup RDS client for aggregator using init_session
        # This will automatically create the RDS server and initialize the session
        ds_client = init_session(
            host=self.aggregator_email,
            email=self.aggregator_email,
            syftbox_dir=self.syftbox_network_dir,
            reset=False,
            start_syft_event_server=True,
        )

        # Set environment variable for SyftBox client config
        os.environ["SYFTBOX_CLIENT_CONFIG_PATH"] = str(
            ds_client.local_store.syftbox_client.config_path
        )

        logger.debug(f"Aggregator setup complete: {self.aggregator_email}")
        return ds_client

    async def _setup_data_owners(self) -> None:
        """Setup all data owner nodes."""
        tasks = []

        for do_info in self.data_owners:
            task = self._setup_single_do(do_info)
            tasks.append(task)

        # Setup all DOs in parallel with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=FederationConfig.ASYNC_GATHER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Data owner setup timed out after {FederationConfig.ASYNC_GATHER_TIMEOUT}s"
            )
            raise TimeoutError(
                f"Data owner setup failed to complete within {FederationConfig.ASYNC_GATHER_TIMEOUT}s"
            )

        logger.debug(f"All {len(self.data_owners)} data owners setup complete")

    async def _setup_single_do(self, do_info: DataOwnerInfo) -> None:
        """Setup a single data owner node.

        Args:
            do_info: DataOwnerInfo object for the data owner
        """
        # Setup RDS client for DO using init_session
        # This will automatically create the RDS server and initialize the session
        do_client = init_session(
            host=do_info.email,
            email=do_info.email,
            syftbox_dir=self.syftbox_network_dir,
            reset=False,
            start_syft_event_server=True,
        )

        # Store client in dictionary - protect with lock
        async with self._state_lock:
            self.do_clients[do_info.email] = do_client

        logger.debug(
            f"Data owner setup complete: {do_info.email} with dataset {do_info.get_dataset_name()}"
        )

    async def _verify_federation(self) -> None:
        """Verify that all nodes can communicate."""
        logger.debug("Verifying federation connectivity")

        # DS should be able to connect to all DOs as guest
        for do_info in self.data_owners:
            try:
                # Try to connect from DS to DO as guest
                guest_client = init_session(
                    host=do_info.email,
                    email=self.aggregator_email,
                    syftbox_dir=self.syftbox_network_dir,
                    reset=False,
                    start_syft_event_server=False,
                )

                # Track guest client for cleanup
                async with self._state_lock:
                    self._guest_clients.append(guest_client)

                assert not guest_client.is_admin, f"Should be guest on {do_info.email}"
                logger.debug(f"✅ DS can connect to {do_info.email} as guest")
            except Exception as e:
                logger.warning(f"⚠️ DS cannot connect to {do_info.email}: {e}")

        # Each DO should be admin on their own datasite
        for do_info in self.data_owners:
            # Protect access with lock
            async with self._state_lock:
                do_client = self.do_clients[do_info.email]
            assert (
                do_client.is_admin
            ), f"{do_info.email} should be admin on their datasite"
            logger.debug(f"✅ {do_info.email} is admin on their datasite")


def _setup_job_process_environment(config_path: str) -> None:
    """Configure logging and environment variables for job process.

    Args:
        config_path: Path to SyftBox client config file
    """
    # Disable verbose syft library logs in this process
    logger.disable("syft_event")
    logger.disable("syft_rds")
    logger.disable("syft_crypto")
    logger.disable("syft_flwr")

    # Set environment for this process
    os.environ["SYFTBOX_CLIENT_CONFIG_PATH"] = config_path

    # Set HuggingFace cache to use shared location (avoid re-downloading models)
    os.environ["HF_HOME"] = str(Path.home() / ".cache" / "huggingface")

    # Set message timeout to 30 minutes (1800 seconds) to allow for FAISS index building
    # DOs may need time to build indexes on first query
    os.environ["SYFT_FLWR_MSG_TIMEOUT"] = "1800"


def _run_ds_job_in_process(ds_email: str, job_uid: str, config_path: str) -> None:
    """Run DS job in a separate process.

    This function must be at module level to be picklable by ProcessPoolExecutor.
    Each process gets its own environment and reconstructs the DS client.

    Args:
        ds_email: Data scientist email
        job_uid: Job UID to execute
        config_path: Path to SyftBox client config file
    """
    # Configure logging and environment for this process
    _setup_job_process_environment(config_path)

    # Initialize session as admin on own datasite using the config
    ds_client = init_session(
        host=ds_email,
        email=ds_email,
        syftbox_client_config_path=config_path,
        start_syft_event_server=True,
    )

    # Get the job
    job = ds_client.job.get(job_uid)
    if job is None:
        raise ValueError(f"Job with UID {job_uid} not found for {ds_email}")

    # Run the job in blocking=True mode with --active and --no-sync flags
    # --no-sync skips dependency synchronization to avoid reinstalling packages
    ds_client.run_private(
        job, blocking=True, uv_args=["--active", "--no-sync", "--quiet"]
    )


def _run_do_job_in_process(do_email: str, job_uid: str, config_path: str) -> None:
    """Run DO job in a separate process.

    This function must be at module level to be picklable by ProcessPoolExecutor.
    Each process gets its own environment and reconstructs the DO client.

    Args:
        do_email: Data owner email
        job_uid: Job UID to execute
        config_path: Path to SyftBox client config file
    """
    # Configure logging and environment for this process
    _setup_job_process_environment(config_path)

    # Initialize session as admin on own datasite using the config
    do_client = init_session(
        host=do_email,
        email=do_email,
        syftbox_client_config_path=config_path,
        start_syft_event_server=True,
    )

    # Get the job
    job = do_client.job.get(job_uid)
    if job is None:
        raise ValueError(f"Job with uid {job_uid} not found for {do_email}")

    # Approve the job (DO approves their own job)
    do_client.job.approve(job)

    # Run the job in blocking=False mode (passive - listens for DS queries)
    # --no-sync skips dependency synchronization to avoid reinstalling packages
    do_client.run_private(
        job, blocking=False, uv_args=["--active", "--no-sync", "--quiet"]
    )
