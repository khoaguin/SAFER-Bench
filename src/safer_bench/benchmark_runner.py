"""
BenchmarkRunner: Main orchestrator for SaferBench framework.
Coordinates the entire federated RAG benchmarking workflow.
"""

from datetime import datetime
from typing_extensions import List

from omegaconf import DictConfig
from loguru import logger

from safer_bench.federation_manager import FederationManager, JobInfo
from safer_bench.fedrag_adapter import FedRAGProjectAdapter
from safer_bench.models import (
    FedRAGExecutionResult,
    BenchmarkMetrics,
    DatasetUploadResult,
)
from safer_bench.metrics_collector import MetricsCollector
from safer_bench.results_reporter import ResultsReporter


class BenchmarkRunner:
    """Main orchestrator for running federated RAG benchmarks."""

    def __init__(self, cfg: DictConfig):
        """Initialize the benchmark runner with configuration.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg

        self.federation_manager = FederationManager(cfg)
        self.fedrag_adapter = FedRAGProjectAdapter(cfg)
        self.metrics_collector = MetricsCollector(cfg)
        self.results_reporter = ResultsReporter(cfg)

        # Track benchmark metadata
        self.start_time = None
        self.end_time = None
        self.benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.federation_manager.benchmark_id = self.benchmark_id
        self.fedrag_adapter.benchmark_id = self.benchmark_id

    async def run(self) -> BenchmarkMetrics:
        """Execute the complete benchmark workflow.

        Returns:
            BenchmarkMetrics containing benchmark metrics and results
        """
        self.start_time = datetime.now()
        logger.info(f"🚀 Starting SaferBench run: {self.benchmark_id}")

        try:
            # Stage 1: Setup federation (DOs and DS datasites)
            logger.info("=" * 60)
            logger.info("\033[1;35mStage 1/8: Setting up federation\033[0m")
            logger.info("=" * 60)
            federation_info = await self.federation_manager.setup_federation()
            logger.success(
                f"✅ Federation ready with {federation_info.num_data_owners} data owners"
            )

            # Stage 2: Data owners create datasets
            logger.info("=" * 60)
            logger.info("\033[1;35mStage 2/8: Data owners create datasets\033[0m")
            logger.info("=" * 60)
            datasets_upload_info = await self.federation_manager.dos_upload_datasets()
            self._log_dataset_upload_results(datasets_upload_info)

            # Stage 3: Data Scientist prepares FedRAG project with injected parameters
            logger.info("=" * 60)
            logger.info(
                "\033[1;35mStage 3/8: Data Scientist prepares FedRAG project with benchmark parameters\033[0m"
            )
            logger.info("=" * 60)
            fedrag_project = await self.fedrag_adapter.prepare_project(federation_info)
            logger.success(f"✅ FedRAG project prepared at: {fedrag_project}")

            # Stage 4: Data Scientist submits jobs to all data owners
            logger.info("=" * 60)
            logger.info(
                "\033[1;35mStage 4/8: Data Scientist submits FedRAG jobs to data owners\033[0m"
            )
            logger.info("=" * 60)
            jobs_info = await self.federation_manager.ds_submits_jobs(
                fedrag_project, self.benchmark_id
            )
            submitted_jobs_info: List[JobInfo] = [
                job_info
                for job_info in jobs_info
                if job_info.status == "submission_succeeded"
            ]
            logger.success(f"✅ Submitted {len(submitted_jobs_info)} jobs successfully")

            # Stage 5: Data Owners review and approve jobs based on approval rate
            logger.info("=" * 60)
            logger.info("\033[1;35mStage 5/8: Data Owners process jobs\033[0m")
            logger.info("=" * 60)
            jobs_processing_results = await self.federation_manager.dos_process_jobs(
                submitted_jobs_info, self.cfg.federation.approval.percentage
            )
            logger.info(
                f"Job processing complete: {jobs_processing_results.num_approved}/{jobs_processing_results.total} approved, "
                f"{jobs_processing_results.num_rejected}/{jobs_processing_results.total} rejected"
            )

            # Stage 6: DS updates FedRAG project's pyproject.toml with only approved DOs
            logger.info("=" * 60)
            logger.info(
                "\033[1;35mStage 6/8: DS updates the FedRAG project pyproject.toml based on approved DOs\033[0m"
            )
            logger.info("=" * 60)
            if jobs_processing_results.num_approved < jobs_processing_results.total:
                logger.info(
                    "Updating FedRAG project datasites with approved DOs only..."
                )
                approved_emails = [
                    job.do_email for job in jobs_processing_results.approved_jobs
                ]
                self.fedrag_adapter.update_datasites_for_approved_jobs(
                    fedrag_project, approved_emails
                )

            # Stage 7: Run federated RAG (DOs and DS in parallel)
            logger.info("=" * 60)
            logger.info(
                "\033[1;35mStage 7/8: Data Owners run FedRAG jobs + DS runs aggregator server\033[0m"
            )
            logger.info("=" * 60)
            fedrag_results: FedRAGExecutionResult = (
                await self.federation_manager.run_fedrag_jobs(
                    jobs_processing_results, fedrag_project
                )
            )
            logger.success(
                f"✅ FedRAG execution complete: {fedrag_results.successful_jobs}/{fedrag_results.total_jobs} jobs succeeded"
            )
            logger.info(
                f"   DS server status: {fedrag_results.ds_server_result.status}"
            )

            # Stage 8: Collect and save metrics
            logger.info("=" * 60)
            logger.info(
                "\033[1;35mStage 8/8: Data Scientist collects metrics and generating report\033[0m"
            )
            logger.info("=" * 60)
            self.end_time = datetime.now()
            metrics = await self.metrics_collector.collect(
                federation_info,
                fedrag_results,
                self.start_time,
                self.end_time,
            )

            # Set data owners for distribution formatting
            self.results_reporter.set_data_owners(self.federation_manager.data_owners)

            # Save results
            metrics_file, summary_file = self.results_reporter.save_results(
                metrics, self.benchmark_id
            )

            # Log summary to console
            logger.info("\n" + "=" * 60)
            logger.info("📊 BENCHMARK RESULTS SUMMARY")
            logger.info("=" * 60)
            logger.info(
                f"Results saved to: {metrics_file}. Summary file saved to: {summary_file}"
            )
            logger.info(
                "\n"
                + self.results_reporter.generate_summary(metrics, self.benchmark_id)
            )

            logger.success("=" * 60)
            logger.success(f"🎉 Benchmark complete! ID: {self.benchmark_id}")
            logger.success(f"📊 Results saved to: {self.cfg.evaluation.output_dir}")
            logger.success("=" * 60)

            return metrics

        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            raise
        finally:
            # Conditional cleanup based on configuration
            should_clean = self.cfg.get("runtime", {}).get("clean", True)
            if should_clean:
                logger.info("🧹 Cleaning up federation directories...")
                await self.federation_manager.cleanup()
            else:
                logger.info("🔍 Keeping federation directories for inspection")
                logger.info(
                    f"   Network directory: {self.federation_manager.root_dir / self.federation_manager.network_key}"
                )

    def _log_dataset_upload_results(self, upload_result: DatasetUploadResult) -> None:
        """Log the results of dataset upload operations.

        Args:
            upload_result: DatasetUploadResult containing upload results
        """
        if upload_result.success_count > 0:
            logger.success(
                f"✅ Datasets uploaded: {upload_result.success_count}/{upload_result.total} successful"
            )
            for result in upload_result.successful:
                logger.info(
                    f"  ✅ {result.do_email}: {result.dataset_name} ({result.data_fraction*100}% data)"
                )

        if upload_result.failure_count > 0:
            logger.error(f"❌ {upload_result.failure_count} dataset uploads failed")
            for result in upload_result.failed:
                logger.error(
                    f"  ❌ {result.do_email}: {result.error or 'Unknown error'}"
                )
