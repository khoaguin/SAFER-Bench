"""MLflow integration for SaferBench experiment tracking.

This module provides centralized MLflow tracking for benchmarking experiments,
including metrics, parameters, artifacts, and visualizations.
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, OmegaConf

from safer_bench.models import BenchmarkMetrics, FedRAGExecutionResult


class MLflowTracker:
    """Track SaferBench experiments with MLflow."""

    def __init__(self, cfg: DictConfig, enabled: bool = True):
        """Initialize MLflow tracker.

        Args:
            cfg: Benchmark configuration
            enabled: Whether MLflow tracking is enabled
        """
        self.cfg = cfg
        self.enabled = enabled
        self.mlflow = None
        self.run_started = False

        if self.enabled:
            self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Setup MLflow with experiment tracking."""
        try:
            import mlflow

            self.mlflow = mlflow

            # Set tracking URI if configured
            # Get the Hydra output directory and construct MLflow path
            try:
                from hydra.core.hydra_config import HydraConfig

                # Get Hydra's runtime output directory
                hydra_cfg = HydraConfig.get()
                output_dir = Path(hydra_cfg.runtime.output_dir)

                # Create MLflow tracking directory inside Hydra's output
                tracking_uri = str(output_dir / ".mlflow")
                mlflow.set_tracking_uri(tracking_uri)
                logger.info(f"MLflow tracking URI: {tracking_uri}")
            except Exception as e:
                logger.warning(
                    f"Could not set MLflow tracking URI: {e}. Using default."
                )

            # Set experiment name
            experiment_name = OmegaConf.select(
                self.cfg, "mlflow.experiment_name", default="SaferBench"
            )
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

        except ImportError:
            logger.warning(
                "MLflow not installed. Install with: pip install mlflow. "
                "Disabling MLflow tracking."
            )
            self.enabled = False

    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Start a new MLflow run.

        Args:
            run_name: Name for the MLflow run
            tags: Optional tags to add to the run
        """
        if not self.enabled or not self.mlflow:
            return

        self.mlflow.start_run(run_name=run_name)
        self.run_started = True

        # Add default tags
        default_tags = {
            "framework": "SaferBench",
            "timestamp": datetime.now().isoformat(),
        }
        if tags:
            default_tags.update(tags)

        for key, value in default_tags.items():
            self.mlflow.set_tag(key, value)

        logger.info(f"Started MLflow run: {run_name}")

    def log_config(self, config: DictConfig) -> None:
        """Log configuration parameters to MLflow.

        Args:
            config: Hydra configuration object
        """
        if not self.enabled or not self.mlflow:
            return

        # Convert config to dict and flatten for MLflow
        # Don't resolve interpolations to avoid issues with ${hydra.run.dir}
        # The actual resolved values will be used during execution
        config_dict = OmegaConf.to_container(config, resolve=False)
        self._log_nested_params(config_dict, prefix="")

    def _log_nested_params(
        self, params: Dict[str, Any], prefix: str, max_depth: int = 3
    ) -> None:
        """Recursively log nested parameters to MLflow.

        Args:
            params: Dictionary of parameters
            prefix: Current parameter path prefix
            max_depth: Maximum nesting depth to log
        """
        if max_depth <= 0:
            return

        for key, value in params.items():
            param_name = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._log_nested_params(value, param_name, max_depth - 1)
            elif isinstance(value, (list, tuple)):
                # Log list/tuple as string
                self.mlflow.log_param(param_name, str(value))
            else:
                # Log scalar values
                self.mlflow.log_param(param_name, value)

    def log_fedrag_results(self, fedrag_results: FedRAGExecutionResult) -> None:
        """Log FedRAG execution results to MLflow.

        Args:
            fedrag_results: FedRAG execution results
        """
        if not self.enabled or not self.mlflow:
            return

        # Log execution metrics
        self.mlflow.log_metric("total_jobs", fedrag_results.total_jobs)
        self.mlflow.log_metric("successful_jobs", fedrag_results.successful_jobs)
        self.mlflow.log_metric("failed_jobs", fedrag_results.failed_jobs)
        self.mlflow.log_metric("success_rate", fedrag_results.success_rate)

        # Log DS server status
        self.mlflow.set_tag("ds_server_status", fedrag_results.ds_server_result.status)

    def log_job_artifacts(
        self,
        job_uid: str,
        job_name: str,
        dataset_name: Optional[str],
        job_status: str,
        logs_dir: Path,
    ) -> None:
        """Log individual job artifacts and metadata to MLflow.

        Args:
            job_uid: Job unique identifier (shortened hex)
            job_name: Name of the job
            dataset_name: Dataset name used by the job
            job_status: Job status string
            logs_dir: Path to the job's logs directory
        """
        if not self.enabled or not self.mlflow:
            return

        try:
            # Log job metadata as parameters
            job_prefix = f"job.{job_uid}"
            self.mlflow.log_param(f"{job_prefix}.name", job_name)
            self.mlflow.log_param(
                f"{job_prefix}.dataset", dataset_name if dataset_name else "None"
            )

            # Log job status as tag
            self.mlflow.set_tag(f"{job_prefix}.status", job_status)

            # Log artifacts (stdout.log, stderr.log)
            if logs_dir.exists():
                self.mlflow.log_artifacts(
                    str(logs_dir), artifact_path=f"jobs/{job_uid}/logs"
                )
                logger.debug(f"Logged job {job_uid} artifacts to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log job {job_uid} artifacts to MLflow: {e}")

    def log_benchmark_metrics(self, metrics: BenchmarkMetrics) -> None:
        """Log complete benchmark metrics to MLflow.

        Args:
            metrics: Complete benchmark metrics
        """
        if not self.enabled or not self.mlflow:
            return

        # Log overall metrics
        overall = metrics.results.overall
        self.mlflow.log_metric("overall_accuracy", overall.weighted_accuracy)
        self.mlflow.log_metric("total_questions", overall.total_questions)
        self.mlflow.log_metric("total_answered", overall.total_answered)
        if overall.mean_query_time is not None:
            self.mlflow.log_metric("mean_query_time", overall.mean_query_time)

        # Log per-dataset metrics
        for dataset_name, dataset_metrics in metrics.results.per_dataset.items():
            prefix = f"dataset.{dataset_name}"
            self.mlflow.log_metric(f"{prefix}.accuracy", dataset_metrics.accuracy)
            self.mlflow.log_metric(
                f"{prefix}.total_questions", dataset_metrics.total_questions
            )
            self.mlflow.log_metric(
                f"{prefix}.answered_questions", dataset_metrics.answered_questions
            )
            if dataset_metrics.mean_query_time is not None:
                self.mlflow.log_metric(
                    f"{prefix}.mean_query_time", dataset_metrics.mean_query_time
                )

        # Log accuracy breakdown if available
        if metrics.results.accuracy_breakdown:
            for dataset_name, breakdown in metrics.results.accuracy_breakdown.items():
                prefix = f"breakdown.{dataset_name}"

                # Log per-option accuracy
                for opt_acc in breakdown.per_option:
                    self.mlflow.log_metric(
                        f"{prefix}.option_{opt_acc.option}_accuracy", opt_acc.accuracy
                    )
                    self.mlflow.log_metric(
                        f"{prefix}.option_{opt_acc.option}_correct", opt_acc.correct
                    )

                # Log option distribution
                for (
                    opt,
                    pct,
                ) in breakdown.option_distribution.option_percentages.items():
                    self.mlflow.log_metric(f"{prefix}.predicted_option_{opt}_pct", pct)

        # Log execution metrics
        execution = metrics.execution
        self.mlflow.log_metric("execution_total_jobs", execution.total_jobs)
        self.mlflow.log_metric("execution_successful_jobs", execution.successful_jobs)
        self.mlflow.log_metric("execution_failed_jobs", execution.failed_jobs)
        self.mlflow.log_metric("execution_success_rate", execution.success_rate)

        # Log benchmark metadata
        self.mlflow.log_metric(
            "total_runtime_seconds", metrics.benchmark_metadata.duration_seconds
        )
        self.mlflow.set_tag("benchmark_id", metrics.benchmark_metadata.benchmark_id)
        self.mlflow.set_tag("start_time", metrics.benchmark_metadata.start_time)
        self.mlflow.set_tag("end_time", metrics.benchmark_metadata.end_time)

        logger.info("Logged benchmark metrics to MLflow")

    def log_artifacts(
        self, local_path: Path, artifact_path: Optional[str] = None
    ) -> None:
        """Log artifacts to MLflow.

        Args:
            local_path: Path to local file or directory to log
            artifact_path: Optional path within artifacts directory
        """
        if not self.enabled or not self.mlflow:
            return

        if local_path.is_file():
            self.mlflow.log_artifact(str(local_path), artifact_path)
        elif local_path.is_dir():
            self.mlflow.log_artifacts(str(local_path), artifact_path)

        logger.debug(f"Logged artifact: {local_path} -> {artifact_path}")

    def log_summary_artifacts(
        self, metrics_file: Path, summary_file: Path, output_dir: Path
    ) -> None:
        """Log benchmark summary artifacts to MLflow.

        Args:
            metrics_file: Path to metrics JSON file
            summary_file: Path to summary markdown file
            output_dir: Path to output directory
        """
        if not self.enabled or not self.mlflow:
            return

        # Log metrics JSON
        if metrics_file.exists():
            self.mlflow.log_artifact(str(metrics_file), "results")

        # Log summary markdown
        if summary_file.exists():
            self.mlflow.log_artifact(str(summary_file), "results")

        # Log all output files
        if output_dir.exists():
            for file in output_dir.glob("*.json"):
                self.mlflow.log_artifact(str(file), "results")
            for file in output_dir.glob("*.md"):
                if file != summary_file:  # Avoid duplicate
                    self.mlflow.log_artifact(str(file), "results")

        logger.info("Logged summary artifacts to MLflow")

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        if not self.enabled or not self.mlflow or not self.run_started:
            return

        self.mlflow.set_tag("status", status)
        self.mlflow.end_run()
        self.run_started = False
        logger.info(f"Ended MLflow run with status: {status}")

    def get_tracking_uri(self) -> Optional[str]:
        """Get the current MLflow tracking URI.

        Returns:
            Tracking URI if MLflow is enabled, None otherwise
        """
        if not self.enabled or not self.mlflow:
            return None
        return self.mlflow.get_tracking_uri()

    def get_run_id(self) -> Optional[str]:
        """Get the current MLflow run ID.

        Returns:
            Run ID if a run is active, None otherwise
        """
        if not self.enabled or not self.mlflow or not self.run_started:
            return None
        active_run = self.mlflow.active_run()
        return active_run.info.run_id if active_run else None
