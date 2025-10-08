"""
MetricsCollector: Collects and aggregates metrics from FedRAG benchmark execution.
"""

import re
from datetime import datetime
from typing_extensions import Dict, Any

from omegaconf import DictConfig, OmegaConf
from loguru import logger

from safer_bench.models import (
    FederationInfo,
    FedRAGExecutionResult,
    BenchmarkMetrics,
    BenchmarkMetadata,
    FederationMetrics,
    ResultsMetrics,
    DatasetMetrics,
    OverallMetrics,
    ExecutionMetrics,
)


class MetricsCollector:
    """Collects metrics from FedRAG execution and generates reports."""

    def __init__(self, cfg: DictConfig):
        """Initialize the metrics collector.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg

    def parse_ds_stdout(self, stdout: str) -> Dict[str, Any]:
        """Parse DS server stdout to extract per-dataset metrics.

        Args:
            stdout: DS server stdout string

        Returns:
            Dictionary with per-dataset metrics
        """
        metrics = {}

        # Pattern to match dataset blocks
        # QA Dataset: pubmedqa
        # Total Questions: 2
        # Answered Questions: 0
        # Accuracy: 0.0
        # Mean Querying Time: nan
        dataset_pattern = r"QA Dataset: (\w+)\s*\n.*?Total Questions: (\d+)\s*\n.*?Answered Questions: (\d+)\s*\n.*?Accuracy: ([\d.]+)\s*\n.*?Mean Querying Time: ([\d.nan]+)"

        matches = re.finditer(dataset_pattern, stdout, re.MULTILINE | re.DOTALL)

        for match in matches:
            dataset_name = match.group(1)
            total_questions = int(match.group(2))
            answered_questions = int(match.group(3))
            accuracy = float(match.group(4))
            mean_time_str = match.group(5)

            # Handle 'nan' values
            mean_time = None if mean_time_str == "nan" else float(mean_time_str)

            metrics[dataset_name] = {
                "total_questions": total_questions,
                "answered_questions": answered_questions,
                "accuracy": accuracy,
                "mean_query_time": mean_time,
            }

        return metrics

    def calculate_overall_metrics(
        self, per_dataset_metrics: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Calculate overall metrics across all datasets.

        Args:
            per_dataset_metrics: Per-dataset metrics dictionary

        Returns:
            Overall aggregated metrics
        """
        total_questions = sum(
            m["total_questions"] for m in per_dataset_metrics.values()
        )
        total_answered = sum(
            m["answered_questions"] for m in per_dataset_metrics.values()
        )

        # Calculate weighted accuracy (by number of answered questions)
        weighted_accuracy = 0.0
        if total_answered > 0:
            accuracy_sum = sum(
                m["accuracy"] * m["answered_questions"]
                for m in per_dataset_metrics.values()
            )
            weighted_accuracy = accuracy_sum / total_answered

        # Calculate mean query time (excluding None/nan values)
        query_times = [
            m["mean_query_time"]
            for m in per_dataset_metrics.values()
            if m["mean_query_time"] is not None
        ]
        mean_query_time = sum(query_times) / len(query_times) if query_times else None

        return {
            "total_questions": total_questions,
            "total_answered": total_answered,
            "weighted_accuracy": weighted_accuracy,
            "mean_query_time": mean_query_time,
        }

    async def collect(
        self,
        federation_info: FederationInfo,
        fedrag_results: FedRAGExecutionResult,
        start_time: datetime,
        end_time: datetime,
    ) -> BenchmarkMetrics:
        """Collect all metrics from the benchmark execution.

        Args:
            federation_info: Federation configuration and metadata
            fedrag_results: FedRAG execution results
            start_time: Benchmark start time
            end_time: Benchmark end time

        Returns:
            Complete benchmark metrics
        """
        logger.info("Collecting benchmark metrics...")

        # Parse DS server output
        per_dataset_metrics_dict = {}
        if fedrag_results.ds_server_result.status == "success":
            per_dataset_metrics_dict = self.parse_ds_stdout(
                fedrag_results.ds_server_result.stdout
            )
            logger.debug(f"Parsed metrics for {len(per_dataset_metrics_dict)} datasets")
        else:
            logger.warning("DS server failed, skipping metrics parsing")

        # Calculate overall metrics
        overall_metrics_dict = self.calculate_overall_metrics(per_dataset_metrics_dict)

        # Convert to Pydantic models
        per_dataset_metrics = {
            name: DatasetMetrics(**metrics)
            for name, metrics in per_dataset_metrics_dict.items()
        }
        overall_metrics = OverallMetrics(**overall_metrics_dict)

        # Build complete metrics model
        metrics: BenchmarkMetrics = BenchmarkMetrics(
            benchmark_metadata=BenchmarkMetadata(
                benchmark_id=federation_info.benchmark_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                configuration=OmegaConf.to_container(self.cfg, resolve=True),
            ),
            federation=FederationMetrics(
                num_data_owners=federation_info.num_data_owners,
                data_owners=[
                    {
                        "email": do.email,
                        "dataset": do.dataset,
                        "data_fraction": do.data_fraction,
                    }
                    for do in federation_info.data_owners
                ],
                aggregator=federation_info.aggregator,
                network_key=federation_info.network_key,
            ),
            results=ResultsMetrics(
                per_dataset=per_dataset_metrics,
                overall=overall_metrics,
            ),
            execution=ExecutionMetrics(
                total_jobs=fedrag_results.total_jobs,
                successful_jobs=fedrag_results.successful_jobs,
                failed_jobs=fedrag_results.failed_jobs,
                success_rate=fedrag_results.success_rate,
                ds_server_status=fedrag_results.ds_server_result.status,
            ),
        )

        logger.success("✅ Metrics collection complete")
        return metrics
