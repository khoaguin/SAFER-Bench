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
    AccuracyBreakdown,
    PerOptionAccuracy,
    OptionDistribution,
    ConfusionMatrix,
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

        # Pattern to match dataset blocks from formatted output with leading spaces
        # 🔍 QA Dataset: pubmedqa
        # ────────────────────────────────────────────────────────────
        #   Total Questions:     2
        #   Answered Questions:  2
        #   Accuracy:            0.0000 (0.00%)
        #   Mean Querying Time:  6.48s
        dataset_pattern = r"QA Dataset:\s*(\w+)\s*\n.*?Total Questions:\s+(\d+)\s*\n.*?Answered Questions:\s+(\d+)\s*\n.*?Accuracy:\s+([\d.]+).*?\n.*?Mean Querying Time:\s+([\d.nan]+)s?"

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

    def parse_accuracy_breakdown(self, stdout: str) -> Dict[str, Dict]:
        """Parse accuracy breakdown from DS server stdout.

        Args:
            stdout: DS server stdout string

        Returns:
            Dictionary with accuracy breakdown per dataset
        """
        accuracy_breakdowns = {}

        # Pattern to match accuracy breakdown section for each dataset
        # Accuracy Breakdown: {dataset}
        # Per-Option Accuracy:
        #   Option A: 0.7500 (3/4)
        #   Option B: 0.5000 (2/4)
        # ...
        # Option Distribution (Predicted):
        #   A: 25.0%, B: 37.5%, C: 25.0%, D: 12.5%
        # Confusion Matrix:
        #        A    B    C    D
        # A      3    1    0    0
        # ...

        # Split by "Accuracy Breakdown:" to get each dataset's section
        sections = stdout.split("Accuracy Breakdown: ")

        for section in sections[1:]:  # Skip first empty split
            lines = section.strip().split("\n")
            if not lines:
                continue

            dataset_name = lines[0].strip()
            accuracy_breakdowns[dataset_name] = {
                "per_option": {},
                "option_distribution": {},
                "confusion_matrix": {},
            }

            # Parse per-option accuracy
            per_option_pattern = r"  Option ([A-Z]): ([\d.]+) \((\d+)/(\d+)\)"
            for line in lines:
                match = re.match(per_option_pattern, line)
                if match:
                    option = match.group(1)
                    accuracy = float(match.group(2))
                    correct = int(match.group(3))
                    total = int(match.group(4))
                    accuracy_breakdowns[dataset_name]["per_option"][option] = {
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": total,
                    }

            # Parse option distribution
            dist_pattern = r"  ([A-Z]): ([\d.]+)%"
            for line in lines:
                if "Option Distribution" in line:
                    continue
                matches = re.findall(dist_pattern, line)
                for match in matches:
                    option = match[0]
                    percentage = float(match[1]) / 100.0  # Convert to 0-1 range
                    accuracy_breakdowns[dataset_name]["option_distribution"][option] = (
                        percentage
                    )

            # Parse confusion matrix
            # Find the confusion matrix section
            matrix_start = -1
            for i, line in enumerate(lines):
                if "Confusion Matrix:" in line:
                    matrix_start = i
                    break

            if matrix_start >= 0 and matrix_start + 2 < len(lines):
                # Next line after "Confusion Matrix:" should be the header with options
                header_line = lines[matrix_start + 1].strip()
                options = header_line.split()

                # Following lines are the matrix rows
                for i in range(matrix_start + 2, len(lines)):
                    line = lines[i].strip()
                    if not line or not line[0].isalpha():
                        break

                    parts = line.split()
                    if len(parts) < len(options) + 1:
                        continue

                    expected_opt = parts[0]
                    counts = [int(x) for x in parts[1 : len(options) + 1]]

                    accuracy_breakdowns[dataset_name]["confusion_matrix"][
                        expected_opt
                    ] = {}
                    for j, predicted_opt in enumerate(options):
                        accuracy_breakdowns[dataset_name]["confusion_matrix"][
                            expected_opt
                        ][predicted_opt] = counts[j]

        return accuracy_breakdowns

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

        # Parse accuracy breakdown
        accuracy_breakdown_dict = {}
        if fedrag_results.ds_server_result.status == "success":
            accuracy_breakdown_dict = self.parse_accuracy_breakdown(
                fedrag_results.ds_server_result.stdout
            )
            logger.debug(
                f"Parsed accuracy breakdown for {len(accuracy_breakdown_dict)} datasets"
            )

        # Convert to Pydantic models
        per_dataset_metrics = {
            name: DatasetMetrics(**metrics)
            for name, metrics in per_dataset_metrics_dict.items()
        }
        overall_metrics = OverallMetrics(**overall_metrics_dict)

        # Convert accuracy breakdown to Pydantic models
        accuracy_breakdown_models = {}
        for dataset_name, breakdown_data in accuracy_breakdown_dict.items():
            # Build per-option list
            per_option_list = [
                PerOptionAccuracy(
                    option=option,
                    accuracy=data["accuracy"],
                    correct=data["correct"],
                    total=data["total"],
                )
                for option, data in breakdown_data["per_option"].items()
            ]

            # Calculate option counts from percentages
            total_predictions = sum(
                breakdown_data["per_option"][opt]["correct"]
                + sum(breakdown_data["confusion_matrix"].get(opt, {}).values())
                - breakdown_data["per_option"].get(opt, {}).get("correct", 0)
                for opt in breakdown_data["per_option"].keys()
            )
            if total_predictions == 0:
                total_predictions = sum(
                    breakdown_data["per_option"][opt]["total"]
                    for opt in breakdown_data["per_option"].keys()
                )

            option_counts = {
                opt: int(
                    breakdown_data["option_distribution"].get(opt, 0.0)
                    * total_predictions
                )
                for opt in breakdown_data["option_distribution"].keys()
            }

            # Build AccuracyBreakdown model
            accuracy_breakdown_models[dataset_name] = AccuracyBreakdown(
                overall_accuracy=per_dataset_metrics_dict.get(dataset_name, {}).get(
                    "accuracy", 0.0
                ),
                per_option=per_option_list,
                option_distribution=OptionDistribution(
                    option_counts=option_counts,
                    option_percentages=breakdown_data["option_distribution"],
                ),
                confusion_matrix=ConfusionMatrix(
                    matrix=breakdown_data["confusion_matrix"]
                ),
            )

        # Build complete metrics model
        metrics: BenchmarkMetrics = BenchmarkMetrics(
            benchmark_metadata=BenchmarkMetadata(
                benchmark_id=federation_info.benchmark_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                configuration=OmegaConf.to_container(self.cfg, resolve=False),
            ),
            federation=FederationMetrics(
                num_data_owners=federation_info.num_data_owners,
                data_owners=[
                    {
                        "email": do.email,
                        "dataset": do.get_dataset_name(),
                        "distribution_strategy": do.distribution_strategy,
                        "datasets": do.datasets,
                    }
                    for do in federation_info.data_owners
                ],
                aggregator=federation_info.aggregator,
                network_key=federation_info.network_key,
            ),
            results=ResultsMetrics(
                per_dataset=per_dataset_metrics,
                overall=overall_metrics,
                accuracy_breakdown=accuracy_breakdown_models
                if accuracy_breakdown_models
                else None,
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
