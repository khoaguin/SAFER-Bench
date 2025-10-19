"""Results reporting and summary generation for SaferBench.

This module handles saving benchmark results and generating formatted
summaries with detailed accuracy breakdowns, confusion matrices, and
performance metrics.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from safer_bench.models import BenchmarkMetrics, AccuracyBreakdown, DataOwnerInfo


class ResultsReporter:
    """Generate and save benchmark results and summaries."""

    def __init__(self, config):
        """Initialize the results reporter.

        Args:
            config: Benchmark configuration object
        """
        self.cfg = config

    def save_results(
        self, metrics: BenchmarkMetrics, benchmark_id: str
    ) -> tuple[Path, Path]:
        """Save benchmark results to disk.

        Args:
            metrics: Complete benchmark metrics
            benchmark_id: Unique identifier for this benchmark run

        Returns:
            Tuple of (metrics_file_path, summary_file_path)
        """
        output_dir = Path(self.cfg.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed metrics as JSON
        metrics_file = output_dir / f"metrics_{benchmark_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)

        # Save summary as markdown
        summary_file = output_dir / f"summary_{benchmark_id}.md"
        with open(summary_file, "w") as f:
            f.write(self.generate_summary(metrics, benchmark_id))

        return metrics_file, summary_file

    def generate_summary(self, metrics: BenchmarkMetrics, benchmark_id: str) -> str:
        """Generate a markdown summary of benchmark results.

        Args:
            metrics: Complete benchmark metrics
            benchmark_id: Unique identifier for this benchmark run

        Returns:
            Markdown formatted summary with detailed breakdowns
        """
        overall = metrics.results.overall
        per_dataset = metrics.results.per_dataset
        execution = metrics.execution
        accuracy_breakdown = metrics.results.accuracy_breakdown

        # Build summary sections
        header = self._format_header(benchmark_id)
        config_section = self._format_configuration()
        overall_section = self._format_overall_results(overall, metrics)
        dataset_section = self._format_per_dataset_results(
            per_dataset, accuracy_breakdown
        )
        execution_section = self._format_execution_summary(execution)
        distribution_section = self._format_data_distribution()

        return "\n".join(
            [
                header,
                config_section,
                overall_section,
                dataset_section,
                execution_section,
                distribution_section,
            ]
        )

    def _format_header(self, benchmark_id: str) -> str:
        """Format the markdown header."""
        return f"""# SaferBench Results Summary

## Benchmark ID: {benchmark_id}
"""

    def _format_configuration(self) -> str:
        """Format the configuration section."""
        return f"""### Configuration
- Federation: {self.cfg.federation.name}
- Data Owners: {self.cfg.federation.num_data_owners}
- Retriever: {self.cfg.retriever.type}
- LLM: {self.cfg.llm.model}
- Merger: {self.cfg.merger.type}
- Dataset Mode: {"Subset" if self.cfg.dataset.use_subset else "Full"}
- Approval Rate: {self.cfg.federation.approval.percentage * 100}%
"""

    def _format_overall_results(self, overall, metrics: BenchmarkMetrics) -> str:
        """Format overall results section."""
        mean_time_str = (
            f"{overall.mean_query_time:.2f}s" if overall.mean_query_time else "N/A"
        )
        return f"""### Overall Results
- **Weighted Accuracy**: {overall.weighted_accuracy:.2%}
- **Total Questions**: {overall.total_answered}/{overall.total_questions} answered
- **Mean Query Time**: {mean_time_str}
- **Total Runtime**: {metrics.benchmark_metadata.duration_seconds:.2f}s
"""

    def _format_per_dataset_results(
        self,
        per_dataset: Dict,
        accuracy_breakdown: Optional[Dict[str, AccuracyBreakdown]],
    ) -> str:
        """Format per-dataset results with detailed breakdowns."""
        sections = ["### Per-Dataset Results\n"]

        for dataset_name, dataset_metrics in per_dataset.items():
            mean_time = dataset_metrics.mean_query_time
            mean_time_str = f"{mean_time:.2f}s" if mean_time is not None else "N/A"

            # Basic stats
            sections.append(f"#### {dataset_name}")
            sections.append(
                f"- **Accuracy**: {dataset_metrics.accuracy:.2%} "
                f"({dataset_metrics.answered_questions}/{dataset_metrics.total_questions} questions)"
            )
            sections.append(f"- **Mean Query Time**: {mean_time_str}\n")

            # Add detailed breakdown if available
            if accuracy_breakdown and dataset_name in accuracy_breakdown:
                breakdown = accuracy_breakdown[dataset_name]
                sections.append(self._format_accuracy_breakdown(breakdown))

        return "\n".join(sections)

    def _format_accuracy_breakdown(self, breakdown: AccuracyBreakdown) -> str:
        """Format detailed accuracy breakdown for a dataset."""
        sections = []

        # Per-option accuracy table
        sections.append("**Per-Option Accuracy:**\n")
        sections.append("| Option | Accuracy | Correct/Total |")
        sections.append("|--------|----------|---------------|")
        for opt_acc in breakdown.per_option:
            sections.append(
                f"| {opt_acc.option} | {opt_acc.accuracy:.2%} | {opt_acc.correct}/{opt_acc.total} |"
            )

        # Option distribution
        sections.append("\n**Option Distribution (Predicted):**")
        dist_items = [
            f"{opt}: {pct:.1%}"
            for opt, pct in breakdown.option_distribution.option_percentages.items()
        ]
        sections.append("  " + ", ".join(dist_items))

        # Confusion matrix
        sections.append("\n**Confusion Matrix:**\n")
        sections.append(self._format_confusion_matrix(breakdown.confusion_matrix))

        return "\n".join(sections)

    def _format_confusion_matrix(self, confusion_matrix) -> str:
        """Format confusion matrix as markdown table."""
        matrix = confusion_matrix.matrix
        if not matrix:
            return "_No confusion matrix available_"

        # Get all options (row and column labels)
        all_options = sorted(set(matrix.keys()))

        # Build table header
        header = "|   | " + " | ".join(all_options) + " |"
        separator = "|" + "|".join(["---"] * (len(all_options) + 1)) + "|"

        # Build table rows
        rows = []
        for expected_opt in all_options:
            row_values = [
                str(matrix[expected_opt].get(pred_opt, 0)) for pred_opt in all_options
            ]
            rows.append(f"| {expected_opt} | " + " | ".join(row_values) + " |")

        return "\n".join([header, separator] + rows)

    def _format_execution_summary(self, execution) -> str:
        """Format execution summary section."""
        return f"""### Execution Summary
- **Jobs**: {execution.successful_jobs}/{execution.total_jobs} successful ({execution.success_rate:.1%})
- **DS Server**: {execution.ds_server_status}
"""

    def _format_data_distribution(self) -> str:
        """Generate dynamic data distribution summary."""
        if not hasattr(self, "data_owners") or not self.data_owners:
            return "### Data Distribution\n_See configuration for details_"

        lines = ["### Data Distribution"]
        for i, do_info in enumerate(self.data_owners, 1):
            lines.append(f"- DO{i}: {do_info.dataset} ({do_info.data_fraction * 100}%)")
        return "\n".join(lines)

    def set_data_owners(self, data_owners: list[DataOwnerInfo]) -> None:
        """Set data owners for distribution formatting.

        Args:
            data_owners: List of DataOwnerInfo objects
        """
        self.data_owners = data_owners
