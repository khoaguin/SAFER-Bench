from typing import List
from safer_bench.config.benchmark_config import BenchmarkConfig, ExperimentConfig


class ExperimentMatrix:
    """Generate experiment combinations from config."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        experiments = []
        baseline = self.config.baseline

        # Add baseline
        experiments.append(baseline)

        # Generate variations
        for dim, values in self.config.dimensions_to_vary.items():
            for value in values:
                if value == getattr(baseline, dim):
                    continue  # Skip baseline value

                exp = baseline.model_copy()
                setattr(exp, dim, value)
                experiments.append(exp)

        return experiments
