from typing import List, Dict, Any
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class ExperimentConfig(BaseModel):
    """Single experiment configuration."""

    retriever: str = "FAISS-IVF"
    merger: str = "RRF"
    llm: str = "Qwen2.5-0.5B"
    privacy: str = "none"
    num_data_owners: int = 2
    approval_rate: float = 1.0
    top_k: int = 5


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    baseline: ExperimentConfig = Field(default_factory=ExperimentConfig)
    dimensions_to_vary: Dict[str, List[Any]] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Convert baseline dict to ExperimentConfig
        if "baseline" in data:
            data["baseline"] = ExperimentConfig(**data["baseline"])

        return cls(**data)
