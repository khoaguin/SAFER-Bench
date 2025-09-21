# Comprehensive Implementation Plan: Baseline Case (local_2do + qwen_small)

## Baseline Configuration

```yaml
# Simplest case configuration
federation: local_2do  # 2 DOs: statpearls (10%), textbooks (10%)
retriever: faiss_ivf   # Default FAISS IVF
merger: rrf            # Reciprocal Rank Fusion
llm: qwen_small        # Qwen 0.5B model
privacy: none          # 100% approval
qa.num_questions: 10   # Start with small set
```

## Implementation Files Structure

```
src/safer_bench/
├── benchmark_runner.py     # Main orchestrator
├── network_manager.py      # SyftBox network setup
├── data_distributor.py     # Dataset distribution logic
├── fedrag_adapter.py       # Parameter injection
├── metrics_collector.py    # Results & metrics
└── main.py                 # Entry point
```

## Detailed Implementation Plan

### 1. BenchmarkRunner (benchmark_runner.py)

```python
import asyncio
from pathlib import Path
import hydra
from omegaconf import DictConfig
from loguru import logger

class BenchmarkRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.network_manager = NetworkManager(cfg)
        self.data_distributor = DataDistributor(cfg)
        self.fedrag_adapter = FedRAGAdapter(cfg)
        self.metrics_collector = MetricsCollector(cfg)

    async def run(self):
        """Main benchmark workflow"""
        # Stage 1: Setup network
        logger.info("Stage 1: Setting up local SyftBox network")
        network_info = await self.network_manager.setup()

        # Stage 2: Distribute datasets
        logger.info("Stage 2: Distributing datasets to DOs")
        await self.data_distributor.distribute_datasets(network_info)

        # Stage 3: Prepare fedrag with injected parameters
        logger.info("Stage 3: Preparing fedrag with benchmark parameters")
        fedrag_project = self.fedrag_adapter.prepare_project()

        # Stage 4: Submit jobs
        logger.info("Stage 4: Submitting fedrag jobs to DOs")
        jobs = await self.submit_jobs(fedrag_project, network_info)

        # Stage 5: Auto-approve based on approval rate
        logger.info("Stage 5: Processing approvals")
        await self.process_approvals(jobs)

        # Stage 6: Run fedrag
        logger.info("Stage 6: Running federated RAG")
        results = await self.run_fedrag(network_info)

        # Stage 7: Collect metrics
        logger.info("Stage 7: Collecting metrics")
        metrics = self.metrics_collector.collect(results)

        return metrics
```

### 2. NetworkManager (network_manager.py)

```python
from syft_rds.orchestra import setup_rds_server, remove_rds_stack_dir

class NetworkManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.aggregator = cfg.federation.aggregator
        self.datasites = self._get_datasites()

    def _get_datasites(self):
        """Extract DO emails from config"""
        return [
            self.cfg.federation.do1.email,
            self.cfg.federation.do2.email
        ]

    async def setup(self):
        """Setup local SyftBox network"""
        # Clean previous runs
        remove_rds_stack_dir(Path("."), key="safer_bench_network")

        # Setup DS (aggregator)
        ds_stack = setup_rds_server(
            email=self.aggregator,
            root_dir=Path("."),
            key="safer_bench_network"
        )

        # Setup DOs
        do_stacks = []
        for do_email in self.datasites:
            do_stack = setup_rds_server(
                email=do_email,
                root_dir=Path("."),
                key="safer_bench_network"
            )
            do_stacks.append(do_stack)

        return {
            "ds": ds_stack,
            "dos": do_stacks,
            "clients": [stack.init_session(stack.client.email)
                       for stack in do_stacks]
        }
```

### 3. DataDistributor (data_distributor.py)

```python
class DataDistributor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.datasets_dir = Path("datasets")

    async def distribute_datasets(self, network_info):
        """Distribute dataset fractions to each DO"""
        do_configs = [self.cfg.federation.do1, self.cfg.federation.do2]

        for i, (do_client, do_config) in enumerate(
            zip(network_info["clients"], do_configs)
        ):
            # Load dataset
            dataset_path = self.datasets_dir / do_config.dataset

            # Sample data based on fraction
            sampled_data = self._sample_data(
                dataset_path,
                do_config.data_fraction
            )

            # Create Syft dataset
            do_client.dataset.create(
                name=do_config.dataset,
                path=sampled_data["private"],
                mock_path=sampled_data["mock"],
                description_path=sampled_data["readme"]
            )

    def _sample_data(self, dataset_path, fraction):
        """Sample fraction of dataset"""
        # Load all chunks
        chunk_dir = dataset_path / "chunk"
        all_chunks = list(chunk_dir.glob("*.jsonl"))

        # Sample based on fraction
        num_chunks = int(len(all_chunks) * fraction)
        sampled_chunks = all_chunks[:num_chunks]

        # Create temp dirs for sampled data
        temp_dir = Path(f"/tmp/safer_bench_{dataset_path.name}_{fraction}")
        private_dir = temp_dir / "private"
        mock_dir = temp_dir / "mock"

        # Copy sampled chunks
        for chunk in sampled_chunks:
            shutil.copy(chunk, private_dir / "chunk" / chunk.name)
            # Create mock version (first 10% of each chunk)
            self._create_mock(chunk, mock_dir / "chunk" / chunk.name)

        return {
            "private": private_dir,
            "mock": mock_dir,
            "readme": dataset_path / "README.md"
        }
```

### 4. FedRAGAdapter (fedrag_adapter.py)

```python
class FedRAGAdapter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fedrag_path = Path("src/safer_bench/assets/fedrag_v1")

    def prepare_project(self):
        """Inject parameters into fedrag project"""
        # 1. Update retriever.yaml
        self._update_retriever_config()

        # 2. Update pyproject.toml with runtime configs
        self._update_pyproject()

        # 3. Bootstrap fedrag project
        self._bootstrap_project()

        return self.fedrag_path

    def _update_retriever_config(self):
        """Update retriever.yaml with benchmark params"""
        retriever_config = {
            "embedding_model": self.cfg.retriever.embedding_model,
            "embedding_dimension": self.cfg.retriever.embedding_dimension
        }

        yaml_path = self.fedrag_path / "fedrag" / "retriever.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(retriever_config, f)

    def _update_pyproject(self):
        """Add runtime configurations"""
        pyproject_path = self.fedrag_path / "pyproject.toml"
        pyproject = toml.load(pyproject_path)

        # Add benchmark configurations
        pyproject["tool"]["flwr"]["app"]["config"] = {
            "k-nn": self.cfg.retrieval.k_nn,
            "k-rrf": self.cfg.merger.get("k_rrf", 60),
            "server-llm-hfpath": self.cfg.llm.model_path,
            "server-qa-num": self.cfg.qa.num_questions
        }

        toml.dump(pyproject, open(pyproject_path, 'w'))
```

### 5. Main Entry Point (main.py)

```python
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"Starting SaferBench with config: {OmegaConf.to_yaml(cfg)}")

    runner = BenchmarkRunner(cfg)

    # Run benchmark
    metrics = asyncio.run(runner.run())

    # Save results
    output_dir = Path(cfg.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Benchmark complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
```

## Execution Flow for Baseline

```bash
# Run with baseline config
python src/safer_bench/main.py

# Expected output:
# Stage 1: Setting up local SyftBox network ✓
# Stage 2: Distributing datasets (statpearls 10%, textbooks 10%) ✓
# Stage 3: Preparing fedrag with qwen_small ✓
# Stage 4: Submitting jobs to 2 DOs ✓
# Stage 5: Auto-approving (100% approval) ✓
# Stage 6: Running federated RAG ✓
# Stage 7: Metrics - Accuracy: X%, Time: Ys ✓
```

## Key Implementation Notes

1. **Modular Design**: Each component has a single responsibility, making it easy to test and modify independently.

2. **Async/Await Pattern**: Uses Python's async capabilities for parallel operations (DO dataset creation, job submissions).

3. **Configuration-Driven**: All parameters come from Hydra configs, enabling easy parameter sweeps.

4. **Data Scarcity**: The `DataDistributor._sample_data()` method implements the data fraction concept, allowing DOs to have different amounts of data.

5. **Parameter Injection**: The `FedRAGAdapter` modifies fedrag configuration files at runtime without changing the source code.

6. **Metrics Collection**: Results are automatically saved in JSON format for analysis.

## Next Steps

1. Implement each module following this plan
2. Test with baseline configuration (local_2do + qwen_small)
3. Add parameter sweeping capabilities
4. Extend to more complex scenarios (local_3do, local_4do)
5. Add visualization and reporting capabilities

This plan provides a clean, modular implementation that can easily scale to more complex scenarios while starting with the simplest baseline case.