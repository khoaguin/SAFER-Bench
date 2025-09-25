"""
BenchmarkRunner: Main orchestrator for SaferBench framework.
Coordinates the entire federated RAG benchmarking workflow.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf
from loguru import logger

from safer_bench.federation_manager import FederationManager


class BenchmarkRunner:
    """Main orchestrator for running federated RAG benchmarks."""

    def __init__(self, cfg: DictConfig):
        """Initialize the benchmark runner with configuration.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.federation_manager = FederationManager(cfg)
        # self.data_distributor = DataDistributor(cfg)
        # self.fedrag_adapter = FedRAGAdapter(cfg)
        # self.metrics_collector = MetricsCollector(cfg)

        # Track benchmark metadata
        self.start_time = None
        self.end_time = None
        self.benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    async def run(self) -> Dict[str, Any]:
        """Execute the complete benchmark workflow.

        Returns:
            Dictionary containing benchmark metrics and results
        """
        self.start_time = datetime.now()
        logger.info(f"🚀 Starting SaferBench run: {self.benchmark_id}")
        logger.info(f"Configuration: {OmegaConf.to_yaml(self.cfg)}")

        try:
            # Stage 1: Setup federation
            logger.info("=" * 60)
            logger.info("Stage 1/7: Setting up federation")
            logger.info("=" * 60)
            federation_info = await self.federation_manager.setup()
            logger.success(
                f"✅ Federation ready with {len(federation_info['dos'])} data owners"
            )

            # # Stage 2: Distribute datasets
            # logger.info("=" * 60)
            # logger.info("Stage 2/7: Distributing datasets to data owners")
            # logger.info("=" * 60)
            # dataset_info = await self.data_distributor.distribute_datasets(federation_info)
            # logger.success(f"✅ Datasets distributed: {dataset_info}")

            # # Stage 3: Prepare fedrag with injected parameters
            # logger.info("=" * 60)
            # logger.info("Stage 3/7: Preparing FedRAG with benchmark parameters")
            # logger.info("=" * 60)
            # fedrag_project = await self.fedrag_adapter.prepare_project()
            # logger.success(f"✅ FedRAG project prepared at: {fedrag_project}")

            # # Stage 4: Submit jobs to data owners
            # logger.info("=" * 60)
            # logger.info("Stage 4/7: Submitting FedRAG jobs to data owners")
            # logger.info("=" * 60)
            # jobs = await self._submit_jobs(fedrag_project, federation_info)
            # logger.success(f"✅ Submitted {len(jobs)} jobs")

            # # Stage 5: Process approvals based on approval rate
            # logger.info("=" * 60)
            # logger.info("Stage 5/7: Processing job approvals")
            # logger.info("=" * 60)
            # approval_results = await self._process_approvals(jobs, federation_info)
            # logger.success(f"✅ Approved {approval_results['approved']}/{len(jobs)} jobs")

            # # Stage 6: Run federated RAG
            # logger.info("=" * 60)
            # logger.info("Stage 6/7: Running Federated RAG")
            # logger.info("=" * 60)
            # fedrag_results = await self._run_fedrag(federation_info, approval_results)
            # logger.success(f"✅ FedRAG execution complete")

            # # Stage 7: Collect and save metrics
            # logger.info("=" * 60)
            # logger.info("Stage 7/7: Collecting metrics and generating report")
            # logger.info("=" * 60)
            # metrics = await self.metrics_collector.collect(
            #     fedrag_results,
            #     federation_info,
            #     approval_results
            # )
            metrics = {}

            self.end_time = datetime.now()
            metrics["benchmark_metadata"] = {
                "benchmark_id": self.benchmark_id,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_seconds": (self.end_time - self.start_time).total_seconds(),
                "configuration": OmegaConf.to_container(self.cfg),
            }

            # Save results
            await self._save_results(metrics)

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

    async def _submit_jobs(self, fedrag_project: Path, federation_info: Dict) -> list:
        """Submit FedRAG jobs to all data owners.

        Args:
            fedrag_project: Path to prepared FedRAG project
            federation_info: Federation setup information

        Returns:
            List of submitted job objects
        """
        jobs = []

        for i, do_client in enumerate(federation_info["clients"]):
            do_info = self.federation_manager.data_owners[i]

            logger.info(
                f"Submitting job to {do_info.email} for dataset {do_info.dataset}"
            )

            # Submit job using syft_rds client
            job = do_client.job.submit(
                name=f"fedrag_benchmark_{self.benchmark_id}",
                user_code_path=fedrag_project,
                dataset_name=do_info.dataset,
                entrypoint="main.py",
            )

            jobs.append(
                {
                    "job": job,
                    "do_email": do_info.email,
                    "dataset": do_info.dataset,
                    "client": do_client,
                }
            )

        return jobs

    async def _process_approvals(self, jobs: list, federation_info: Dict) -> Dict:
        """Process job approvals based on configured approval rate.

        Args:
            jobs: List of submitted jobs
            federation_info: Federation setup information

        Returns:
            Dictionary with approval results
        """
        approval_rate = self.cfg.federation.approval.percentage
        approved_jobs = []
        rejected_jobs = []

        for job_info in jobs:
            # Simulate approval based on rate
            # In real implementation, this would check DO approval policies
            import random

            if random.random() < approval_rate:
                # Approve job
                logger.info(f"✅ Approving job for {job_info['do_email']}")
                # In real implementation: job_info["client"].job.approve(job_info["job"])
                approved_jobs.append(job_info)
            else:
                logger.info(f"❌ Rejecting job for {job_info['do_email']}")
                rejected_jobs.append(job_info)

        return {
            "approved": len(approved_jobs),
            "rejected": len(rejected_jobs),
            "approved_jobs": approved_jobs,
            "rejected_jobs": rejected_jobs,
            "approval_rate": approval_rate,
        }

    async def _run_fedrag(self, federation_info: Dict, approval_results: Dict) -> Dict:
        """Execute the federated RAG workflow.

        Args:
            federation_info: Federation setup information
            approval_results: Job approval results

        Returns:
            Dictionary with FedRAG execution results
        """
        # This would run the actual FedRAG workflow
        # For now, returning mock results
        logger.info("Running FedRAG with approved data owners...")

        # In real implementation, this would:
        # 1. Start the DS aggregator server
        # 2. Wait for DO clients to connect and run
        # 3. Collect retrieval results
        # 4. Run LLM queries
        # 5. Return results

        await asyncio.sleep(2)  # Simulate execution time

        return {
            "num_queries": self.cfg.qa.num_questions,
            "participating_dos": len(approval_results["approved_jobs"]),
            "retrieval_k": self.cfg.retrieval.k_nn,
            "llm_model": self.cfg.llm.name,
            "merger_algorithm": self.cfg.merger.name,
            # Mock results - will be replaced with actual FedRAG results
            "accuracy": 0.75,
            "mean_query_time": 1.23,
            "total_documents_retrieved": 80,
        }

    async def _save_results(self, metrics: Dict):
        """Save benchmark results to disk.

        Args:
            metrics: Complete benchmark metrics
        """
        output_dir = Path(self.cfg.evaluation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed metrics as JSON
        metrics_file = output_dir / f"metrics_{self.benchmark_id}.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save summary as markdown
        summary_file = output_dir / f"summary_{self.benchmark_id}.md"
        with open(summary_file, "w") as f:
            f.write(self._generate_summary(metrics))

    def _generate_summary(self, metrics: Dict) -> str:
        """Generate a markdown summary of benchmark results.

        Args:
            metrics: Complete benchmark metrics

        Returns:
            Markdown formatted summary
        """
        return f"""# SaferBench Results Summary

## Benchmark ID: {self.benchmark_id}

### Configuration
- Federation: {self.cfg.federation.name}
- Data Owners: {self.cfg.federation.num_data_owners}
- Retriever: {self.cfg.retriever.type}
- LLM: {self.cfg.llm.model}
- Merger: {self.cfg.merger.type}
- Dataset Mode: {"Subset" if self.cfg.dataset.use_subset else "Full"}
- Approval Rate: {self.cfg.federation.approval.percentage * 100}%

### Results
- Accuracy: {metrics.get('accuracy', 'N/A')}
- Mean Query Time: {metrics.get('mean_query_time', 'N/A')}s
- Total Runtime: {metrics['benchmark_metadata']['duration_seconds']:.2f}s

### Data Distribution
{self._format_data_distribution()}
"""

    def _format_data_distribution(self) -> str:
        """Generate dynamic data distribution summary."""
        lines = []
        for i, do_info in enumerate(self.federation_manager.data_owners, 1):
            lines.append(f"- DO{i}: {do_info.dataset} ({do_info.data_fraction * 100}%)")
        return "\n".join(lines)
