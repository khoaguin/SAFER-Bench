"""Bootstrap confidence intervals for SAFER-Bench v2 experiment results.

Reads metrics JSON files, reconstructs per-question binary outcomes from
confusion matrices, and computes 95% bootstrap CIs for per-dataset and
weighted overall accuracy.

Usage:
    uv run python scripts/bootstrap_ci.py
    uv run python scripts/bootstrap_ci.py --results-dir results/v2 --n-resamples 10000
    uv run python scripts/bootstrap_ci.py --output results/v2/bootstrap_cis.csv
"""

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_metrics(path: Path) -> dict:
    """Load a metrics JSON file."""
    with open(path) as f:
        return json.load(f)


def extract_binary_outcomes(confusion_matrix: dict) -> list[int]:
    """Reconstruct per-question binary outcomes (1=correct, 0=wrong) from a confusion matrix.

    The confusion matrix is a nested dict: matrix[expected][predicted] = count.
    Diagonal entries (expected == predicted) are correct.
    """
    outcomes: list[int] = []
    matrix = confusion_matrix["matrix"]
    for expected, predictions in matrix.items():
        for predicted, count in predictions.items():
            correct = 1 if expected == predicted else 0
            outcomes.extend([correct] * count)
    return outcomes


def bootstrap_ci(
    outcomes_by_dataset: dict[str, list[int]],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap CIs for per-dataset accuracy and weighted overall accuracy.

    Resampling is stratified by dataset to preserve the weighting structure.

    Returns dict with per-dataset and overall results, each containing:
        accuracy, ci_lower, ci_upper, n_questions
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2  # 0.025 for 95% CI

    # Pre-convert to arrays
    arrays = {
        name: np.array(outcomes) for name, outcomes in outcomes_by_dataset.items()
    }

    # Per-dataset bootstrap
    per_dataset_resamples: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        n = len(arr)
        if n == 0:
            continue
        # Generate all resample indices at once: shape (n_resamples, n)
        indices = rng.integers(0, n, size=(n_resamples, n))
        resampled_accs = arr[indices].mean(axis=1)
        per_dataset_resamples[name] = resampled_accs

    # Weighted overall accuracy per resample
    # weight = number of answered questions per dataset
    dataset_names = list(per_dataset_resamples.keys())
    weights = np.array([len(arrays[name]) for name in dataset_names], dtype=float)
    total_weight = weights.sum()

    # Stack per-dataset resamples: shape (n_datasets, n_resamples)
    stacked = np.stack([per_dataset_resamples[name] for name in dataset_names])
    # Weighted mean: (n_resamples,)
    overall_resamples = (stacked * weights[:, None]).sum(axis=0) / total_weight

    # Build results
    results = {"per_dataset": {}, "overall": {}}

    for name in dataset_names:
        arr = arrays[name]
        resampled = per_dataset_resamples[name]
        results["per_dataset"][name] = {
            "accuracy": float(arr.mean()),
            "ci_lower": float(np.percentile(resampled, alpha * 100)),
            "ci_upper": float(np.percentile(resampled, (1 - alpha) * 100)),
            "n_questions": len(arr),
        }

    # Overall
    results["overall"] = {
        "accuracy": float(
            (weights * np.array([arrays[n].mean() for n in dataset_names])).sum()
            / total_weight
        ),
        "ci_lower": float(np.percentile(overall_resamples, alpha * 100)),
        "ci_upper": float(np.percentile(overall_resamples, (1 - alpha) * 100)),
        "n_questions": int(total_weight),
    }

    # Cross-dataset std (robustness measure)
    dataset_accs = [arrays[name].mean() for name in dataset_names]
    results["cross_dataset_std"] = float(np.std(dataset_accs))

    return results


def find_metrics_files(results_dir: Path) -> list[Path]:
    """Find all metrics JSON files, excluding .mlflow duplicates."""
    all_files = sorted(results_dir.rglob("metrics_*.json"))
    return [f for f in all_files if ".mlflow" not in str(f)]


def extract_experiment_info(metrics: dict) -> dict:
    """Extract model name, federation config, and merger type from metrics."""
    config = metrics["benchmark_metadata"]["configuration"]
    federation_name = config["federation"]["name"]
    llm_model = config["llm"]["model"]
    # Shorten model name
    model_short = llm_model.split("/")[-1].replace("-GGUF", "").replace("-Instruct", "")
    merger_type = config.get("merger", {}).get("type", "RRF")
    return {
        "model": model_short,
        "federation": federation_name,
        "merger": merger_type,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for SAFER-Bench results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/v2"),
        help="Root directory containing metrics JSON files",
    )
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results/v2/bootstrap_cis.csv)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = Path.cwd() / results_dir

    output_path = args.output or results_dir / "bootstrap_cis.csv"

    # Find all metrics files
    metrics_files = find_metrics_files(results_dir)
    if not metrics_files:
        print(f"No metrics files found in {results_dir}")
        sys.exit(1)
    print(f"Found {len(metrics_files)} metrics files\n")

    # Process each experiment
    all_results: list[dict] = []
    dataset_order = ["pubmedqa", "bioasq", "mmlu_med", "medqa_us", "medmcqa"]

    for mf in metrics_files:
        metrics = load_metrics(mf)
        info = extract_experiment_info(metrics)
        breakdown = metrics["results"]["accuracy_breakdown"]

        # Reconstruct binary outcomes per dataset
        outcomes_by_dataset: dict[str, list[int]] = {}
        for ds_name in dataset_order:
            if ds_name not in breakdown:
                continue
            cm = breakdown[ds_name].get("confusion_matrix")
            if cm is None:
                continue
            outcomes = extract_binary_outcomes(cm)
            if outcomes:
                outcomes_by_dataset[ds_name] = outcomes

        if not outcomes_by_dataset:
            print(f"  SKIP {mf.name} — no confusion matrices")
            continue

        # Bootstrap
        ci_results = bootstrap_ci(
            outcomes_by_dataset,
            n_resamples=args.n_resamples,
            ci=args.ci,
            seed=args.seed,
        )

        row = {
            "model": info["model"],
            "federation": info["federation"],
            "merger": info["merger"],
            "overall_acc": ci_results["overall"]["accuracy"],
            "overall_ci_lower": ci_results["overall"]["ci_lower"],
            "overall_ci_upper": ci_results["overall"]["ci_upper"],
            "overall_n": ci_results["overall"]["n_questions"],
            "cross_dataset_std": ci_results["cross_dataset_std"],
        }
        for ds_name in dataset_order:
            if ds_name in ci_results["per_dataset"]:
                ds = ci_results["per_dataset"][ds_name]
                row[f"{ds_name}_acc"] = ds["accuracy"]
                row[f"{ds_name}_ci_lower"] = ds["ci_lower"]
                row[f"{ds_name}_ci_upper"] = ds["ci_upper"]
                row[f"{ds_name}_n"] = ds["n_questions"]

        all_results.append(row)

    # Sort by model, then federation
    all_results.sort(key=lambda r: (r["model"], r["federation"], r["merger"]))

    # Print summary table
    print(
        f"{'Model':<30} {'Federation':<18} {'Merger':<12} {'Accuracy':>10} {'95% CI':>20} {'N':>5} {'Cross-DS Std':>12}"
    )
    print("-" * 115)
    for r in all_results:
        ci_str = f"({r['overall_ci_lower']:.2%}, {r['overall_ci_upper']:.2%})"
        print(
            f"{r['model']:<30} {r['federation']:<18} {r['merger']:<12} "
            f"{r['overall_acc']:>9.2%} {ci_str:>20} {r['overall_n']:>5} "
            f"{r['cross_dataset_std']:>11.4f}"
        )

    # Print per-dataset detail for each experiment
    print("\n\nPer-Dataset Breakdown:")
    print(
        f"{'Model':<30} {'Federation':<18} {'Dataset':<12} {'Accuracy':>10} {'95% CI':>20} {'N':>5}"
    )
    print("-" * 100)
    for r in all_results:
        for ds_name in dataset_order:
            acc_key = f"{ds_name}_acc"
            if acc_key in r:
                ci_str = (
                    f"({r[f'{ds_name}_ci_lower']:.2%}, {r[f'{ds_name}_ci_upper']:.2%})"
                )
                print(
                    f"{r['model']:<30} {r['federation']:<18} {ds_name:<12} "
                    f"{r[acc_key]:>9.2%} {ci_str:>20} {r[f'{ds_name}_n']:>5}"
                )

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(all_results[0].keys())
    with open(output_path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in all_results:
            values = [str(r.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")

    print(f"\nCSV written to {output_path}")


if __name__ == "__main__":
    main()
