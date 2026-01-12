# SAFER-Bench

SAFER-Bench is a comprehensive benchmark framework for evaluating Federated Retrieval-Augmented Generation (RAG) systems with approval-based privacy controls. It enables systematic evaluation of retrieval algorithms, merging strategies, LLMs, and privacy mechanisms in realistic federated settings where Data Owners (DOs) maintain control over their data.

<p align="center">
  <img src="./docs/images/architecture.png" alt="safer-bench-architecture" width="700">
</p>

## Federation Configurations

Federation configs define how data is distributed across Data Owners (DOs):

| Config | DOs | Distribution | Description |
|--------|-----|--------------|-------------|
| `centralized_1do` | 1 | Centralized | Baseline: single DO holds all datasets (StatPearls + Textbooks) |
| `separated_2do` | 2 | Partitioned | Each DO holds one complete dataset (DO1: StatPearls, DO2: Textbooks) |
| `hybrid_2do` | 2 | Hybrid | Each DO holds 50% of both datasets |
| `hybrid_3do` | 3 | Hybrid | Each DO holds ~33% of both datasets |
| `hybrid_4do` | 4 | Hybrid | Each DO holds 25% of both datasets |
| `specialty_7do` | 7 | Specialty | Each DO holds clinical notes from one medical specialty (MIMIC-IV-Note) |

### Distribution Strategies

- **Centralized**: Non-federated baseline for comparison. All data in one location.
- **Separated/Partitioned**: Each DO specializes in one corpus. Simulates domain-specific data silos.
- **Hybrid**: Data distributed evenly across DOs. Simulates collaborative networks with shared data types.
- **Specialty**: Each DO holds clinical notes from a single medical specialty (Cardiology, Oncology, Neurology, Pulmonology, Gastroenterology, Nephrology, General). Simulates real-world healthcare institutions with specialty-specific records.

## Benchmarked Models

| Model | Size | Type | Config |
|-------|------|------|--------|
| SmolLM2-1.7B-Instruct | 1.7B | General | `smollm_1.7b.yaml` |
| BioMistral-7B | 7B | Medical | `biomistral_7b.yaml` |
| Mistral-7B-Instruct-v0.3 | 7B | General | `mistral_7b_instruct.yaml` |
| OpenBioLLM-Llama3-70B (GGUF Q5_K_M) | 70B | Medical | `openbiollm_70b_gguf.yaml` |
| Llama-3.3-70B-Instruct (GGUF Q5_K_M) | 70B | General | `llama3.3_70b_gguf.yaml` |

**Note**: 70B models use GGUF quantization (Q5_K_M) for reduced memory footprint (~50GB) with Metal GPU acceleration on Apple Silicon.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [just](https://github.com/casey/just) (command runner)

## Setup

1. Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd safer-bench
uv sync
```

2. Download datasets by running the notebook `datasets/dataset_download.ipynb`:
```python
# Downloads from https://huggingface.co/datasets/khoaguin/medical-corpus
# Available corpora: statpearls, textbooks
```

## Running Benchmarks

Use `just` commands to run benchmarks. Run `just` to see all available commands.

### Quick Start (Subset Mode)

For development and quick iteration with data subsets:
```bash
# Default: separated_2do federation with smollm_1.7b
just run-dry

# Specify federation and LLM
just run-dry federation=hybrid_2do llm=biomistral_7b
```

### Full Benchmark (Production Mode)

For complete benchmarks with full datasets:
```bash
# Default configuration
just run

# Specify federation and LLM
just run federation=centralized_1do llm=mistral_7b_instruct

# With 70B models (requires ~52GB RAM)
just run federation=separated_2do llm=openbiollm_70b_gguf
```

### Run All Federations

```bash
# Subset mode (fast)
just run-all-federations-dry llm=biomistral_7b

# Full mode (long-running)
just run-all-federations llm=biomistral_7b
```

### Debug Mode (Keep Artifacts)

```bash
# Keep directories for inspection
just run-dry-inspect
just run-inspect
```

### Cleanup

```bash
just clean         # Clean running artifacts and caches
just clean-output  # Clean output directories
```

### Specialty-Based Benchmarks (MIMIC-IV-Note)

Specialty-based distribution partitions clinical notes by medical specialty, modeling real-world healthcare institutions.

**1. Generate specialty mapping (one-time prerequisite):**
```bash
just generate-specialty-mapping              # Both subset and full
just generate-specialty-mapping --subset-only
just generate-specialty-mapping --full-only
```

**2. Run specialty benchmark:**
```bash
# Quick test (subset, 2 questions)
just run-dry federation=specialty_7do llm=smollm_1.7b

# Full benchmark
just run federation=specialty_7do llm=biomistral_7b
```

**Medical Specialties:**
| DO | Specialty | Description |
|----|-----------|-------------|
| DO1 | Cardiology | Heart and cardiovascular conditions |
| DO2 | Oncology | Cancer and tumor-related cases |
| DO3 | Neurology | Brain and nervous system disorders |
| DO4 | Pulmonology | Lung and respiratory conditions |
| DO5 | Gastroenterology | Digestive system disorders |
| DO6 | Nephrology | Kidney and renal conditions |
| DO7 | General | Infectious diseases and miscellaneous |

See `docs/plans/specialty_based_distribution.md` for implementation details.
