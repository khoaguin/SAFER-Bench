set dotenv-load := true
set shell := ["bash", "-c"]

# ---------------------------------------------------------------------------------------------------------------------

[private]
_red := '\033[1;31m'
[private]
_cyan := '\033[1;36m'
[private]
_green := '\033[1;32m'
[private]
_yellow := '\033[1;33m'
[private]
_nc := '\033[0m'

# ---------------------------------------------------------------------------------------------------------------------

@default:
    just --list

# ---------------------------------------------------------------------------------------------------------------------

# Development: quick run with data subsets (fast iteration)
run-dry:
    @echo -e "{{_cyan}}🚀 Running SAFER-Bench in SUBSET mode (fast)...{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true

# Production: full benchmark with complete datasets
run:
    @echo -e "{{_green}}🔬 Running SAFER-Bench in FULL mode (complete)...{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false

# Development sweep: quick parameter exploration with subsets
sweep-dry:
    @echo -e "{{_cyan}}🔍 Running parameter sweep in SUBSET mode...{{_nc}}"
    uv run python src/safer_bench/main.py --multirun \
        dataset.use_subset=true \
        federation=local_2do,local_3do \
        federation.approval.percentage=0.5,1.0

# Production sweep: full parameter exploration (long-running)
sweep:
    @echo -e "{{_green}}🔍 Running parameter sweep in FULL mode...{{_nc}}"
    uv run python src/safer_bench/main.py --multirun \
        dataset.use_subset=false \
        federation=local_2do,local_3do,local_4do \
        federation.approval.percentage=0.25,0.5,0.75,1.0

# Show current configuration
show-config:
    @echo -e "{{_cyan}}📋 Current configuration:{{_nc}}"
    uv run safer-bench --cfg job

# Clean build artifacts
clean:
    @echo -e "{{_yellow}}🧹 Cleaning up...{{_nc}}"
    rm -rf outputs/ multirun/ logs/ .pytest_cache/ .mypy_cache/ .ruff_cache/ safer_bench_network/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo -e "{{_green}}✅ Cleaned{{_nc}}"