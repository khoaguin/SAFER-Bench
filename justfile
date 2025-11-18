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
run-dry federation="separated_2do" llm="smollm_1.7b" *args="":
    @echo -e "{{_cyan}}🚀 Running SAFER-Bench in SUBSET mode (fast) with {{federation}} and {{llm}}...{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 federation={{federation}} llm={{llm}} {{args}}

# Production: full benchmark with complete datasets
run federation="separated_2do" llm="smollm_1.7b" *args="":
    @echo -e "{{_green}}🔬 Running SAFER-Bench in FULL mode (complete) with {{federation}} and {{llm}}...{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false federation={{federation}} llm={{llm}} {{args}}

# Development: run with inspection (keep directories for debugging)
run-dry-inspect federation="separated_2do" llm="smollm_1.7b" *args="":
    @echo -e "{{_yellow}}🔍 Running SAFER-Bench with directories kept for inspection ({{federation}}, {{llm}})...{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 runtime.clean=false federation={{federation}} llm={{llm}} {{args}}

# Production: run with inspection (keep directories for debugging)
run-inspect federation="separated_2do" llm="smollm_1.7b" *args="":
    @echo -e "{{_yellow}}🔍 Running SAFER-Bench FULL mode with directories kept for inspection ({{federation}}, {{llm}})...{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false runtime.clean=false federation={{federation}} llm={{llm}} {{args}}

# Run all federation configs sequentially (SUBSET mode - fast)
run-all-federations-dry llm="smollm_1.7b" privacy="none":
    @echo -e "{{_cyan}}🔄 Running all federations in SUBSET mode with LLM={{llm}}, privacy={{privacy}}{{_nc}}"
    @echo -e "{{_cyan}}Running: separated_2do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 federation=separated_2do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_cyan}}Running: hybrid_2do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 federation=hybrid_2do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_cyan}}Running: centralized_1do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 federation=centralized_1do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_cyan}}Running: hybrid_3do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 federation=hybrid_3do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_cyan}}Running: hybrid_4do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=true qa.num_questions=2 federation=hybrid_4do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_green}}✅ All federation configs completed!{{_nc}}"

# Run all federation configs sequentially (FULL mode - long-running)
run-all-federations llm="smollm_1.7b" privacy="none":
    @echo -e "{{_green}}🔄 Running all federations in FULL mode with LLM={{llm}}, privacy={{privacy}}{{_nc}}"
    @echo -e "{{_green}}Running: separated_2do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false federation=separated_2do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_green}}Running: hybrid_2do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false federation=hybrid_2do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_green}}Running: centralized_1do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false federation=centralized_1do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_green}}Running: hybrid_3do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false federation=hybrid_3do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_yellow}}⏳ Waiting 5 seconds before next run...{{_nc}}"
    @sleep 5
    @echo -e "{{_green}}Running: hybrid_4do{{_nc}}"
    uv run python src/safer_bench/main.py dataset.use_subset=false federation=hybrid_4do llm={{llm}} privacy={{privacy}}
    @echo -e "{{_green}}✅ All federation configs completed!{{_nc}}"

# # Development sweep: quick parameter exploration with subsets
# sweep-dry:
#     @echo -e "{{_cyan}}🔍 Running parameter sweep in SUBSET mode...{{_nc}}"
#     uv run python src/safer_bench/main.py --multirun \
#         dataset.use_subset=true \
#         federation=separated_2do,hybrid_3do \
#         federation.approval.percentage=0.5,1.0

# # Production sweep: full parameter exploration (long-running)
# sweep:
#     @echo -e "{{_green}}🔍 Running parameter sweep in FULL mode...{{_nc}}"
#     uv run python src/safer_bench/main.py --multirun \
#         dataset.use_subset=false \
#         federation=separated_2do,hybrid_3do,hybrid_4do \
#         federation.approval.percentage=0.25,0.5,0.75,1.0

# Show current configuration
show-config:
    @echo -e "{{_cyan}}📋 Current configuration:{{_nc}}"
    uv run safer-bench --cfg job

# Clean running artifacts and caches
clean:
    @echo -e "{{_yellow}}🧹 Cleaning up...{{_nc}}"
    rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
    rm -rf local_syftbox_network/ safer_bench_network/ multirun/ logs/
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".syftbox" -exec rm -rf {} + 2>/dev/null || true
    @echo -e "{{_green}}✅ Cleaned{{_nc}}"

clean-output:
    @echo -e "{{_yellow}}🧹 Cleaning up output directories...{{_nc}}"
    rm -rf outputs/*
    @echo -e "{{_green}}✅ Output directories cleaned{{_nc}}"