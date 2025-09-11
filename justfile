set dotenv-load := true

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

# Run the SAFER-Bench pipeline with default config
run config="configs/baseline.yaml":
    echo "{{_cyan}}🚀 Running SAFER-Bench pipeline...{{_nc}}"
    uv run safer-bench --config {{config}}

clean:
    echo "{{_yellow}}🧹 Cleaning up...{{_nc}}"
    rm -rf outputs/* logs/*
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    echo "{{_green}}✅ Cleaned{{_nc}}"