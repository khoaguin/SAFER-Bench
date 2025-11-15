#!/usr/bin/env bash
# Simple script to run all federation configs sequentially
# Automatically discovers all .yaml files in configs/federation and runs them

echo "🚀 Running all federation configurations..."
echo ""

# Get script directory and locate configs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${SCRIPT_DIR}/../configs/federation"

# Loop through all .yaml files in configs/federation
for config_file in "${CONFIGS_DIR}"/*.yaml; do
    if [ -f "$config_file" ]; then
        # Extract federation name (filename without .yaml extension)
        federation_name=$(basename "$config_file" .yaml)

        echo "──────────────────────────────────────────────────────"
        echo "📋 Running: ${federation_name}"
        echo "──────────────────────────────────────────────────────"

        # Run the federation config
        just run "${federation_name}"

        # Clean up after each run
        just clean

        # Small delay between runs
        sleep 2
        echo ""
    fi
done

echo "✅ All federation configs completed!"
