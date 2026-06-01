#!/usr/bin/env bash
# Run LLM-based strategy control evaluation on all V3 experiments.
# Usage: bash scripts/run_llm_eval.sh [MODEL_PATH]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${1:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
CONFIG="configs/v3/b6v3_dest_rs.json"

echo "================================================================================"
echo " LLM Strategy Control Evaluator"
echo " Model: $MODEL_PATH"
echo " Config: $CONFIG (for few-shot examples)"
echo "================================================================================"
echo ""

# Collect all JSONL files
JSONL_FILES=""
for exp in b2v3_lora_only b3v3_prefix_only b4v3_prefix_lora b5v3_orth b6v3_dest_rs; do
    f="outputs/v3/${exp}/swap_samples_valid.jsonl"
    if [ -f "$f" ]; then
        JSONL_FILES="$JSONL_FILES $f"
    else
        echo "[WARN] Missing: $f"
    fi
done

if [ -z "$JSONL_FILES" ]; then
    echo "No JSONL files found."
    exit 1
fi

python scripts/evaluate_strategy_control_llm.py \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --jsonl $JSONL_FILES \
    --out outputs/v3/llm_strategy_eval.json \
    --few-shot-per-strategy 2

echo ""
echo "Done. Results: outputs/v3/llm_strategy_eval.json"
