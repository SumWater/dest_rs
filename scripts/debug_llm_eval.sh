#!/usr/bin/env bash
# Quick debug: run LLM evaluator on just 1 sample to see raw model outputs.
# Usage: bash scripts/debug_llm_eval.sh [MODEL_PATH]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${1:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
CONFIG="configs/v3/b6v3_dest_rs.json"

# Only evaluate on B6v3, max 2 samples
python scripts/evaluate_strategy_control_llm.py \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --jsonl outputs/v3/b6v3_dest_rs/swap_samples_valid.jsonl \
    --max-samples 2 \
    --out outputs/v3/llm_eval_debug.json

echo ""
echo "=== Raw responses ==="
python3 -c "
import json
with open('outputs/v3/llm_eval_debug.json') as f:
    d = json.load(f)
for i, row in enumerate(d['details']):
    print(f'--- Sample {i} ---')
    print(f'target: {row[\"target_strategy\"]}')
    print(f'utterance: {row[\"utterance\"][:200]}')
    print(f'raw_response: {repr(row.get(\"raw_response\", \"MISSING\"))}')
    print()
"