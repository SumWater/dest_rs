#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
REFERENCE="output/need/casino_augmented_new_fix_seed42/b9_prefix_then_lora/swap_samples_valid.jsonl"
OUT="reports/verbalizer_upper_bound"

python scripts/generate_verbalizer_upper_bound.py \
  --model-path "${MODEL_PATH}" --reference "${REFERENCE}" --out-dir "${OUT}" \
  --batch-size "${BATCH_SIZE:-8}" --max-new-tokens 40

for condition in v0_context v1_strategy_name v2_strategy_definition v3_profile_definition v4_profile_definition_example; do
  python scripts/evaluate_strategy_multilabel_llm.py \
    --model-path "${MODEL_PATH}" \
    --jsonl "${OUT}/${condition}.jsonl" \
    --out "${OUT}/${condition}_multilabel.json" \
    --blind-profile
done
