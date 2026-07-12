#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"

for seed in 42 43 44; do
  root="output/need/casino_augmented_new_fix_seed${seed}"
  for experiment in b3_prefix_only b4_prefix_lora b9_prefix_then_lora; do
    input="${root}/${experiment}/swap_samples_valid.jsonl"
    output="${root}/${experiment}/strategy_eval_multilabel.json"
    echo "Evaluating seed=${seed} experiment=${experiment}"
    python scripts/evaluate_strategy_multilabel_llm.py \
      --model-path "${MODEL_PATH}" \
      --jsonl "${input}" \
      --out "${output}" \
      --resume
  done
done
