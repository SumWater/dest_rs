#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
ROOT="output/need/casino_augmented_new_fix_seed42/b9_prefix_then_lora"
CHECKPOINT="output/other/casino_augmented_new_fix_seed42/b9_prefix_then_lora"
OUT="reports/profile_ablation"
mkdir -p "${OUT}"

python scripts/generate_profile_ablation.py \
  --config "${ROOT}/run_config.json" \
  --checkpoint "${CHECKPOINT}" \
  --label-map "${ROOT}/label_map.json" \
  --reference-p1 "${ROOT}/swap_samples_valid.jsonl" \
  --out "${OUT}/p0_no_profile.jsonl" \
  --num-examples 30

python scripts/evaluate_strategy_multilabel_llm.py \
  --model-path "${MODEL_PATH}" \
  --jsonl "${OUT}/p0_no_profile.jsonl" \
  --out "${OUT}/p0_blind_multilabel.json" \
  --blind-profile

python scripts/evaluate_strategy_multilabel_llm.py \
  --model-path "${MODEL_PATH}" \
  --jsonl "${ROOT}/swap_samples_valid.jsonl" \
  --out "${OUT}/p1_blind_multilabel.json" \
  --blind-profile

# P1 generation and its judge result already exist; record their canonical paths.
python -c 'import json; from pathlib import Path; Path("reports/profile_ablation/manifest.json").write_text(json.dumps({"p0_generation":"reports/profile_ablation/p0_no_profile.jsonl","p0_evaluation":"reports/profile_ablation/p0_blind_multilabel.json","p1_generation":"output/need/casino_augmented_new_fix_seed42/b9_prefix_then_lora/swap_samples_valid.jsonl","p1_evaluation":"reports/profile_ablation/p1_blind_multilabel.json","judge_blinded_to_profile":True,"controlled_difference":"generator_include_profile"}, indent=2), encoding="utf-8")'
