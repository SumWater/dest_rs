#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
ROOT="output/need/casino_augmented_new_fix_seed42/b9_prefix_then_lora"
CKPT="output/other/casino_augmented_new_fix_seed42/b9_prefix_then_lora"
OUT="reports/b9_topk_pilot"; mkdir -p "${OUT}"
python scripts/generate_b9_topk_candidates.py --config "${ROOT}/run_config.json" --checkpoint "${CKPT}" \
  --label-map "${ROOT}/label_map.json" --reference "${ROOT}/swap_samples_valid.jsonl" \
  --out "${OUT}/candidates_k8.jsonl" --num-contexts 10 --k 8 --seed 42000 --temperature 0.8 --top-p 0.9
python scripts/evaluate_candidates_multilabel_batch.py --model-path "${MODEL_PATH}" \
  --jsonl "${OUT}/candidates_k8.jsonl" --out "${OUT}/candidates_k8_multilabel.json" --batch-size "${JUDGE_BATCH_SIZE:-16}"
python scripts/summarize_b9_topk.py --candidate-eval "${OUT}/candidates_k8_multilabel.json" \
  --top1-eval "reports/profile_ablation/p1_blind_multilabel.json" --out "${OUT}/summary.json"
