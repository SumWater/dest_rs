#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports/strategy_classifier
MODES="${MODES:-context response context_response}"
for mode in ${MODES}; do
  echo "Running ${mode}"
  python scripts/train_strategy_classifier.py \
    --data-dir CaSiNo-main/data/split \
    --mode "${mode}" \
    --seed 42 \
    --out "reports/strategy_classifier/${mode}.json"
done
