#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
DEFAULT_DATASET_DIR="./CaSiNo-main/data"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
if [ -z "${DATASET_TAG:-}" ]; then
    if [ "$DATASET_DIR" = "$DEFAULT_DATASET_DIR" ]; then
        DATASET_TAG="casino_original"
    else
        DATASET_TAG="$(basename "$DATASET_DIR")"
    fi
fi
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-45}"
FAIL_COUNT=0
TOTAL_START=$(date '+%Y-%m-%d %H:%M:%S')

NEED="output/need/${DATASET_TAG}"
OTHER="output/other/${DATASET_TAG}"

# ══════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════
run_step() {
    local name="$1"; shift
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "[开始] $name — $(date '+%H:%M:%S')"
    echo "────────────────────────────────────────────────────────────────"
    if "$@"; then
        echo "[完成] $name — $(date '+%H:%M:%S')"
        return 0
    else
        echo "[失败] $name — 退出码 $? — $(date '+%H:%M:%S')"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

train_if_needed() {
    local name="$1"
    local config="$2"
    local exp_name="$3"
    local required_artifact="${4:-}"
    if [ "$FORCE_RETRAIN" != "1" ] && [ -f "${NEED}/${exp_name}/metrics.json" ]; then
        if [ -z "$required_artifact" ] || [ -e "$required_artifact" ]; then
            echo "[跳过] $name — ${NEED}/${exp_name}/metrics.json 已存在"
            return 0
        fi
        echo "[重跑] $name — metrics 已存在，但依赖产物缺失: $required_artifact"
    fi
    run_step "$name — 训练" python train.py --config "$config" \
        --dataset-tag "$DATASET_TAG" --dataset-dir "$DATASET_DIR"
}

train_base_if_needed() {
    local name="$1"
    local config="$2"
    local exp_name="$3"
    local artifact="$4"
    train_if_needed "$name" "$config" "$exp_name" "$artifact"
    if [ ! -e "$artifact" ]; then
        echo "[警告] $name 完成后仍未找到关键产物: $artifact"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

train_dependent_if_ready() {
    local name="$1"
    local config="$2"
    local exp_name="$3"
    local dependency="$4"
    local artifact="$5"
    if [ ! -e "$dependency" ]; then
        echo "[跳过] $name — 依赖产物不存在: $dependency"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
    train_if_needed "$name" "$config" "$exp_name" "$artifact"
    if [ ! -e "$artifact" ]; then
        echo "[警告] $name 完成后仍未找到关键产物: $artifact"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

llm_eval() {
    local name="$1"
    local config="$2"
    local exp_name="$3"
    if [ ! -f "${NEED}/${exp_name}/swap_samples_valid.jsonl" ]; then
        echo "[跳过] $name LLM eval — swap_samples_valid.jsonl 不存在"
        return 0
    fi
    if [ "$FORCE_EVAL" != "1" ] && [ -f "${NEED}/${exp_name}/strategy_eval_llm.json" ]; then
        echo "[跳过] $name LLM eval — ${NEED}/${exp_name}/strategy_eval_llm.json 已存在"
        return 0
    fi
    run_step "$name — LLM eval" python scripts/evaluate_strategy_control_llm.py \
        --model-path "$MODEL_PATH" \
        --config "$config" \
        --dataset-tag "$DATASET_TAG" \
        --dataset-dir "$DATASET_DIR" \
        --jsonl "${NEED}/${exp_name}/swap_samples_valid.jsonl" \
        --out "${NEED}/${exp_name}/strategy_eval_llm.json" \
        --max-samples "$EVAL_MAX_SAMPLES"
}

echo "════════════════════════════════════════════════════════════════════════"
echo " run_all.sh — 全部实验一键运行"
echo " 数据集标签: ${DATASET_TAG}"
echo " 开始时间: $TOTAL_START"
echo "════════════════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: 无依赖的基线实验
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1: 基线实验 (B2, B3, B4, B5, B6)                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_base_if_needed "B2 LoRA only"        configs/b2_lora_only.json        b2_lora_only        "${OTHER}/b2_lora_only/lora_adapter/adapter_config.json"
train_base_if_needed "B3 Prefix only"      configs/b3_prefix_only.json      b3_prefix_only      "${OTHER}/b3_prefix_only/prefix_bank.pt"
train_base_if_needed "B4 Prefix+LoRA"      configs/b4_prefix_lora.json      b4_prefix_lora      "${OTHER}/b4_prefix_lora/prefix_bank.pt"
train_base_if_needed "B5 Prefix+LoRA+Orth" configs/b5_prefix_lora_orth.json b5_prefix_lora_orth "${OTHER}/b5_prefix_lora_orth/prefix_bank.pt"
train_base_if_needed "B6 DeSTRS"           configs/b6_dest_rs.json          b6_dest_rs          "${OTHER}/b6_dest_rs/prefix_bank.pt"

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: 有依赖的实验
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 2: 依赖实验 (B7, B9 ← B3; B8 ← B2)                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_dependent_if_ready "B7 warm-start"  configs/b7_dest_rs_warm.json     b7_dest_rs_warm          "${OTHER}/b3_prefix_only/prefix_bank.pt"                  "${OTHER}/b7_dest_rs_warm/prefix_bank.pt"
train_dependent_if_ready "B9 Prefix→LoRA" configs/b9_prefix_then_lora.json b9_p2_lora_frozen_prefix "${OTHER}/b3_prefix_only/prefix_bank.pt"                  "${OTHER}/b9_p2_lora_frozen_prefix/lora_adapter/adapter_config.json"
train_dependent_if_ready "B8 LoRA→Prefix" configs/b8_lora_then_prefix.json b8_p2_prefix_frozen_lora "${OTHER}/b2_lora_only/lora_adapter/adapter_config.json" "${OTHER}/b8_p2_prefix_frozen_lora/prefix_bank.pt"

# ══════════════════════════════════════════════════════════════════════════
# Phase 3: LLM 策略控制评估
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 3: LLM 策略控制评估                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

llm_eval "B2"     configs/b2_lora_only.json          b2_lora_only
llm_eval "B3"     configs/b3_prefix_only.json        b3_prefix_only
llm_eval "B4"     configs/b4_prefix_lora.json        b4_prefix_lora
llm_eval "B5"     configs/b5_prefix_lora_orth.json   b5_prefix_lora_orth
llm_eval "B6"     configs/b6_dest_rs.json            b6_dest_rs
llm_eval "B7"     configs/b7_dest_rs_warm.json    b7_dest_rs_warm
llm_eval "B8"     configs/b8_lora_then_prefix.json       b8_p2_prefix_frozen_lora
llm_eval "B9"     configs/b9_prefix_then_lora.json b9_p2_lora_frozen_prefix

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
TOTAL_END=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " 全部完成 — $TOTAL_START → $TOTAL_END"
echo " 失败步骤数: $FAIL_COUNT"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

python3 -c "
import json
from pathlib import Path

dataset_tag = '${DATASET_TAG}'
need = Path('output/need') / dataset_tag

experiments = [
    ('B2  LoRA only',        'b2_lora_only'),
    ('B3  Prefix only',      'b3_prefix_only'),
    ('B4  Prefix+LoRA',      'b4_prefix_lora'),
    ('B5  +Orth',            'b5_prefix_lora_orth'),
    ('B6  DeSTRS',           'b6_dest_rs'),
    ('B7  warm-start',       'b7_dest_rs_warm'),
    ('B8  LoRA→Prefix',      'b8_p2_prefix_frozen_lora'),
    ('B9  Prefix→LoRA',      'b9_p2_lora_frozen_prefix'),
]

mapping = {
    'elicit-pref': 'elicit-pref',
    'self-need': 'need-expression', 'other-need': 'need-expression',
    'no-need': 'need-expression', 'uv-part': 'need-expression',
    'promote-coordination': 'collaboration', 'vouch-fair': 'collaboration',
    'small-talk': 'rapport', 'showing-empathy': 'rapport',
}

header = f\"{'实验':<22} {'valid_ppl':>10} {'test_ppl':>10} {'9-class':>10} {'4-class':>10}\"
print(header)
print('─' * len(header))

for name, exp_name in experiments:
    out = need / exp_name
    mf = out / 'metrics.json'
    ef = out / 'strategy_eval_llm.json'

    vppl = tppl = acc9 = acc4 = '—'

    if mf.exists():
        m = json.loads(mf.read_text())
        best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
        vppl = f\"{best.get('valid_ppl', 0):.2f}\"
        tppl = f\"{best.get('test_ppl', 0):.2f}\"

    if ef.exists():
        e = json.loads(ef.read_text())
        acc9 = f\"{e['overall_accuracy']:.4f}\"
        correct = total = 0
        for row in e['details']:
            if row['pred_strategy'] is None:
                continue
            if mapping.get(row['target_strategy']) == mapping.get(row['pred_strategy']):
                correct += 1
            total += 1
        acc4 = f'{correct/total:.4f}' if total > 0 else '—'

    print(f'{name:<22} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10}')

print()
print(f'数据集: {dataset_tag}')
"
