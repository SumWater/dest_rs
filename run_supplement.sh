#!/usr/bin/env bash
set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════
# run_supplement.sh — 补充实验一键运行
#
# 输出目录（独立，不影响原有实验结果）：
#   output/need/{DATASET_TAG}/     ← 默认 casino_augmented_fix_b6
#   output/other/{DATASET_TAG}/    ← 默认 casino_augmented_fix_b6
#
# 依赖：
#   B7_ep2 / B7_ep3 依赖 B3 prefix（默认从 casino_augmented 读取）
#   其余实验无依赖，从零训练
#
# 用法：
#   ./run_supplement.sh
#   DATASET_DIR=./augmented_data ./run_supplement.sh
#   DATASET_TAG=my_custom_tag DATASET_DIR=./augmented_data ./run_supplement.sh
#   FORCE_RETRAIN=1 FORCE_EVAL=1 ./run_supplement.sh
# ══════════════════════════════════════════════════════════════════════════

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
DEFAULT_DATASET_DIR="./CaSiNo-main/data"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"

# 补充实验使用独立的 dataset_tag，默认为 casino_augmented_fix_b6
if [ -z "${DATASET_TAG:-}" ]; then
    DATASET_TAG="casino_augmented_fix_b6"
fi

# B7 依赖的 B3 prefix 来源（原始实验结果，不是补充实验目录）
WARM_START_TAG="${WARM_START_TAG:-casino_augmented}"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-270}"
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
    if [ "$FORCE_RETRAIN" != "1" ] && [ -f "${NEED}/${exp_name}/metrics.json" ]; then
        echo "[跳过] $name — ${NEED}/${exp_name}/metrics.json 已存在"
        return 0
    fi
    run_step "$name — 训练" python train.py --config "$config" \
        --dataset-tag "$DATASET_TAG" --dataset-dir "$DATASET_DIR"
}

train_dependent_if_ready() {
    local name="$1"
    local config="$2"
    local exp_name="$3"
    local dependency="$4"
    if [ ! -e "$dependency" ]; then
        echo "[跳过] $name — 依赖产物不存在: $dependency"
        echo "        请确保已先运行原始 B3 实验（run_all.sh）"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
    train_if_needed "$name" "$config" "$exp_name"
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
echo " run_supplement.sh — 补充实验一键运行"
echo " 数据目录:       ${DATASET_DIR}"
echo " 输出标签:       ${DATASET_TAG}"
echo " B3依赖来源:     ${WARM_START_TAG}"
echo " 开始时间:       $TOTAL_START"
echo " EVAL_MAX_SAMPLES: ${EVAL_MAX_SAMPLES}"
echo "════════════════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════
# 第一组：基础修复验证
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  第一组: B6_fix — 基础修复验证（必须做）                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_if_needed "B6_fix (bug修复 + 每步orth + λ_orth=0.1 + λ_cls=1.0)" \
    configs/b6_fix.json b6_fix

# ══════════════════════════════════════════════════════════════════════════
# 第二组：lambda_cls 扫描
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  第二组: lambda_cls 扫描 (0.5, 2.0, 5.0)                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_if_needed "B6_cls_0.5 (λ_cls=0.5)" configs/b6_cls_0.5.json b6_cls_0.5
train_if_needed "B6_cls_2.0 (λ_cls=2.0)" configs/b6_cls_2.0.json b6_cls_2.0
train_if_needed "B6_cls_5.0 (λ_cls=5.0)" configs/b6_cls_5.0.json b6_cls_5.0

# ══════════════════════════════════════════════════════════════════════════
# 第三组：lambda_orth 扫描
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  第三组: lambda_orth 扫描 (0.5, 1.0)                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_if_needed "B6_orth_0.5 (λ_orth=0.5)" configs/b6_orth_0.5.json b6_orth_0.5
train_if_needed "B6_orth_1.0 (λ_orth=1.0)" configs/b6_orth_1.0.json b6_orth_1.0

# ══════════════════════════════════════════════════════════════════════════
# 第四组：B7 扩展训练（依赖 B3 prefix，从 WARM_START_TAG 读取）
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  第四组: B7 扩展训练 (依赖 B3 prefix，来源: ${WARM_START_TAG})   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

B3_PREFIX="output/other/${WARM_START_TAG}/b3_prefix_only/prefix_bank.pt"

train_dependent_if_ready "B7_ep2 (B7 warm-start, 2 epochs)" \
    configs/b7_ep2.json b7_ep2 "$B3_PREFIX"

train_dependent_if_ready "B7_ep3 (B7 warm-start, 3 epochs)" \
    configs/b7_ep3.json b7_ep3 "$B3_PREFIX"

# ══════════════════════════════════════════════════════════════════════════
# 第五组（可选）：降低 LoRA 强度
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  第五组（可选）: B6_lora_r8 — 降低 LoRA 强度                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_if_needed "B6_lora_r8 (LoRA r=8 α=16)" configs/b6_lora_r8.json b6_lora_r8

# ══════════════════════════════════════════════════════════════════════════
# LLM 策略控制评估
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  LLM 策略控制评估                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

llm_eval "B6_fix"      configs/b6_fix.json      b6_fix
llm_eval "B6_cls_0.5"  configs/b6_cls_0.5.json  b6_cls_0.5
llm_eval "B6_cls_2.0"  configs/b6_cls_2.0.json  b6_cls_2.0
llm_eval "B6_cls_5.0"  configs/b6_cls_5.0.json  b6_cls_5.0
llm_eval "B6_orth_0.5" configs/b6_orth_0.5.json b6_orth_0.5
llm_eval "B6_orth_1.0" configs/b6_orth_1.0.json b6_orth_1.0
llm_eval "B7_ep2"      configs/b7_ep2.json      b7_ep2
llm_eval "B7_ep3"      configs/b7_ep3.json      b7_ep3
llm_eval "B6_lora_r8"  configs/b6_lora_r8.json  b6_lora_r8

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

supp_tag = '${DATASET_TAG}'
orig_tag = '${WARM_START_TAG}'

supp_need = Path('output/need') / supp_tag
orig_need = Path('output/need') / orig_tag

experiments = [
    ('B6_fix       ', 'b6_fix'),
    ('B6_cls_0.5   ', 'b6_cls_0.5'),
    ('B6_cls_2.0   ', 'b6_cls_2.0'),
    ('B6_cls_5.0   ', 'b6_cls_5.0'),
    ('B6_orth_0.5  ', 'b6_orth_0.5'),
    ('B6_orth_1.0  ', 'b6_orth_1.0'),
    ('B7_ep2       ', 'b7_ep2'),
    ('B7_ep3       ', 'b7_ep3'),
    ('B6_lora_r8   ', 'b6_lora_r8'),
]

mapping = {
    'elicit-pref': 'elicit-pref',
    'self-need': 'need-expression', 'other-need': 'need-expression',
    'no-need': 'need-expression', 'uv-part': 'need-expression',
    'promote-coordination': 'collaboration', 'vouch-fair': 'collaboration',
    'small-talk': 'rapport', 'showing-empathy': 'rapport',
}

header = f\"{'Experiment':<18} {'valid_ppl':>10} {'test_ppl':>10} {'9-class':>10} {'4-class':>10} {'best_ep':>8}\"
print('═' * 70)
print('  补充实验结果')
print('═' * 70)
print(header)
print('─' * len(header))

for name, exp_name in experiments:
    out = supp_need / exp_name
    mf = out / 'metrics.json'
    ef = out / 'strategy_eval_llm.json'

    vppl = tppl = acc9 = acc4 = best_ep = '—'

    if mf.exists():
        m = json.loads(mf.read_text())
        best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
        vppl = f\"{best.get('valid_ppl', 0):.2f}\"
        tppl = f\"{best.get('test_ppl', 0):.2f}\"
        best_ep = str(m.get('best_epoch', '—'))

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

    print(f'{name:<18} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10} {best_ep:>8}')

# Original results for comparison
print()
print('═' * 70)
print(f'  原始实验结果（{orig_tag}，用于对比）')
print('═' * 70)
print(header)
print('─' * len(header))

orig_experiments = [
    ('B2  LoRA only  ', 'b2_lora_only'),
    ('B3  Prefix only', 'b3_prefix_only'),
    ('B4  Prefix+LoRA', 'b4_prefix_lora'),
    ('B5  +Orth       ', 'b5_prefix_lora_orth'),
    ('B6  DeSTRS (崩) ', 'b6_dest_rs'),
    ('B7  warm-start  ', 'b7_dest_rs_warm'),
    ('B8  LoRA→Prefix ', 'b8_p2_prefix_frozen_lora'),
    ('B9  Prefix→LoRA ', 'b9_p2_lora_frozen_prefix'),
]

for name, exp_name in orig_experiments:
    out = orig_need / exp_name
    mf = out / 'metrics.json'
    ef = out / 'strategy_eval_llm.json'

    vppl = tppl = acc9 = acc4 = best_ep = '—'

    if mf.exists():
        m = json.loads(mf.read_text())
        best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
        vppl = f\"{best.get('valid_ppl', 0):.2f}\"
        tppl = f\"{best.get('test_ppl', 0):.2f}\"
        best_ep = str(m.get('best_epoch', '—'))

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

    print(f'{name:<18} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10} {best_ep:>8}')

print()
print(f'补充实验输出标签: {supp_tag}')
print(f'原始实验输出标签: {orig_tag}')
print(f'数据目录:         ${DATASET_DIR}')
"
