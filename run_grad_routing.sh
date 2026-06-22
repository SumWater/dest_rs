#!/usr/bin/env bash
set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════
# run_grad_routing.sh — 路线 A 梯度路由实验一键运行
#
# 输出目录：output/need/casino_augmented_fix_b6/
#           output/other/casino_augmented_fix_b6/
#
# 实验：B6_grad / B6_grad_no_orth / B6_grad_no_cls
#
# 用法：
#   ./run_grad_routing.sh
#   DATASET_DIR=./augmented_data ./run_grad_routing.sh
#   FORCE_RETRAIN=1 FORCE_EVAL=1 ./run_grad_routing.sh
# ══════════════════════════════════════════════════════════════════════════

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
DEFAULT_DATASET_DIR="./CaSiNo-main/data"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
DATASET_TAG="${DATASET_TAG:-casino_augmented_fix_b6}"

FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-270}"
FAIL_COUNT=0

NEED="output/need/${DATASET_TAG}"
OTHER="output/other/${DATASET_TAG}"

EXPERIMENTS=(
    "b6_grad:B6_grad (完整梯度路由: gen→LoRA, cls→Prefix, orth→both)"
    "b6_grad_no_orth:B6_grad_no_orth (梯度路由 无 orth)"
    "b6_grad_no_cls:B6_grad_no_cls (梯度路由 无 cls)"
)

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

# ══════════════════════════════════════════════════════════════════════════

TOTAL_START=$(date '+%Y-%m-%d %H:%M:%S')

echo "════════════════════════════════════════════════════════════════════════"
echo " run_grad_routing.sh — 路线 A 梯度路由实验"
echo " 数据目录:       ${DATASET_DIR}"
echo " 输出标签:       ${DATASET_TAG}"
echo " 开始时间:       $TOTAL_START"
echo " EVAL_MAX_SAMPLES: ${EVAL_MAX_SAMPLES}"
echo "════════════════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Training
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1: 训练                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

for entry in "${EXPERIMENTS[@]}"; do
    exp_name="${entry%%:*}"
    exp_desc="${entry##*:}"

    if [ "$FORCE_RETRAIN" != "1" ] && [ -f "${NEED}/${exp_name}/metrics.json" ]; then
        echo "[跳过] $exp_desc — ${NEED}/${exp_name}/metrics.json 已存在"
        continue
    fi

    run_step "$exp_desc — 训练" \
        python train.py \
            --config "configs/${exp_name}.json" \
            --dataset-tag "$DATASET_TAG" \
            --dataset-dir "$DATASET_DIR"
done

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: LLM Strategy Evaluation
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 2: LLM 策略控制评估                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

for entry in "${EXPERIMENTS[@]}"; do
    exp_name="${entry%%:*}"
    exp_desc="${entry##*:}"
    jsonl="${NEED}/${exp_name}/swap_samples_valid.jsonl"
    out="${NEED}/${exp_name}/strategy_eval_llm.json"

    if [ ! -f "$jsonl" ]; then
        echo "[跳过] $exp_desc LLM eval — swap_samples_valid.jsonl 不存在（训练可能失败）"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    if [ "$FORCE_EVAL" != "1" ] && [ -f "$out" ]; then
        echo "[跳过] $exp_desc LLM eval — ${out} 已存在"
        continue
    fi

    run_step "$exp_desc — LLM eval" \
        python scripts/evaluate_strategy_control_llm.py \
            --model-path "$MODEL_PATH" \
            --config "configs/${exp_name}.json" \
            --dataset-tag "$DATASET_TAG" \
            --dataset-dir "$DATASET_DIR" \
            --jsonl "$jsonl" \
            --out "$out" \
            --max-samples "$EVAL_MAX_SAMPLES"
done

# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Summary
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
supp_need = Path('output/need') / supp_tag

mapping = {
    'elicit-pref': 'information', 'express-pref': 'information', 'no-need': 'information',
    'small-talk': 'affective', 'self-need': 'substantive', 'other-need': 'substantive',
    'uv-part': 'substantive', 'vouch-fair': 'affective', 'promote-coordination': 'affective',
}

experiments = [
    ('B6_grad (完整梯度路由)',   'b6_grad'),
    ('B6_grad_no_orth (无orth)', 'b6_grad_no_orth'),
    ('B6_grad_no_cls  (无cls)',  'b6_grad_no_cls'),
]

header = f\"{'Experiment':<30} {'epoch':>6} {'valid_ppl':>10} {'test_ppl':>10} {'9-class':>10} {'4-class':>10} {'gen_loss':>10} {'orth_loss':>12} {'cls_loss':>10}\"
print('═' * 120)
print('  路线 A 梯度路由实验结果')
print('═' * 120)
print(header)
print('─' * 120)

for name, exp_name in experiments:
    out_dir = supp_need / exp_name
    mf = out_dir / 'metrics.json'
    ef = out_dir / 'strategy_eval_llm.json'

    epoch = vppl = tppl = acc9 = acc4 = gen_loss = orth_loss = cls_loss = '—'

    if mf.exists():
        m = json.loads(mf.read_text())
        best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
        epoch = str(m.get('best_epoch', '—'))
        vppl = f\"{best.get('valid_ppl', 0):.2f}\"
        tppl = f\"{best.get('test_ppl', 0):.2f}\"
        gen_loss = f\"{best.get('train_gen_loss', 0):.4f}\"
        orth_loss = f\"{best.get('train_orth_loss', 0):.6f}\"
        cls_loss = f\"{best.get('train_cls_loss', 0):.6f}\"

    if ef.exists():
        e = json.loads(ef.read_text())
        acc9 = f\"{e['overall_accuracy']:.4f}\"
        correct = total = 0
        for row in e['details']:
            if row.get('pred_strategy') is None:
                continue
            if mapping.get(row.get('target_strategy','')) == mapping.get(row.get('pred_strategy','')):
                correct += 1
            total += 1
        acc4 = f'{correct/total:.4f}' if total > 0 else '—'

    print(f'{name:<30} {epoch:>6} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10} {gen_loss:>10} {orth_loss:>12} {cls_loss:>10}')

print()
print('─' * 120)
print('  对比：已有实验结果')
print('─' * 120)

orig_tag = 'casino_augmented'
orig_need = Path('output/need') / orig_tag
fix_tag = 'casino_augmented_fix_b6'
fix_need = Path('output/need') / fix_tag

comparisons = [
    ('B3  Prefix-only       ', orig_need / 'b3_prefix_only',      'original'),
    ('B4  Prefix+LoRA       ', orig_need / 'b4_prefix_lora',      'original'),
    ('B5  +Orth (naive)     ', orig_need / 'b5_prefix_lora_orth', 'original'),
    ('B6_fix (naive fixed)  ', fix_need  / 'b6_fix',              'fix'),
    ('B6_orth_1.0 (naive)   ', fix_need  / 'b6_orth_1.0',        'fix'),
    ('B6_cls_5.0  (naive)   ', fix_need  / 'b6_cls_5.0',         'fix'),
    ('B7  warm-start        ', orig_need / 'b7_dest_rs_warm',     'original'),
]

print(f\"{'Experiment':<26} {'epoch':>6} {'valid_ppl':>10} {'test_ppl':>10} {'9-class':>10} {'4-class':>10}\")
for name, d, src in comparisons:
    mf = d / 'metrics.json'
    ef = d / 'strategy_eval_llm.json'
    epoch = vppl = tppl = acc9 = acc4 = '—'
    if mf.exists():
        m = json.loads(mf.read_text())
        best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
        epoch = str(m.get('best_epoch', '—'))
        vppl = f\"{best.get('valid_ppl', 0):.2f}\"
        tppl = f\"{best.get('test_ppl', 0):.2f}\"
    if ef.exists():
        e = json.loads(ef.read_text())
        acc9 = f\"{e['overall_accuracy']:.4f}\"
        correct = total = 0
        for row in e['details']:
            if row.get('pred_strategy') is None:
                continue
            if mapping.get(row.get('target_strategy','')) == mapping.get(row.get('pred_strategy','')):
                correct += 1
            total += 1
        acc4 = f'{correct/total:.4f}' if total > 0 else '—'
    tag = '  ← 原实验' if src == 'original' else '  ← fix系列'
    print(f'{name:<26} {epoch:>6} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10}{tag}')

print()
print('═' * 120)
" 2>&1

echo ""
echo "输出目录: ${NEED}/"
echo "模型权重: ${OTHER}/"
echo "日志文件位于对应实验目录下的 metrics.json 和 strategy_eval_llm.json"
