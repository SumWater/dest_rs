#!/usr/bin/env bash
set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════
# run_grad_parallel.sh — 路线 A 梯度路由并行实验（双卡调度）
#
# GPU 0: b6_grad (主实验)
# GPU 1: b6_grad_no_orth (消融1)
# 先跑完的 GPU 接 b6_grad_no_cls (消融2)
# 每个实验训练完自动做 LLM 评估
# ══════════════════════════════════════════════════════════════════════════

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
DATASET_DIR="${DATASET_DIR:-./CaSiNo-main/data}"
DATASET_TAG="${DATASET_TAG:-casino_augmented_fix_b6}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-270}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

NEED="output/need/${DATASET_TAG}"

TOTAL_START=$(date '+%Y-%m-%d %H:%M:%S')

# ══════════════════════════════════════════════════════════════════════════
# Per-experiment function (train + eval)
# ══════════════════════════════════════════════════════════════════════════
run_experiment() {
    local gpu="$1"
    local exp_name="$2"
    local config="configs/${exp_name}.json"
    local log_file="log_${exp_name}.log"
    local metrics="${NEED}/${exp_name}/metrics.json"
    local jsonl="${NEED}/${exp_name}/swap_samples_valid.jsonl"
    local eval_out="${NEED}/${exp_name}/strategy_eval_llm.json"

    echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 开始"

    # ── 训练 ──
    if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$metrics" ]; then
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 训练跳过 (metrics.json 已存在)"
    else
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 训练中..."
        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --config "$config" \
            --dataset-tag "$DATASET_TAG" \
            --dataset-dir "$DATASET_DIR" >> "$log_file" 2>&1
        if [ $? -ne 0 ]; then
            echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 训练失败!"
            return 1
        fi
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 训练完成"
    fi

    # ── LLM 评估 ──
    if [ ! -f "$jsonl" ]; then
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} swap_samples 不存在，跳过评估"
        return 1
    fi
    if [ "$FORCE_EVAL" != "1" ] && [ -f "$eval_out" ]; then
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 评估跳过 (strategy_eval_llm.json 已存在)"
    else
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} LLM 评估中..."
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_strategy_control_llm.py \
            --model-path "$MODEL_PATH" \
            --config "$config" \
            --dataset-tag "$DATASET_TAG" \
            --dataset-dir "$DATASET_DIR" \
            --jsonl "$jsonl" \
            --out "$eval_out" \
            --max-samples "$EVAL_MAX_SAMPLES" >> "$log_file" 2>&1
        if [ $? -ne 0 ]; then
            echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 评估失败!"
            return 1
        fi
        echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 评估完成"
    fi

    echo "[$(date '+%H:%M:%S')] GPU${gpu}: ${exp_name} 全部完成"
    return 0
}

echo "════════════════════════════════════════════════════════════════════════"
echo " run_grad_parallel.sh — 双卡并行梯度路由实验"
echo " 开始时间:       $TOTAL_START"
echo " 数据目录:       ${DATASET_DIR}"
echo " 输出标签:       ${DATASET_TAG}"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "[调度] GPU0 → b6_grad, GPU1 → b6_grad_no_orth"
echo "[调度] 先完成者 → b6_grad_no_cls"
echo ""

FAIL=0

# ══════════════════════════════════════════════════════════════════════════
# Round 1: 并行启动 b6_grad (GPU0) + b6_grad_no_orth (GPU1)
# ══════════════════════════════════════════════════════════════════════════
run_experiment 0 b6_grad &
PID0=$!

run_experiment 1 b6_grad_no_orth &
PID1=$!

# ══════════════════════════════════════════════════════════════════════════
# 等待两个 GPU 都完成（或任一失败）
# ══════════════════════════════════════════════════════════════════════════
wait $PID0 || FAIL=1
DONE0=$(date '+%H:%M:%S')
echo "[$DONE0] GPU0 结束"

wait $PID1 || FAIL=1
DONE1=$(date '+%H:%M:%S')
echo "[$DONE1] GPU1 结束"

# ══════════════════════════════════════════════════════════════════════════
# Round 2: 跑 b6_grad_no_cls（用空闲 GPU，优先 GPU0）
# ══════════════════════════════════════════════════════════════════════════
# 用 GPU0 跑最后一个实验
run_experiment 0 b6_grad_no_cls
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    FAIL=1
fi

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
TOTAL_END=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " 全部完成 — $TOTAL_START → $TOTAL_END"
echo " 失败: $FAIL"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

python3 -c "
import json
from pathlib import Path

supp_need = Path('output/need') / '${DATASET_TAG}'

mapping = {
    'elicit-pref': 'information', 'express-pref': 'information', 'no-need': 'information',
    'small-talk': 'affective', 'self-need': 'substantive', 'other-need': 'substantive',
    'uv-part': 'substantive', 'vouch-fair': 'affective', 'promote-coordination': 'affective',
}

# 新实验
print('═' * 110)
print('  路线 A 梯度路由实验结果')
print('═' * 110)

experiments = [
    ('B6_grad (完整梯度路由)',   'b6_grad'),
    ('B6_grad_no_orth (无orth)', 'b6_grad_no_orth'),
    ('B6_grad_no_cls  (无cls)',  'b6_grad_no_cls'),
]

header = f\"{'Experiment':<30} {'epoch':>6} {'valid_ppl':>10} {'test_ppl':>10} {'9-class':>10} {'4-class':>10}\"
print(header)
print('─' * 110)

for name, exp_name in experiments:
    mf = supp_need / exp_name / 'metrics.json'
    ef = supp_need / exp_name / 'strategy_eval_llm.json'

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

    print(f'{name:<30} {epoch:>6} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10}')

# 对比已有实验
print()
print('─' * 110)
print('  对比：已有实验')
print('─' * 110)

orig_need = Path('output/need') / 'casino_augmented'
fix_need  = Path('output/need') / 'casino_augmented_fix_b6'

comparisons = [
    ('B3  Prefix-only       ', orig_need / 'b3_prefix_only'),
    ('B4  Prefix+LoRA       ', orig_need / 'b4_prefix_lora'),
    ('B5  +Orth (naive)     ', orig_need / 'b5_prefix_lora_orth'),
    ('B6_fix (naive fixed)  ', fix_need  / 'b6_fix'),
    ('B6_orth_1.0 (naive)   ', fix_need  / 'b6_orth_1.0'),
    ('B6_cls_5.0  (naive)   ', fix_need  / 'b6_cls_5.0'),
    ('B7  warm-start        ', orig_need / 'b7_dest_rs_warm'),
]

for name, d in comparisons:
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
    print(f'{name:<26} {epoch:>6} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10}')

print()
print('═' * 110)
"
