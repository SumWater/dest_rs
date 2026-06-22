#!/usr/bin/env bash
set -uo pipefail

# ══════════════════════════════════════════════════════════════════════════
# run_method1_b9.sh — 方法1 (contrastive) + B9 双卡自动调度
#
# GPU 0: B6_contrastive          (慢，~5 forwards/step)
# GPU 1: B6_contrastive_no_orth  (快，~2 forwards/step)
# 谁先跑完谁接 B9
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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local gpu="$1" exp="$2"
    local config="configs/${exp}.json"
    local logf="log_${exp}.log"
    local metrics="${NEED}/${exp}/metrics.json"
    local jsonl="${NEED}/${exp}/swap_samples_valid.jsonl"
    local eval_out="${NEED}/${exp}/strategy_eval_llm.json"

    log "GPU${gpu}: ${exp} 开始"

    if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$metrics" ]; then
        log "GPU${gpu}: ${exp} 训练跳过 (已存在)"
    else
        CUDA_VISIBLE_DEVICES="$gpu" python train.py \
            --config "$config" --dataset-tag "$DATASET_TAG" \
            --dataset-dir "$DATASET_DIR" >> "$logf" 2>&1 || { log "GPU${gpu}: ${exp} 训练失败!"; return 1; }
        log "GPU${gpu}: ${exp} 训练完成"
    fi

    if [ ! -f "$jsonl" ]; then
        log "GPU${gpu}: ${exp} swap_samples 不存在"; return 1
    fi
    if [ "$FORCE_EVAL" != "1" ] && [ -f "$eval_out" ]; then
        log "GPU${gpu}: ${exp} 评估跳过 (已存在)"
    else
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/evaluate_strategy_control_llm.py \
            --model-path "$MODEL_PATH" --config "$config" \
            --dataset-tag "$DATASET_TAG" --dataset-dir "$DATASET_DIR" \
            --jsonl "$jsonl" --out "$eval_out" \
            --max-samples "$EVAL_MAX_SAMPLES" >> "$logf" 2>&1 || { log "GPU${gpu}: ${exp} 评估失败!"; return 1; }
        log "GPU${gpu}: ${exp} 评估完成"
    fi
    log "GPU${gpu}: ${exp} 全部完成"
    return 0
}

echo "════════════════════════════════════════════════════════════════════════"
echo " 方法1 (contrastive) + B9 双卡自动调度"
echo " 开始: $TOTAL_START  数据: ${DATASET_DIR}  输出: ${DATASET_TAG}"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# ── 并行启动 ──
log "启动 GPU0: b6_contrastive        GPU1: b6_contrastive_no_orth"

# GPU 0: 先跑 contrastive，完成后标记
mkfifo /tmp/gpu0_done_$$ 2>/dev/null; rm -f /tmp/gpu0_done_$$
mkfifo /tmp/gpu1_done_$$ 2>/dev/null; rm -f /tmp/gpu1_done_$$

# 用文件标记
GPU0_FLAG="/tmp/gpu0_free_$$"
GPU1_FLAG="/tmp/gpu1_free_$$"
rm -f "$GPU0_FLAG" "$GPU1_FLAG"

(
    run_exp 0 b6_contrastive
    echo "done" > "$GPU0_FLAG"
) &
PID0=$!

(
    run_exp 1 b6_contrastive_no_orth
    echo "done" > "$GPU1_FLAG"
) &
PID1=$!

# ── 等待第一个完成的 GPU，在上面跑 B9 ──
B9_STARTED=0
while [ $B9_STARTED -eq 0 ]; do
    if [ -f "$GPU0_FLAG" ]; then
        log "GPU0 先完成 → 在 GPU0 跑 B9"
        run_exp 0 b9_prefix_then_lora &
        PID_B9=$!
        B9_STARTED=1
    elif [ -f "$GPU1_FLAG" ]; then
        log "GPU1 先完成 → 在 GPU1 跑 B9"
        run_exp 1 b9_prefix_then_lora &
        PID_B9=$!
        B9_STARTED=1
    else
        sleep 30
    fi
done

# ── 等待全部结束 ──
wait $PID0 || true
wait $PID1 || true
wait $PID_B9 || true

rm -f "$GPU0_FLAG" "$GPU1_FLAG"

TOTAL_END=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " 全部完成 — $TOTAL_START → $TOTAL_END"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

python3 -c "
import json
from pathlib import Path

need = Path('output/need') / '${DATASET_TAG}'
mapping = {
    'elicit-pref': 'information', 'express-pref': 'information', 'no-need': 'information',
    'small-talk': 'affective', 'self-need': 'substantive', 'other-need': 'substantive',
    'uv-part': 'substantive', 'vouch-fair': 'affective', 'promote-coordination': 'affective',
}

print('═' * 100)
print('  全部结果汇总')
print('═' * 100)
hdr = f\"{'Experiment':<30} {'ep':>4} {'valid_ppl':>10} {'test_ppl':>10} {'9-class':>10} {'4-class':>10}\"
print(hdr)
print('─' * 100)

for label, tag, exp_name in [
    # 新实验
    ('B6_contrastive (主方法)     ', '${DATASET_TAG}', 'b6_contrastive'),
    ('B6_contrastive_no_orth      ', '${DATASET_TAG}', 'b6_contrastive_no_orth'),
    ('B9  Prefix→LoRA (兜底)      ', '${DATASET_TAG}', 'b9_prefix_then_lora'),
    # 对比
    ('B3  Prefix-only             ', 'casino_augmented', 'b3_prefix_only'),
    ('B4  Prefix+LoRA             ', 'casino_augmented', 'b4_prefix_lora'),
    ('B6_fix (naive orth+cls)     ', 'casino_augmented_fix_b6', 'b6_fix'),
    ('B6_grad (grad routing)      ', 'casino_augmented_fix_b6', 'b6_grad'),
    ('B6_grad_no_orth (崩溃)      ', 'casino_augmented_fix_b6', 'b6_grad_no_orth'),
    ('B7  warm-start              ', 'casino_augmented', 'b7_dest_rs_warm'),
]:
    mf = Path('output/need') / tag / exp_name / 'metrics.json'
    ef = Path('output/need') / tag / exp_name / 'strategy_eval_llm.json'
    ep = vppl = tppl = acc9 = acc4 = '—'
    status = ''
    if mf.exists():
        m = json.loads(mf.read_text())
        best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
        ep = str(m.get('best_epoch', '—'))
        vppl = f\"{best.get('valid_ppl', 0):.2f}\"
        tppl = f\"{best.get('test_ppl', 0):.2f}\"
        if best.get('valid_ppl', 0) > 100:
            status = ' 💥崩溃'
    if ef.exists():
        e = json.loads(ef.read_text())
        acc9 = f\"{e['overall_accuracy']:.4f}\"
        correct = total = 0
        for row in e['details']:
            if row.get('pred_strategy') is None: continue
            if mapping.get(row.get('target_strategy','')) == mapping.get(row.get('pred_strategy','')):
                correct += 1
            total += 1
        acc4 = f'{correct/total:.4f}' if total > 0 else '—'
    print(f'{label:<30} {ep:>4} {vppl:>10} {tppl:>10} {acc9:>10} {acc4:>10}{status}')
print()
print('═' * 100)
"
