#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
FAIL_COUNT=0
TOTAL_START=$(date '+%Y-%m-%d %H:%M:%S')

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
    local out_dir="$3"
    if [ -f "$out_dir/metrics.json" ]; then
        echo "[跳过] $name — $out_dir/metrics.json 已存在"
        return 0
    fi
    run_step "$name — 训练" python train.py --config "$config"
}

llm_eval() {
    local name="$1"
    local config="$2"
    local out_dir="$3"
    if [ ! -f "$out_dir/swap_samples_valid.jsonl" ]; then
        echo "[跳过] $name LLM eval — swap_samples_valid.jsonl 不存在"
        return 0
    fi
    if [ -f "$out_dir/strategy_eval_llm.json" ]; then
        echo "[跳过] $name LLM eval — 已存在"
        return 0
    fi
    run_step "$name — LLM eval" python scripts/evaluate_strategy_control_llm.py \
        --model-path "$MODEL_PATH" \
        --config "$config" \
        --jsonl "$out_dir/swap_samples_valid.jsonl" \
        --out "$out_dir/strategy_eval_llm.json" \
        --max-samples 45
}

echo "════════════════════════════════════════════════════════════════════════"
echo " run_all.sh — 全部实验一键运行"
echo " 数据集: 重新划分后的 CaSiNo (仅有标注对话, 分层抽样)"
echo " 开始时间: $TOTAL_START"
echo "════════════════════════════════════════════════════════════════════════"

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: 无依赖的基线实验
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 1: 基线实验 (B2, B3, B4, B5, B6, B8-P1, B8dup-P1)         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

train_if_needed "B2 LoRA only"        configs/b2_lora_only.json        outputs/b2_lora_only
train_if_needed "B3 Prefix only"      configs/b3_prefix_only.json      outputs/b3_prefix_only
train_if_needed "B4 Prefix+LoRA"      configs/b4_prefix_lora.json      outputs/b4_prefix_lora
train_if_needed "B5 Prefix+LoRA+Orth" configs/b5_prefix_lora_orth.json outputs/b5_prefix_lora_orth
train_if_needed "B6 DeSTRS"           configs/b6_dest_rs.json          outputs/b6_dest_rs

train_if_needed "B8 Phase1 LoRA"      configs/b8/b8_p1_lora.json      outputs/b8/p1_lora_only
# 删除 Phase 1 的随机 prefix_bank.pt，防止 Phase 2 误加载
if [ -f "outputs/b8/p1_lora_only/prefix_bank.pt" ]; then
    rm "outputs/b8/p1_lora_only/prefix_bank.pt"
    echo "  已删除 B8 Phase1 随机 prefix_bank.pt"
fi

train_if_needed "B8dup Phase1 LoRA"   configs/b8_dup/b8_dup_p1_lora.json outputs/b8_dup/p1_lora_only
if [ -f "outputs/b8_dup/p1_lora_only/prefix_bank.pt" ]; then
    rm "outputs/b8_dup/p1_lora_only/prefix_bank.pt"
    echo "  已删除 B8dup Phase1 随机 prefix_bank.pt"
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: 有依赖的实验
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 2: 依赖实验 (B7, B7w ← B3; B8-P2 ← B8-P1; B8dup-P2)     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

if [ -f "outputs/b3_prefix_only/prefix_bank.pt" ]; then
    train_if_needed "B7 warm-start"       configs/b7/b7_dest_rs_warm.json    outputs/b7/b7_dest_rs_warm
    train_if_needed "B7w frozen prefix"   configs/b7/b7w_frozen_prefix.json  outputs/b7/b7w_frozen_prefix
else
    echo "[跳过] B7, B7w — B3 的 prefix_bank.pt 不存在 (B3 训练可能失败)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ -f "outputs/b8/p1_lora_only/metrics.json" ]; then
    train_if_needed "B8 Phase2 Prefix"    configs/b8/b8_p2_prefix.json       outputs/b8/p2_prefix_frozen_lora
else
    echo "[跳过] B8 Phase2 — Phase1 未完成"
fi

if [ -f "outputs/b8_dup/p1_lora_only/metrics.json" ]; then
    train_if_needed "B8dup Phase2 Prefix" configs/b8_dup/b8_dup_p2_prefix.json outputs/b8_dup/p2_prefix_frozen_lora
else
    echo "[跳过] B8dup Phase2 — Phase1 未完成"
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 3: LLM 策略控制评估
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  Phase 3: LLM 策略控制评估                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"

llm_eval "B2"     configs/b2_lora_only.json          outputs/b2_lora_only
llm_eval "B3"     configs/b3_prefix_only.json        outputs/b3_prefix_only
llm_eval "B4"     configs/b4_prefix_lora.json        outputs/b4_prefix_lora
llm_eval "B5"     configs/b5_prefix_lora_orth.json   outputs/b5_prefix_lora_orth
llm_eval "B6"     configs/b6_dest_rs.json            outputs/b6_dest_rs
llm_eval "B7"     configs/b7/b7_dest_rs_warm.json    outputs/b7/b7_dest_rs_warm
llm_eval "B7w"    configs/b7/b7w_frozen_prefix.json  outputs/b7/b7w_frozen_prefix
llm_eval "B8"     configs/b8/b8_p2_prefix.json       outputs/b8/p2_prefix_frozen_lora
llm_eval "B8dup"  configs/b8_dup/b8_dup_p2_prefix.json outputs/b8_dup/p2_prefix_frozen_lora

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

experiments = [
    ('B2  LoRA only',        'outputs/b2_lora_only'),
    ('B3  Prefix only',      'outputs/b3_prefix_only'),
    ('B4  Prefix+LoRA',      'outputs/b4_prefix_lora'),
    ('B5  +Orth',            'outputs/b5_prefix_lora_orth'),
    ('B6  DeSTRS',           'outputs/b6_dest_rs'),
    ('B7  warm-start',       'outputs/b7/b7_dest_rs_warm'),
    ('B7w frozen prefix',    'outputs/b7/b7w_frozen_prefix'),
    ('B8  2-phase(drop)',    'outputs/b8/p2_prefix_frozen_lora'),
    ('B8d 2-phase(dup)',     'outputs/b8_dup/p2_prefix_frozen_lora'),
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

for name, out_dir in experiments:
    out = Path(out_dir)
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
print('数据集: 重新划分后的 CaSiNo (396 有标注对话, 分层 80/10/10)')
"
