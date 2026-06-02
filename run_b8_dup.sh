#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
P1_OUT="outputs/b8_dup/p1_lora_only"
P2_OUT="outputs/b8_dup/p2_prefix_frozen_lora"

echo "================================================================================"
echo " B8-dup: Phase 1 (LoRA) → Phase 2 (prefix on frozen LoRA)"
echo "          multi_label_policy = duplicate"
echo "================================================================================"
echo " 假设: 先让 LoRA 学好生成，锁死，再加 prefix 学策略"
echo "       prefix 工作在固定 attention 上，不会被 LoRA 干扰"
echo "  变化: duplicate 替换 drop，回收多标签数据，低频策略增幅 165-322%"
echo "================================================================================"
echo ""

START_TIME=$(date '+%H:%M:%S')

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Train LoRA only
# ═══════════════════════════════════════════════════════════════════════
echo "[Phase 1/2] 训练 LoRA only..."
if [ -f "$P1_OUT/metrics.json" ]; then
    echo "  [跳过] $P1_OUT 已存在"
else
    python train.py --config configs/b8_dup/b8_dup_p1_lora.json

    # Phase 1 saves a random (untrained) prefix_bank.pt. Delete it so
    # Phase 2 doesn't accidentally load random prefix values.
    if [ -f "$P1_OUT/prefix_bank.pt" ]; then
        rm "$P1_OUT/prefix_bank.pt"
        echo "  已删除 Phase 1 的随机 prefix_bank.pt（避免 Phase 2 误加载）"
    fi

    echo "  Phase 1 PPL:"
    python3 -c "
import json
m = json.load(open('$P1_OUT/metrics.json'))
best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
print(f'    valid_ppl={best.get(\"valid_ppl\",0):.2f}  test_ppl={best.get(\"test_ppl\",0):.2f}')
"
fi

# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Prefix only on frozen LoRA
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "[Phase 2/2] 训练 prefix（LoRA 冻结）..."
if [ -f "$P2_OUT/metrics.json" ]; then
    echo "  [跳过] $P2_OUT 已存在"
else
    python train.py --config configs/b8_dup/b8_dup_p2_prefix.json
fi

# ═══════════════════════════════════════════════════════════════════════
# LLM Evaluation
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "[Eval] LLM 策略控制评估..."
python scripts/evaluate_strategy_control_llm.py \
    --model-path "$MODEL_PATH" \
    --config configs/b8_dup/b8_dup_p2_prefix.json \
    --jsonl "$P2_OUT/swap_samples_valid.jsonl" \
    --out "$P2_OUT/strategy_eval_llm.json" \
    --max-samples 45

# ═══════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════
END_TIME=$(date '+%H:%M:%S')
echo ""
echo "================================================================================"
echo " B8-dup 完成 ($START_TIME → $END_TIME)"
echo ""

python3 -c "
import json
from pathlib import Path

p1 = Path('$P1_OUT') / 'metrics.json'
p2 = Path('$P2_OUT') / 'metrics.json'
ef = Path('$P2_OUT') / 'strategy_eval_llm.json'

if p2.exists():
    m = json.loads(p2.read_text())
    last = m['history'][-1]
    best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
    print('Phase 2 (prefix on frozen LoRA):')
    print(f'  train_loss:      {last[\"train_loss\"]:.4f}')
    print(f'  valid_ppl:       {best.get(\"valid_ppl\", 0):.2f}')
    print(f'  test_ppl:        {best.get(\"test_ppl\", 0):.2f}')

if ef.exists():
    e = json.loads(ef.read_text())
    print(f'  llm_9class_acc:  {e[\"overall_accuracy\"]:.4f}')

    mapping = {
        'elicit-pref': 'elicit-pref',
        'self-need': 'need-expression', 'other-need': 'need-expression',
        'no-need': 'need-expression', 'uv-part': 'need-expression',
        'promote-coordination': 'collaboration', 'vouch-fair': 'collaboration',
        'small-talk': 'rapport', 'showing-empathy': 'rapport',
    }
    correct = total = 0
    for row in e['details']:
        if row['pred_strategy'] is None: continue
        if mapping.get(row['target_strategy']) == mapping.get(row['pred_strategy']):
            correct += 1
        total += 1
    print(f'  llm_4class_acc:  {correct/total:.4f}')

print()
print('── 对比基线 ──')
print('  B3  (prefix only):      9-class=0.289  4-class=0.622')
print('  B7  (warm+trainable):   9-class=0.156  4-class=0.444')
print('  B7w (warm+frozen pref): 9-class=0.111  4-class=0.356')
print('  B8  (lora first+pref):  9-class=0.279  4-class=0.488')
print('  B8-dup (duplicate):     见上方')
print('================================================================================")
