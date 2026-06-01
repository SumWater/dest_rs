#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
CONFIG="configs/b7/b7w_frozen_prefix.json"
OUT_DIR="outputs/b7/b7w_frozen_prefix"

echo "================================================================================"
echo " B7w: 方向 A — freeze prefix + LoRA + weak orth + cls"
echo "================================================================================"
echo " 设计:"
echo "   warm_start: B3 prefix_bank.pt (已证明有策略信号)"
echo "   freeze_prefix: true → prefix 锁死"
echo "   LoRA + cls 可训练"
echo "   λ_orth=0.05 → 微弱正交约束，防止 LoRA 扰动 prefix"
echo "   λ_cls=2.0 → cls 从 delta_prefix 监督策略"
echo "================================================================================"
echo ""

START_TIME=$(date '+%H:%M:%S')
echo "开始时间: $START_TIME"

# ── Train ──
echo "[1/3] 训练..."
python train.py --config "$CONFIG"

# ── LLM Strategy Eval ──
echo ""
echo "[2/3] LLM 策略控制评估..."
python scripts/evaluate_strategy_control_llm.py \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --jsonl "$OUT_DIR/swap_samples_valid.jsonl" \
    --out "$OUT_DIR/strategy_eval_llm.json" \
    --max-samples 45

# ── Generation ──
echo ""
echo "[3/3] 生成多样性..."
python scripts/evaluate_generations.py --jsonl "$OUT_DIR/swap_samples_valid.jsonl"

END_TIME=$(date '+%H:%M:%S')
echo ""
echo "================================================================================"
echo " B7w 完成 ($START_TIME → $END_TIME)"
echo ""
echo "── 快速结果 ──"
python3 -c "
import json
from pathlib import Path

out = Path('$OUT_DIR')

# Training
mf = out / 'metrics.json'
if mf.exists():
    m = json.loads(mf.read_text())
    last = m['history'][-1]
    best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
    print(f'train_loss:      {last[\"train_loss\"]:.4f}')
    print(f'valid_loss:      {last.get(\"valid_loss\", 0):.4f}')
    print(f'valid_ppl:       {best.get(\"valid_ppl\", 0):.2f}')
    print(f'test_ppl:        {best.get(\"test_ppl\", 0):.2f}')
    print(f'train_orth_loss: {last.get(\"train_orth_loss\", 0):.6f}')
    print(f'train_cls_loss:  {last.get(\"train_cls_loss\", 0):.6f}')

# LLM eval
ef = out / 'strategy_eval_llm.json'
if ef.exists():
    e = json.loads(ef.read_text())
    print(f'llm_9class_acc:  {e[\"overall_accuracy\"]:.4f}')

print()
print('── 4-class 重算 ──')
mapping = {
    'elicit-pref': 'elicit-pref',
    'self-need': 'need-expression', 'other-need': 'need-expression',
    'no-need': 'need-expression', 'uv-part': 'need-expression',
    'promote-coordination': 'collaboration', 'vouch-fair': 'collaboration',
    'small-talk': 'rapport', 'showing-empathy': 'rapport',
}
if ef.exists():
    e = json.loads(ef.read_text())
    correct = total = 0
    for row in e['details']:
        if row['pred_strategy'] is None: continue
        if mapping.get(row['target_strategy']) == mapping.get(row['pred_strategy']):
            correct += 1
        total += 1
    print(f'llm_4class_acc:  {correct/total:.4f}')

print()
print('── 对比基线 ──')
print('  B3  (prefix only):     9-class=0.289  4-class=0.622')
print('  V3 B6 (dest_rs):       9-class=0.178  4-class=0.333')
print('  B7w (frozen prefix):   见上方')
print('================================================================================")
