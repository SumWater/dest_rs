#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
CONFIG="configs/b7/b7_dest_rs_warm.json"
EXP="b7_dest_rs_warm"
OUT_DIR="outputs/b7/$EXP"

echo "================================================================================"
echo " B7: prefix warm start + λ_orth=0.01 + cls"
echo "================================================================================"
echo " 核心假设:"
echo "   1. B3 warm-start prefix → 已具备策略信号"
echo "   2. λ_orth=0.01 → 不强行正交，只做微量正则"
echo "   3. cls 从 delta_prefix → 允许重叠策略表示"
echo "   4. multi_label_policy=drop → 干净信号"
echo "================================================================================"
echo ""

# ── Train ──
echo "[1/4] 训练..."
python train.py --config "$CONFIG"

# ── Orthogonality ──
echo ""
echo "[2/4] 正交性分析..."
python scripts/analyze_orthogonality.py \
    --checkpoint-dir "$OUT_DIR" \
    --split test \
    --max-samples 64 \
    --out "$OUT_DIR/orthogonality.json"

# ── Generation ──
echo ""
echo "[3/4] 生成多样性..."
python scripts/evaluate_generations.py --jsonl "$OUT_DIR/swap_samples_valid.jsonl"

# ── LLM Strategy Eval ──
echo ""
echo "[4/4] LLM 策略控制评估..."
python scripts/evaluate_strategy_control_llm.py \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --jsonl "$OUT_DIR/swap_samples_valid.jsonl" \
    --out "$OUT_DIR/strategy_eval_llm.json" \
    --max-samples 45

echo ""
echo "================================================================================"
echo " B7 完成"
echo ""
echo " 对比基线:"
echo "   B3 (prefix only):    9-class=28.9%  4-class=62.2%"
echo "   V3 B6 (dest_rs):     9-class=17.8%  4-class=33.3%"
echo "   B7 (warm+weak orth): TBD"
echo "================================================================================"

# ── Quick summary ──
echo ""
echo "── B7 快速结果 ──"
python3 -c "
import json, sys
from pathlib import Path

out = Path('$OUT_DIR')

# Training metrics
mf = out / 'metrics.json'
if mf.exists():
    m = json.loads(mf.read_text())
    last = m['history'][-1]
    best = min(m['history'], key=lambda h: h.get('valid_loss', float('inf')))
    print(f'best_valid_loss: {m[\"best_valid_loss\"]:.4f}')
    print(f'best_valid_ppl:  {best.get(\"valid_ppl\", 0):.2f}')
    print(f'last_orth_loss:  {last.get(\"train_orth_loss\", 0):.6f}')
    print(f'last_cls_loss:   {last.get(\"train_cls_loss\", 0):.6f}')

# Orthogonality
of = out / 'orthogonality.json'
if of.exists():
    o = json.loads(of.read_text())
    print(f'mean_cosine:     {o[\"mean_cosine\"]:.4f}')
    print(f'mean_abs_cosine: {o[\"mean_abs_cosine\"]:.4f}')

# LLM eval
ef = out / 'strategy_eval_llm.json'
if ef.exists():
    e = json.loads(ef.read_text())
    print(f'llm_9class_acc:  {e[\"overall_accuracy\"]:.4f}')
    print(f'null_predictions: {e[\"null_predictions\"]}')

print()
print('对比:')
print('  B3 (prefix only):    9-class=0.289  4-class=0.622')
print('  V3 B6 (dest_rs):     9-class=0.178  4-class=0.333')
"
