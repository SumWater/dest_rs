#!/usr/bin/env bash
# 去掉 set -e，改为每步手动检查，避免一步失败导致全部中断
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

FAIL_COUNT=0

run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "[开始] $step_name — $(date '+%H:%M:%S')"
    if "$@"; then
        echo "[完成] $step_name — $(date '+%H:%M:%S')"
        return 0
    else
        echo "[失败] $step_name — 退出码 $? — $(date '+%H:%M:%S')"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================================================"
echo " DEST-RS 今晚实验批处理脚本"
echo " 启动时间: $TIMESTAMP"
echo " 工作目录: $PROJECT_DIR"
echo "========================================================================"

# ──────────────────────────────────────────────────────────────────────
# 阶段 1：补齐核心消融实验 B2 / B3 / B6
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 1/4：核心消融实验"
echo "════════════════════════════════════════════════════════════════════"

# --- B2: LoRA-only 基线 ---
if [ -f outputs/b2_lora_only/metrics.json ]; then
    echo "[跳过] B2 (lora_only) 已有 metrics.json"
else
    run_step "B2 (lora_only)" python train.py --config configs/b2_lora_only.json
fi

# --- B3: Prefix-only 基线 ---
if [ -f outputs/b3_prefix_only/metrics.json ]; then
    echo "[跳过] B3 (prefix_only) 已有 metrics.json"
else
    run_step "B3 (prefix_only)" python train.py --config configs/b3_prefix_only.json
fi

# --- B6: 完整 DEST-RS ---
if [ -f outputs/b6_dest_rs/metrics.json ]; then
    echo "[跳过] B6 (dest_rs) 已有 metrics.json"
else
    run_step "B6 (dest_rs)" python train.py --config configs/b6_dest_rs.json
fi

echo ""
echo "阶段 1 结束 — $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────────────
# 阶段 2：B6 正交性分析
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 2/4：正交性分析"
echo "════════════════════════════════════════════════════════════════════"

if [ -f outputs/b6_seed42_orthogonality_test.json ]; then
    echo "[跳过] B6 正交性分析已存在"
elif [ ! -f outputs/b6_dest_rs/metrics.json ]; then
    echo "[跳过] B6 尚未训练完成，无法进行正交性分析"
else
    run_step "B6 正交性分析" python scripts/analyze_orthogonality.py \
        --checkpoint-dir outputs/b6_dest_rs \
        --split test \
        --max-samples 64 \
        --out outputs/b6_seed42_orthogonality_test.json
fi

# ──────────────────────────────────────────────────────────────────────
# 阶段 3：策略可控性评估
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 3/4：策略可控性评估 (Strategy Accuracy / Macro-F1)"
echo "════════════════════════════════════════════════════════════════════"

# 3a) 训练外部策略评估器
if [ -f outputs/strategy_evaluator.pt ]; then
    echo "[跳过] strategy_evaluator.pt 已存在"
else
    run_step "训练策略评估器" python scripts/train_strategy_evaluator.py \
        --config configs/b4_prefix_lora.json \
        --out outputs/strategy_evaluator.pt
fi

# 3b) 对各实验的 swap samples 计算策略可控性指标
EVAL_DIRS=(
    "outputs/b2_lora_only"
    "outputs/b3_prefix_only"
    "outputs/followup/b4_seed_42"
    "outputs/followup/b5_seed_42"
    "outputs/b6_dest_rs"
)

if [ ! -f outputs/strategy_evaluator.pt ]; then
    echo "[跳过] strategy_evaluator.pt 不存在，跳过全部策略评估"
else
    for dir in "${EVAL_DIRS[@]}"; do
        swap_file="$dir/swap_samples_valid.jsonl"
        out_file="$dir/strategy_control.csv"
        if [ -f "$out_file" ]; then
            echo "[跳过] $dir 策略评估已完成"
            continue
        fi
        if [ ! -f "$swap_file" ]; then
            echo "[跳过] $dir 缺少 swap_samples_valid.jsonl"
            continue
        fi
        run_step "策略评估 $dir" \
            python scripts/evaluate_strategy_control.py \
            --evaluator outputs/strategy_evaluator.pt \
            --jsonl "$swap_file" \
            --out "$out_file"
    done
fi

echo ""
echo "阶段 3 结束 — $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────────────
# 阶段 4：LLM Judge 标注（需要 DEEPSEEK_API_KEY）
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 4/4：LLM Judge 标注"
echo "════════════════════════════════════════════════════════════════════"

JUDGE_INPUT="outputs/judge_b4_vs_b5_seed42.csv"
JUDGE_OUTPUT="outputs/judge_b4_vs_b5_seed42_labeled.csv"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "[跳过] 未设置 DEEPSEEK_API_KEY 环境变量，跳过 LLM Judge"
    echo "       如需运行，请先执行: export DEEPSEEK_API_KEY=\"你的key\""
elif [ ! -f "$JUDGE_INPUT" ]; then
    echo "[跳过] judge 输入文件不存在: $JUDGE_INPUT"
else
    run_step "LLM Judge 标注" python scripts/llm_judge_sheet.py \
        --input "$JUDGE_INPUT" \
        --out "$JUDGE_OUTPUT" \
        --resume
fi

# ──────────────────────────────────────────────────────────────────────
# 汇总
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================================"
echo " 全部结束 — $(date '+%Y-%m-%d %H:%M:%S')"
echo " 失败步骤数: $FAIL_COUNT"
echo "========================================================================"
echo ""
echo "产出文件检查："
echo ""

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK]   $1"
    else
        echo "  [缺失] $1"
    fi
}

echo "核心消融 metrics："
check_file "outputs/b2_lora_only/metrics.json"
check_file "outputs/b3_prefix_only/metrics.json"
check_file "outputs/followup/b4_seed_42/metrics.json"
check_file "outputs/followup/b5_seed_42/metrics.json"
check_file "outputs/b6_dest_rs/metrics.json"

echo ""
echo "正交性分析："
check_file "outputs/b4_seed42_orthogonality_test.json"
check_file "outputs/b5_seed42_orthogonality_test.json"
check_file "outputs/b6_seed42_orthogonality_test.json"

echo ""
echo "策略可控性评估："
check_file "outputs/strategy_evaluator.pt"
for dir in "${EVAL_DIRS[@]}"; do
    check_file "$dir/strategy_control.csv"
done

echo ""
echo "LLM Judge："
check_file "$JUDGE_OUTPUT"

echo ""
if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "⚠ 有 $FAIL_COUNT 个步骤失败，请检查上方日志中的 [失败] 行。"
else
    echo "全部步骤成功完成。"
fi
echo "========================================================================"
