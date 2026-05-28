#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

FAIL_COUNT=0
LOG_FILE="outputs/v2/tonight_v2_log.txt"
mkdir -p outputs/v2

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

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK]   $1"
    else
        echo "  [缺失] $1"
    fi
}

{

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "========================================================================"
echo " DEST-RS V2 实验批处理脚本"
echo " 启动时间: $TIMESTAMP"
echo " 工作目录: $PROJECT_DIR"
echo ""
echo " 改动摘要:"
echo "   - losses.py: cls 输入从 h_both 改为 delta_prefix"
echo "   - lambda_orth: 0.05 -> 1.0"
echo "   - lambda_cls:  0.2  -> 2.0 (仅 B6v2)"
echo "   - orth_every_n_steps: 20 -> 1"
echo "   - orth_start_step: 50 -> 0"
echo "   - num_epochs: 2 -> 3"
echo "========================================================================"

# ──────────────────────────────────────────────────────────────────────
# 阶段 1：训练
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 1/3：训练 B5v2 + B6v2"
echo "════════════════════════════════════════════════════════════════════"

if [ -f outputs/v2/b5v2_orth_strong/metrics.json ]; then
    echo "[跳过] B5v2 已有 metrics.json"
else
    run_step "B5v2 (prefix_lora_orth, lambda_orth=1.0, every_n=1, 3ep)" \
        python train.py --config configs/v2/b5v2_orth_strong.json
fi

if [ -f outputs/v2/b6v2_dest_rs_fixed/metrics.json ]; then
    echo "[跳过] B6v2 已有 metrics.json"
else
    run_step "B6v2 (dest_rs, lambda_orth=1.0, lambda_cls=2.0, cls=delta_prefix, 3ep)" \
        python train.py --config configs/v2/b6v2_dest_rs_fixed.json
fi

echo ""
echo "阶段 1 结束 — $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────────────
# 阶段 2：正交性分析
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 2/3：正交性分析"
echo "════════════════════════════════════════════════════════════════════"

if [ -f outputs/v2/b5v2_orthogonality_test.json ]; then
    echo "[跳过] B5v2 正交性分析已存在"
elif [ ! -f outputs/v2/b5v2_orth_strong/metrics.json ]; then
    echo "[跳过] B5v2 尚未训练完成"
else
    run_step "B5v2 正交性分析" python scripts/analyze_orthogonality.py \
        --checkpoint-dir outputs/v2/b5v2_orth_strong \
        --split test \
        --max-samples 64 \
        --out outputs/v2/b5v2_orthogonality_test.json
fi

if [ -f outputs/v2/b6v2_orthogonality_test.json ]; then
    echo "[跳过] B6v2 正交性分析已存在"
elif [ ! -f outputs/v2/b6v2_dest_rs_fixed/metrics.json ]; then
    echo "[跳过] B6v2 尚未训练完成"
else
    run_step "B6v2 正交性分析" python scripts/analyze_orthogonality.py \
        --checkpoint-dir outputs/v2/b6v2_dest_rs_fixed \
        --split test \
        --max-samples 64 \
        --out outputs/v2/b6v2_orthogonality_test.json
fi

echo ""
echo "阶段 2 结束 — $(date '+%H:%M:%S')"

# ──────────────────────────────────────────────────────────────────────
# 阶段 3：生成质量评估
# ──────────────────────────────────────────────────────────────────────

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo " 阶段 3/3：生成质量评估 (distinct / repetition)"
echo "════════════════════════════════════════════════════════════════════"

for dir in outputs/v2/b5v2_orth_strong outputs/v2/b6v2_dest_rs_fixed; do
    swap_file="$dir/swap_samples_valid.jsonl"
    if [ ! -f "$swap_file" ]; then
        echo "[跳过] $dir 缺少 swap_samples_valid.jsonl"
        continue
    fi
    echo ""
    echo "--- $dir ---"
    python scripts/evaluate_generations.py --jsonl "$swap_file"
done

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

echo "训练产出："
check_file "outputs/v2/b5v2_orth_strong/metrics.json"
check_file "outputs/v2/b5v2_orth_strong/swap_samples_valid.jsonl"
check_file "outputs/v2/b6v2_dest_rs_fixed/metrics.json"
check_file "outputs/v2/b6v2_dest_rs_fixed/swap_samples_valid.jsonl"

echo ""
echo "正交性分析："
check_file "outputs/v2/b5v2_orthogonality_test.json"
check_file "outputs/v2/b6v2_orthogonality_test.json"

echo ""
echo "========================================================================"
echo ""
echo "与旧实验的关键对比项（手动检查）："
echo "  旧 B5: outputs/b5_seed42_orthogonality_test.json  mean_cosine=0.086"
echo "  旧 B6: outputs/b6_seed42_orthogonality_test.json  mean_cosine=0.133"
echo "  旧 B4: outputs/b4_seed42_orthogonality_test.json  mean_cosine=0.122"
echo ""
echo "  期望 B5v2 mean_cosine 大幅低于 0.086"
echo "  期望 B6v2 mean_cosine < B5v2（cls 不再破坏正交性）"
echo "  期望 test PPL 接近 8.0（生成质量不恶化）"
echo "========================================================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "⚠ 有 $FAIL_COUNT 个步骤失败，请检查上方日志中的 [失败] 行。"
else
    echo "全部步骤成功完成。"
fi

} 2>&1 | tee "$LOG_FILE"
