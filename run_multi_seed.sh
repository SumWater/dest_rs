#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_multi_seed.sh — 对 B3/B4/B7/B9 运行多 seed 实验
#
# 用法:
#   # 默认 3 个 seed (42, 43, 44)
#   DATASET_DIR=./augmented_data ./run_multi_seed.sh
#
#   # 自定义 seed 列表
#   SEEDS="42 43 44 45" DATASET_DIR=./augmented_data ./run_multi_seed.sh
#
#   # 强制重跑
#   FORCE_RETRAIN=1 FORCE_EVAL=1 DATASET_DIR=./augmented_data ./run_multi_seed.sh
#
# 每个 seed 的输出隔离在独立的 dataset_tag 下：
#   output/need/{dataset_tag}_seed{N}/
#   output/other/{dataset_tag}_seed{N}/
# ══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
DEFAULT_DATASET_DIR="./CaSiNo-main/data"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
BASE_TAG="${BASE_TAG:-casino_augmented}"
SEEDS="${SEEDS:-42 43 44}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-270}"

# ── 只跑关键实验 ──
EXPERIMENTS=(
    "B3|configs/b3_prefix_only.json|b3_prefix_only|prefix_bank.pt|"
    "B4|configs/b4_prefix_lora.json|b4_prefix_lora|prefix_bank.pt|"
)

DEPENDENT_EXPERIMENTS=(
    "B7|configs/b7_dest_rs_warm.json|b7_dest_rs_warm|prefix_bank.pt|prefix_bank.pt"
    "B9|configs/b9_prefix_then_lora.json|b9_prefix_then_lora|lora_adapter/adapter_config.json|prefix_bank.pt"
)

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
        return 1
    fi
}

llm_eval_exp() {
    local tag="$1" exp_name="$2" config="$3"
    local swap="${PROJECT_DIR}/output/need/${tag}/${exp_name}/swap_samples_valid.jsonl"
    local out="${PROJECT_DIR}/output/need/${tag}/${exp_name}/strategy_eval_llm.json"

    if [ ! -f "$swap" ]; then
        echo "[跳过] LLM eval ${exp_name} — swap_samples 不存在"
        return 0
    fi
    if [ "$FORCE_EVAL" != "1" ] && [ -f "$out" ]; then
        echo "[跳过] LLM eval ${exp_name} — 已存在"
        return 0
    fi
    run_step "${exp_name} LLM eval" python scripts/evaluate_strategy_control_llm.py \
        --model-path "$MODEL_PATH" \
        --config "$config" \
        --dataset-tag "$tag" \
        --dataset-dir "$DATASET_DIR" \
        --jsonl "$swap" \
        --out "$out" \
        --max-samples "$EVAL_MAX_SAMPLES"
}

echo "════════════════════════════════════════════════════════════════════════"
echo " run_multi_seed.sh — 多 seed 关键实验"
echo " 基础 tag: ${BASE_TAG}"
echo " Seeds: ${SEEDS}"
echo " 数据集: ${DATASET_DIR}"
echo " 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════════════"

for seed in $SEEDS; do
    SEED_TAG="${BASE_TAG}_seed${seed}"
    SEED_OTHER="${PROJECT_DIR}/output/other/${SEED_TAG}"
    B3_PREFIX="${SEED_OTHER}/b3_prefix_only/prefix_bank.pt"

    echo ""
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║  Seed ${seed} — tag: ${SEED_TAG}                                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"

    # ── Phase 1: 独立实验 (B3, B4) ──
    for exp_info in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r name config exp_name artifact _ <<< "$exp_info"
        metrics="${PROJECT_DIR}/output/need/${SEED_TAG}/${exp_name}/metrics.json"

        if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$metrics" ]; then
            echo "[跳过] ${name} seed=${seed} — metrics 已存在"
            continue
        fi
        run_step "${name} seed=${seed} 训练" python train.py \
            --config "$config" \
            --dataset-tag "$SEED_TAG" \
            --dataset-dir "$DATASET_DIR" \
            --seed "$seed"
    done

    # ── Phase 2: 依赖实验 (B7, B9 ← B3) ──
    if [ ! -f "$B3_PREFIX" ]; then
        echo "[跳过] B7/B9 seed=${seed} — B3 prefix 不存在: $B3_PREFIX"
        continue
    fi

    B3_OTHER_DIR="${SEED_OTHER}/b3_prefix_only"
    for exp_info in "${DEPENDENT_EXPERIMENTS[@]}"; do
        IFS='|' read -r name config exp_name artifact depends_on <<< "$exp_info"
        metrics="${PROJECT_DIR}/output/need/${SEED_TAG}/${exp_name}/metrics.json"

        if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$metrics" ]; then
            echo "[跳过] ${name} seed=${seed} — metrics 已存在"
            continue
        fi
        run_step "${name} seed=${seed} 训练" python train.py \
            --config "$config" \
            --dataset-tag "$SEED_TAG" \
            --dataset-dir "$DATASET_DIR" \
            --seed "$seed" \
            --warm-start-dir "$B3_OTHER_DIR"
    done

    # ── Phase 3: LLM 评估 ──
    echo ""
    echo "── Seed ${seed} LLM 评估 ──"
    for exp_info in "${EXPERIMENTS[@]}" "${DEPENDENT_EXPERIMENTS[@]}"; do
        IFS='|' read -r name config exp_name _ __ <<< "$exp_info"
        llm_eval_exp "$SEED_TAG" "$exp_name" "$config"
    done
done

# ── 汇总 ──
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo " 多 seed 实验完成 — $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo " 汇总结果:"
echo "────────────────────────────────────────────────────────────────────────"

for seed in $SEEDS; do
    SEED_TAG="${BASE_TAG}_seed${seed}"
    echo " Seed ${seed}:"
    for exp_name in b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
        ef="${PROJECT_DIR}/output/need/${SEED_TAG}/${exp_name}/strategy_eval_llm.json"
        if [ -f "$ef" ]; then
            acc=$(python3 -c "import json; print(f\"{json.load(open('$ef'))['overall_accuracy']*100:.2f}%\")")
            echo "    ${exp_name}: ${acc}"
        else
            echo "    ${exp_name}: —"
        fi
    done
done

echo "════════════════════════════════════════════════════════════════════════"
