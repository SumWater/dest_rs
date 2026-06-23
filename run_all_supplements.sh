#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# run_all_supplements.sh — 一键运行全部补充实验
#
# 包含 5.1-5.5 所有补充实验，自动利用多 GPU 并行调度。
#
# 用法:
#   DATASET_DIR=./augmented_data ./run_all_supplements.sh
#
#   DUAL_GPU=1  DATASET_DIR=./augmented_data ./run_all_supplements.sh  # 双卡
#   FORCE_RETRAIN=1 FORCE_EVAL=1 ...                                   # 强制重跑
#   SEEDS="42 43 44 45" ...                                            # 自定义 seed
# ══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── 配置 ──
MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
DATASET_DIR="${DATASET_DIR:-./CaSiNo-main/data}"
BASE_TAG="${BASE_TAG:-casino_augmented}"
SEEDS="${SEEDS:-42 43 44}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-900}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10000}"
HUMAN_EVAL_CONTEXTS="${HUMAN_EVAL_CONTEXTS:-5}"
REPR_MAX_SAMPLES="${REPR_MAX_SAMPLES:-60}"

# ── 并行配置 ──
# DUAL_GPU=0 → 单 GPU，1 job at a time
# DUAL_GPU=1 → 双 GPU，每个 GPU 2 jobs = 4 并行
DUAL_GPU="${DUAL_GPU:-0}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SLOTS_PER_GPU=2                         # 每 GPU 并行槽位数

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
LOG_DIR="${PROJECT_DIR}/.supplement_logs"
mkdir -p "$LOG_DIR"
FAIL_COUNT=0

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

green()  { echo -e "\033[32m$*\033[0m"; }
yellow() { echo -e "\033[33m$*\033[0m"; }
red()    { echo -e "\033[31m$*\033[0m"; }
bold()   { echo -e "\033[1m$*\033[0m"; }

# ── Per-GPU PID 追踪 ──
declare -a GPU0_PIDS=()
declare -a GPU1_PIDS=()

wait_for_slot() {
    # 等目标 GPU 上空出一个槽位，清理已完成的 PID
    local gpu="$1"
    local max_slots="$SLOTS_PER_GPU"
    local -n pids_ref
    if [ "$gpu" = "$GPU0" ]; then
        pids_ref=GPU0_PIDS
    else
        pids_ref=GPU1_PIDS
    fi

    # 清理已退出的 PID
    local new_array=()
    for pid in "${pids_ref[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            new_array+=("$pid")
        fi
    done
    pids_ref=("${new_array[@]}")

    # 等直到有空位
    while [ "${#pids_ref[@]}" -ge "$max_slots" ]; do
        # 等任意一个子进程退出
        wait -n -p done_pid "${pids_ref[@]}" 2>/dev/null || true
        # 重新清理
        new_array=()
        for pid in "${pids_ref[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_array+=("$pid")
            fi
        done
        pids_ref=("${new_array[@]}")
    done
}

launch_job() {
    # launch_job <gpu> <log_name> <command...>
    # 等待 GPU 上有空槽位后启动，返回 PID
    local gpu="$1"; shift
    local log_name="$1"; shift
    local log_file="${LOG_DIR}/${log_name}.log"

    wait_for_slot "$gpu"

    echo "  [GPU ${gpu}] $(yellow "→ ${log_name}")"
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" > "$log_file" 2>&1 &
    local pid=$!

    if [ "$gpu" = "$GPU0" ]; then
        GPU0_PIDS+=("$pid")
    else
        GPU1_PIDS+=("$pid")
    fi
    echo "$pid"
}

wait_all_jobs() {
    # 等所有 GPU 上的任务完成
    local all_pids=("${GPU0_PIDS[@]}" "${GPU1_PIDS[@]}")
    local failed=0
    for pid in "${all_pids[@]}"; do
        if ! wait "$pid" 2>/dev/null; then
            failed=$((failed + 1))
        fi
    done
    GPU0_PIDS=()
    GPU1_PIDS=()
    return $failed
}

train_job() {
    # train_job <gpu> <tag> <config> <exp_name> [extra args...]
    local gpu="$1"; shift
    local tag="$1"; shift
    local config="$1"; shift
    local exp_name="$1"; shift
    local extra_args=("$@")
    local metrics="${PROJECT_DIR}/output/need/${tag}/${exp_name}/metrics.json"

    if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$metrics" ]; then
        echo "  $(yellow "跳过 ${tag}/${exp_name} — 已存在")"
        return 0
    fi

    launch_job "$gpu" "train_${tag}_${exp_name}" \
        python train.py \
            --config "$config" \
            --dataset-tag "$tag" \
            --dataset-dir "$DATASET_DIR" \
            "${extra_args[@]}"
}

llm_eval_job() {
    # llm_eval_job <gpu> <tag> <config> <exp_name>
    local gpu="$1"; shift
    local tag="$1"; shift
    local config="$1"; shift
    local exp_name="$1"; shift
    local swap="${PROJECT_DIR}/output/need/${tag}/${exp_name}/swap_samples_valid.jsonl"
    local out="${PROJECT_DIR}/output/need/${tag}/${exp_name}/strategy_eval_llm.json"

    if [ ! -f "$swap" ]; then
        echo "  $(yellow "跳过 eval ${tag}/${exp_name} — swap_samples 不存在")"
        return 0
    fi
    if [ "$FORCE_EVAL" != "1" ] && [ -f "$out" ]; then
        echo "  $(yellow "跳过 eval ${tag}/${exp_name} — 已存在")"
        return 0
    fi

    launch_job "$gpu" "eval_${tag}_${exp_name}" \
        python scripts/evaluate_strategy_control_llm.py \
            --model-path "$MODEL_PATH" \
            --config "$config" \
            --dataset-tag "$tag" \
            --dataset-dir "$DATASET_DIR" \
            --jsonl "$swap" \
            --out "$out" \
            --max-samples "$EVAL_MAX_SAMPLES"
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

echo ""
echo "$(bold '═══════════════════════════════════════════════════════════')"
echo "$(bold '  run_all_supplements.sh')"
echo "$(bold '═══════════════════════════════════════════════════════════')"
echo ""
if [ "$DUAL_GPU" = "1" ]; then
    echo "  模式: 双卡 (GPU${GPU0}/${GPU1} × ${SLOTS_PER_GPU} slots = $((2*SLOTS_PER_GPU)) 并行)"
else
    echo "  模式: 单卡 (GPU${GPU0}, 串行)"
fi
echo "  tag: ${BASE_TAG}  seeds: ${SEEDS}"
echo "  数据集: ${DATASET_DIR}"
echo "  日志:   ${LOG_DIR}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: 主实验 B2/B3/B4/B7/B9
# ═══════════════════════════════════════════════════════════════════════════
echo "$(bold '══════════════════════════════════════════')"
echo "$(bold '  Phase 1: B2/B3/B4 训练 (独立，可并行)')"
echo "$(bold '══════════════════════════════════════════')"
echo ""

if [ "$DUAL_GPU" = "1" ]; then
    # B2, B3, B4 三者独立，丢到 4 个槽位里自动调度
    train_job "$GPU0" "$BASE_TAG" configs/b2_lora_only.json   b2_lora_only
    train_job "$GPU0" "$BASE_TAG" configs/b3_prefix_only.json b3_prefix_only
    train_job "$GPU1" "$BASE_TAG" configs/b4_prefix_lora.json b4_prefix_lora
    wait_all_jobs || FAIL_COUNT=$((FAIL_COUNT + 1))

    # B7/B9 依赖 B3
    B3_PREFIX="${PROJECT_DIR}/output/other/${BASE_TAG}/b3_prefix_only/prefix_bank.pt"
    if [ -f "$B3_PREFIX" ]; then
        B3_OTHER="${PROJECT_DIR}/output/other/${BASE_TAG}/b3_prefix_only"
        echo ""
        echo "$(bold '  B7/B9 训练 (依赖 B3)')"
        train_job "$GPU0" "$BASE_TAG" configs/b7_dest_rs_warm.json    b7_dest_rs_warm    --warm-start-dir "$B3_OTHER"
        train_job "$GPU1" "$BASE_TAG" configs/b9_prefix_then_lora.json b9_prefix_then_lora --warm-start-dir "$B3_OTHER"
        wait_all_jobs || FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo "  $(red 'B3 prefix 不存在，跳过 B7/B9')"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
else
    train_job "$GPU0" "$BASE_TAG" configs/b2_lora_only.json   b2_lora_only
    train_job "$GPU0" "$BASE_TAG" configs/b3_prefix_only.json b3_prefix_only
    train_job "$GPU0" "$BASE_TAG" configs/b4_prefix_lora.json b4_prefix_lora
    wait_all_jobs

    B3_PREFIX="${PROJECT_DIR}/output/other/${BASE_TAG}/b3_prefix_only/prefix_bank.pt"
    if [ -f "$B3_PREFIX" ]; then
        B3_OTHER="${PROJECT_DIR}/output/other/${BASE_TAG}/b3_prefix_only"
        train_job "$GPU0" "$BASE_TAG" configs/b7_dest_rs_warm.json    b7_dest_rs_warm    --warm-start-dir "$B3_OTHER"
        train_job "$GPU0" "$BASE_TAG" configs/b9_prefix_then_lora.json b9_prefix_then_lora --warm-start-dir "$B3_OTHER"
        wait_all_jobs
    else
        echo "  $(red 'B3 prefix 不存在，跳过 B7/B9')"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
fi

# LLM 评估（串行，不占太多 GPU 资源冲突）
echo ""
echo "$(bold '  Phase 1 LLM 评估')"
for exp_name in b2_lora_only b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
    config="configs/${exp_name}.json"
    [ "$exp_name" = "b9_prefix_then_lora" ] && config="configs/b9_prefix_then_lora.json"
    [ "$exp_name" = "b7_dest_rs_warm" ]    && config="configs/b7_dest_rs_warm.json"
    llm_eval_job "$GPU0" "$BASE_TAG" "$config" "$exp_name"
done
wait_all_jobs

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: 多 seed
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "$(bold '══════════════════════════════════════════')"
echo "$(bold '  Phase 2: 多 seed (B3/B4/B7/B9 × each)')"
echo "$(bold '══════════════════════════════════════════')"
echo ""

if [ "$DUAL_GPU" = "1" ]; then
    # ── 双卡：跨 seed 并行，所有 seed 的 B3+B4 一起塞进槽位 ──
    echo "  提交所有 seed 的 B3 + B4（独立任务，填满 4 槽）..."
    for seed in $SEEDS; do
        SEED_TAG="${BASE_TAG}_seed${seed}"
        train_job "$GPU0" "$SEED_TAG" configs/b3_prefix_only.json b3_prefix_only --seed "$seed"
        train_job "$GPU1" "$SEED_TAG" configs/b4_prefix_lora.json b4_prefix_lora --seed "$seed"
    done
    wait_all_jobs || FAIL_COUNT=$((FAIL_COUNT + 1))

    # ── B7/B9 依赖各自 seed 的 B3 ──
    echo ""
    echo "  提交所有 seed 的 B7 + B9（依赖 B3）..."
    for seed in $SEEDS; do
        SEED_TAG="${BASE_TAG}_seed${seed}"
        B3_SEED_PREFIX="${PROJECT_DIR}/output/other/${SEED_TAG}/b3_prefix_only/prefix_bank.pt"
        if [ -f "$B3_SEED_PREFIX" ]; then
            B3_SEED_OTHER="${PROJECT_DIR}/output/other/${SEED_TAG}/b3_prefix_only"
            train_job "$GPU0" "$SEED_TAG" configs/b7_dest_rs_warm.json    b7_dest_rs_warm    --seed "$seed" --warm-start-dir "$B3_SEED_OTHER"
            train_job "$GPU1" "$SEED_TAG" configs/b9_prefix_then_lora.json b9_prefix_then_lora --seed "$seed" --warm-start-dir "$B3_SEED_OTHER"
        else
            echo "  $(red "B3 seed=${seed} 缺失，跳过 B7/B9")"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
    wait_all_jobs || FAIL_COUNT=$((FAIL_COUNT + 1))
else
    # ── 单卡串行 ──
    for seed in $SEEDS; do
        SEED_TAG="${BASE_TAG}_seed${seed}"
        echo "$(bold "── Seed ${seed} ──")"
        train_job "$GPU0" "$SEED_TAG" configs/b3_prefix_only.json b3_prefix_only --seed "$seed"
        train_job "$GPU0" "$SEED_TAG" configs/b4_prefix_lora.json b4_prefix_lora --seed "$seed"
        wait_all_jobs

        B3_SEED_PREFIX="${PROJECT_DIR}/output/other/${SEED_TAG}/b3_prefix_only/prefix_bank.pt"
        if [ -f "$B3_SEED_PREFIX" ]; then
            B3_SEED_OTHER="${PROJECT_DIR}/output/other/${SEED_TAG}/b3_prefix_only"
            train_job "$GPU0" "$SEED_TAG" configs/b7_dest_rs_warm.json    b7_dest_rs_warm    --seed "$seed" --warm-start-dir "$B3_SEED_OTHER"
            train_job "$GPU0" "$SEED_TAG" configs/b9_prefix_then_lora.json b9_prefix_then_lora --seed "$seed" --warm-start-dir "$B3_SEED_OTHER"
            wait_all_jobs
        fi
    done
fi

# Phase 2 评估
echo ""
echo "$(bold '  Phase 2 LLM 评估')"
for seed in $SEEDS; do
    SEED_TAG="${BASE_TAG}_seed${seed}"
    for exp_name in b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
        config="configs/${exp_name}.json"
        [ "$exp_name" = "b9_prefix_then_lora" ] && config="configs/b9_prefix_then_lora.json"
        [ "$exp_name" = "b7_dest_rs_warm" ]    && config="configs/b7_dest_rs_warm.json"
        llm_eval_job "$GPU0" "$SEED_TAG" "$config" "$exp_name"
    done
done
wait_all_jobs

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: 分析脚本（不需要 GPU 训练，单进程即可）
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "$(bold '══════════════════════════════════════════')"
echo "$(bold '  Phase 3: 分析')"
echo "$(bold '══════════════════════════════════════════')"

# ── 3a: Bootstrap CI ──
echo ""
echo "── 3a: Bootstrap CI ──"
BOOTSTRAP_PAIRS=(
    "B3_vs_B4|${BASE_TAG}|b3_prefix_only|b4_prefix_lora"
    "B3_vs_B7|${BASE_TAG}|b3_prefix_only|b7_dest_rs_warm"
    "B3_vs_B9|${BASE_TAG}|b3_prefix_only|b9_prefix_then_lora"
    "B4_vs_B7|${BASE_TAG}|b4_prefix_lora|b7_dest_rs_warm"
    "B4_vs_B9|${BASE_TAG}|b4_prefix_lora|b9_prefix_then_lora"
)

for pair_info in "${BOOTSTRAP_PAIRS[@]}"; do
    IFS='|' read -r label tag exp_a exp_b <<< "$pair_info"
    eval_a="${PROJECT_DIR}/output/need/${tag}/${exp_a}/strategy_eval_llm.json"
    eval_b="${PROJECT_DIR}/output/need/${tag}/${exp_b}/strategy_eval_llm.json"
    if [ -f "$eval_a" ] && [ -f "$eval_b" ]; then
        echo "  ${label}..."
        python scripts/bootstrap_confidence.py \
            -i "$eval_a" -c "$eval_b" \
            --label-a "${exp_a}" --label-b "${exp_b}" \
            --n-bootstrap "$N_BOOTSTRAP" \
            --out "${PROJECT_DIR}/output/need/${tag}/bootstrap_${label}.json" \
            > "${LOG_DIR}/bootstrap_${label}.log" 2>&1
    fi
done

for seed in $SEEDS; do
    SEED_TAG="${BASE_TAG}_seed${seed}"
    for exp_a in b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
        for exp_b in b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
            [ "$exp_a" = "$exp_b" ] && continue
            eval_a="${PROJECT_DIR}/output/need/${SEED_TAG}/${exp_a}/strategy_eval_llm.json"
            eval_b="${PROJECT_DIR}/output/need/${SEED_TAG}/${exp_b}/strategy_eval_llm.json"
            [ -f "$eval_a" ] && [ -f "$eval_b" ] || continue
            python scripts/bootstrap_confidence.py \
                -i "$eval_a" -c "$eval_b" \
                --label-a "${exp_a}" --label-b "${exp_b}" \
                --n-bootstrap "$N_BOOTSTRAP" \
                --out "${PROJECT_DIR}/output/need/${SEED_TAG}/bootstrap_${exp_a}_vs_${exp_b}.json" \
                > "${LOG_DIR}/bootstrap_${SEED_TAG}_${exp_a}_vs_${exp_b}.log" 2>&1
        done
    done
done

# ── 3b: 人工评估抽样 ──
echo ""
echo "── 3b: 人工评估抽样 ──"
HUMAN_SWAP_ARGS=()
HUMAN_LABELS=()
for exp_name in b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
    swap="${PROJECT_DIR}/output/need/${BASE_TAG}/${exp_name}/swap_samples_valid.jsonl"
    if [ -f "$swap" ]; then
        HUMAN_SWAP_ARGS+=("--swap" "$swap")
        HUMAN_LABELS+=("$exp_name")
    fi
done
if [ ${#HUMAN_SWAP_ARGS[@]} -gt 0 ]; then
    python scripts/sample_for_human_eval.py \
        "${HUMAN_SWAP_ARGS[@]}" \
        --labels "${HUMAN_LABELS[@]}" \
        --contexts-per-model "$HUMAN_EVAL_CONTEXTS" \
        --out "${PROJECT_DIR}/human_eval_samples.csv" \
        > "${LOG_DIR}/sample_human_eval.log" 2>&1
    echo "  ✓ human_eval_samples.csv"
fi

# ── 3c: 表示空间分析 ──
echo ""
echo "── 3c: 表示空间分析 ──"
for exp_name in b4_prefix_lora b9_prefix_then_lora; do
    ckpt="${PROJECT_DIR}/output/other/${BASE_TAG}/${exp_name}"
    need="${PROJECT_DIR}/output/need/${BASE_TAG}/${exp_name}"
    if [ -d "$ckpt" ] && [ -f "${need}/label_map.json" ]; then
        python scripts/analyze_lora_prefix_interference.py \
            --checkpoint-dir "$ckpt" --need-dir "$need" \
            --split valid --max-samples "$REPR_MAX_SAMPLES" \
            --out "${need}/lora_prefix_interference.json" \
            > "${LOG_DIR}/analyze_interference_${exp_name}.log" 2>&1
        echo "  ✓ ${exp_name}/lora_prefix_interference.json"
    fi
done

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
echo ""
echo "$(bold '═══════════════════════════════════════════════════════════')"
echo "$(bold '  完成')"
echo "$(bold '═══════════════════════════════════════════════════════════')"
echo "  ${START_TIME} → ${END_TIME}"
echo "  失败: ${FAIL_COUNT}"
echo ""

for tag in "$BASE_TAG" $(for s in $SEEDS; do echo "${BASE_TAG}_seed${s}"; done); do
    need_dir="${PROJECT_DIR}/output/need/${tag}"
    [ -d "$need_dir" ] || continue
    echo "  ${tag}/"
    for exp_name in b2_lora_only b3_prefix_only b4_prefix_lora b7_dest_rs_warm b9_prefix_then_lora; do
        ef="${need_dir}/${exp_name}/strategy_eval_llm.json"
        if [ -f "$ef" ]; then
            acc=$(python3 -c "import json; print(f\"{json.load(open('$ef'))['overall_accuracy']*100:.2f}%\")" 2>/dev/null || echo "—")
            echo "    ${exp_name}: ${acc}"
        fi
    done
done
echo ""
