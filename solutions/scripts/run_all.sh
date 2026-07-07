#!/usr/bin/env bash
# =============================================================================
# run_all.sh — One-click parallel experiment orchestrator
#
# Spawns S1, S2×3, S3, S4 on dual-GPU machines (RTX 5880, 48GB each).
# Two experiments per GPU (4-bit Qwen3-8B ≈ 8-12GB each, safe fit).
# All stdout/stderr captured to per-job log files.
#
# Usage:
#   bash solutions/scripts/run_all.sh              # full run, 2 GPUs
#   bash solutions/scripts/run_all.sh --dry-run    # preview only
#   bash solutions/scripts/run_all.sh --skip-s2    # skip S2 orth variants
#   bash solutions/scripts/run_all.sh --eval-only  # only S3 + S4
#   nohup bash solutions/scripts/run_all.sh &      # background + survive SSH hangup
# =============================================================================

set -o pipefail

# ── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOLUTIONS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$SOLUTIONS_DIR")"
cd "$PROJECT_ROOT" || exit 1

# User-tunable defaults
GPUS=2
SERIAL=false
TRAIN_GPU=0
DATASET_TAG="casino_augmented_new_fix_seed42"
# MODEL_PATH: set this to your Qwen3-8B path, or export MODEL_PATH before running
MODEL_PATH="${MODEL_PATH:-/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B}"
# S3/S4 evaluation checkpoints (existing baselines from main experiments in output/)
B3_CHECKPOINT="output/other/${DATASET_TAG}/b3_prefix_only"
B9_CHECKPOINT="output/other/${DATASET_TAG}/b9_prefix_then_lora"
CPD_CHECKPOINT="output/other/${DATASET_TAG}/b4_prefix_lora"
# S1/S2 solutions output goes to solutions/output/ (set via output_root in JSON configs)
CPD_SAMPLES=50
ATTN_SAMPLES=20
PYTHON="${PYTHON:-python}"

# Logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="solutions/output/logs/${TIMESTAMP}"
RUN_LOG="${LOG_DIR}/run_all.log"
STATUS_FILE="${LOG_DIR}/status.txt"
LOCK_DIR="solutions/output/run_all.lock"

# Flags (overridden by CLI args)
DRY_RUN=false
EVAL_ONLY=false
SKIP_S1=false
SKIP_S2=false
SKIP_S3=false
SKIP_S4=false
SKIP_STRATEGY_EVAL=false

# ── Parse CLI ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)        GPUS="$2"; shift 2 ;;
        --serial)      SERIAL=true; shift ;;
        --train-gpu)   TRAIN_GPU="$2"; shift 2 ;;
        --dataset-tag) DATASET_TAG="$2"; shift 2 ;;
        --dry-run)     DRY_RUN=true; shift ;;
        --eval-only)   EVAL_ONLY=true; shift ;;
        --skip-s1)     SKIP_S1=true; shift ;;
        --skip-s2)     SKIP_S2=true; shift ;;
        --skip-s3)     SKIP_S3=true; shift ;;
        --skip-s4)     SKIP_S4=true; shift ;;
        --skip-strategy-eval) SKIP_STRATEGY_EVAL=true; shift ;;
        --cpd-samples) CPD_SAMPLES="$2"; shift 2 ;;
        --attn-samples) ATTN_SAMPLES="$2"; shift 2 ;;
        --b3-checkpoint) B3_CHECKPOINT="$2"; shift 2 ;;
        --b9-checkpoint) B9_CHECKPOINT="$2"; shift 2 ;;
        --cpd-checkpoint) CPD_CHECKPOINT="$2"; shift 2 ;;
        --model-path)   MODEL_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
if mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "$$" > "${LOCK_DIR}/pid"
    trap 'rm -rf "$LOCK_DIR"' EXIT
else
    echo "Another run_all.sh appears to be running. Lock: $LOCK_DIR"
    if [[ -f "${LOCK_DIR}/pid" ]]; then
        echo "Existing PID: $(cat "${LOCK_DIR}/pid")"
    fi
    echo "If this is stale, remove it with: rm -rf $LOCK_DIR"
    exit 1
fi

# Array to track PIDs and their labels
declare -a PIDS=()
declare -a PID_LABELS=()
declare -a PID_GPUS=()
declare -a PID_STATUS=()
declare -a PID_LOG_FILES=()

# Color helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() {
    local msg="[$(date +%H:%M:%S)] $1"
    echo -e "$msg" | tee -a "$RUN_LOG"
}

log_green()  { log "${GREEN}$1${NC}"; }
log_red()    { log "${RED}$1${NC}"; }
log_yellow() { log "${YELLOW}$1${NC}"; }
log_cyan()   { log "${CYAN}$1${NC}"; }

# ── Job launcher ─────────────────────────────────────────────────────────────
launch_job() {
    local label="$1"
    local gpu_id="$2"
    local log_file="${LOG_DIR}/${label}.log"

    shift 2  # remaining args = command

    if $DRY_RUN; then
        log_cyan "[DRY] $label → GPU $gpu_id  log=$log_file"
        log_cyan "      CMD: $*"
        PIDS+=(-1)
        PID_LABELS+=("$label")
        PID_GPUS+=($gpu_id)
        PID_STATUS+=(0)  # Assume success for dry-run
        PID_LOG_FILES+=("$log_file")
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local busy_pids
        busy_pids=$(nvidia-smi --id="$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ' | xargs || true)
        if [[ -n "$busy_pids" ]]; then
            log_red "[BUSY] GPU $gpu_id already has compute PID(s): $busy_pids"
            log_red "       Refusing to launch $label on the same GPU."
            log_red "       Stop old jobs first, or choose another --train-gpu."
            exit 1
        fi
    fi

    log_cyan "[LAUNCH] $label → GPU $gpu_id  log=$log_file"

    # Run in background, redirect stdout+stderr to log file
    CUDA_VISIBLE_DEVICES=$gpu_id "$@" > "$log_file" 2>&1 &
    local pid=$!

    PIDS+=($pid)
    PID_LABELS+=("$label")
    PID_GPUS+=($gpu_id)
    PID_STATUS+=(0)  # placeholder, filled after wait
    PID_LOG_FILES+=("$log_file")

    log "        PID=$pid"
}

# ── Wait for specific PIDs and check exit codes ──────────────────────────────
wait_jobs() {
    local phase="$1"
    shift
    local pids=("$@")

    if ${DRY_RUN}; then
        log_yellow "[${phase}] DRY RUN — skipping wait"
        return 0
    fi

    log "[${phase}] Waiting for ${#pids[@]} job(s)..."

    local all_ok=true
    for pid in "${pids[@]}"; do
        wait "$pid"
        local rc=$?

        # Find this PID in our tracking arrays
        for i in "${!PIDS[@]}"; do
            if [[ "${PIDS[$i]}" -eq "$pid" ]]; then
                PID_STATUS[$i]=$rc
                local label="${PID_LABELS[$i]}"
                local logf="${PID_LOG_FILES[$i]}"
                if [[ $rc -eq 0 ]]; then
                    log_green "[DONE] $label (rc=0)"
                else
                    log_red "[FAIL] $label (rc=$rc)"
                    log_red "       tail -20 $logf"
                    tail -20 "$logf" | while IFS= read -r line; do
                        log_red "       | $line"
                    done
                    all_ok=false
                fi
                break
            fi
        done
    done

    if $all_ok; then
        return 0
    else
        return 1
    fi
}

# ── GPU round-robin scheduler ────────────────────────────────────────────────
# Simple: alternate jobs across GPUs. For N GPUs and M jobs, each job gets
# GPU index = job_index % N_GPUS. Phase 1 has 4 training jobs; with 2 GPUs
# that's 2 per GPU.

schedule_phase() {
    local phase="$1"
    shift
    local cmds=("$@")  # each element: "label|gpu_offset|command..."

    local phase_pids=()

    for cmd_spec in "${cmds[@]}"; do
        IFS='|' read -r label gpu_rule rest_cmd <<< "$cmd_spec"

        # Resolve GPU: "auto" = round-robin, otherwise literal number or rule
        local gpu_id
        if [[ "$gpu_rule" == "auto" ]]; then
            # Count currently running jobs per GPU (approximate)
            local jobs_on_0=0 jobs_on_1=0
            for i in "${!PID_GPUS[@]}"; do
                if [[ "${PID_GPUS[$i]}" -eq 0 ]]; then ((jobs_on_0++)); fi
                if [[ "${PID_GPUS[$i]}" -eq 1 ]]; then ((jobs_on_1++)); fi
            done
            if [[ $jobs_on_0 -le $jobs_on_1 ]]; then
                gpu_id=0
            else
                gpu_id=1
            fi
            # Clamp to available GPUs
            if [[ $gpu_id -ge $GPUS ]]; then gpu_id=0; fi
        else
            gpu_id=$gpu_rule
        fi

        # Split command string into array
        IFS=' ' read -ra cmd_array <<< "$rest_cmd"
        launch_job "$label" "$gpu_id" "${cmd_array[@]}"
        phase_pids+=(${PIDS[-1]})
    done

    # Wait for all jobs in this phase
    wait_jobs "$phase" "${phase_pids[@]}"
    local phase_ok=$?

    return $phase_ok
}

# ── Checkpoint existence check ───────────────────────────────────────────────
checkpoint_ready() {
    local ckpt_dir="$1"
    if [[ -z "$ckpt_dir" ]]; then return 1; fi
    if [[ -f "$ckpt_dir/prefix_bank.pt" ]]; then return 0; fi
    # Also check lora_adapter for LoRA-only checkpoints
    if [[ -d "$ckpt_dir/lora_adapter" ]]; then return 0; fi
    return 1
}

# ── Find S1 stage1 checkpoint ────────────────────────────────────────────────
find_s1_stage1_ckpt() {
    # Try exact path first
    local exact="solutions/output/other/${DATASET_TAG}/s1_clean_stage1_lora_only"
    if checkpoint_ready "$exact"; then echo "$exact"; return 0; fi

    # Search with wildcard
    for dir in solutions/output/other/*/s1_clean_stage1_lora_only; do
        if checkpoint_ready "$dir"; then echo "$dir"; return 0; fi
    done

    echo ""
    return 1
}

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

log "════════════════════════════════════════════════════════════════════"
log_cyan "  run_all.sh — Parallel Experiment Orchestrator"
log "════════════════════════════════════════════════════════════════════"
log "  GPUs:        $GPUS"
log "  Dataset tag: $DATASET_TAG"
log "  Dry run:     $DRY_RUN"
log "  Eval only:   $EVAL_ONLY"
log "  Skip S2:     $SKIP_S2"
log "  Skip S3:     $SKIP_S3"
log "  Skip S4:     $SKIP_S4"
log "  Skip judge:  $SKIP_STRATEGY_EVAL"
log "  Log dir:     $LOG_DIR"
log "  Timestamp:   $TIMESTAMP"
log "════════════════════════════════════════════════════════════════════"
log ""

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Parallel Training (2 jobs at a time, 1 per GPU)
#   Phase 1a: S1_stage1 (GPU 0) + S2_qk  (GPU 1)
#   Phase 1b: S2_vo       (GPU 0) + S2_full(GPU 1)
# ═══════════════════════════════════════════════════════════════════════════

if $EVAL_ONLY; then
    log_yellow "[Phase 1] SKIPPED (eval-only mode)"
elif $SERIAL; then
    log_cyan "[Phase 1] SERIAL training on GPU ${TRAIN_GPU}"
    if ! $SKIP_S2; then
        schedule_phase "S2_qk" \
            "S2_qk|${TRAIN_GPU}|$PYTHON train.py --config solutions/configs/s2_param_orth_qk.json --dataset-tag $DATASET_TAG"
        schedule_phase "S2_vo" \
            "S2_vo|${TRAIN_GPU}|$PYTHON train.py --config solutions/configs/s2_param_orth_vo.json --dataset-tag $DATASET_TAG"
        schedule_phase "S2_full" \
            "S2_full|${TRAIN_GPU}|$PYTHON train.py --config solutions/configs/s2_param_orth_full.json --dataset-tag $DATASET_TAG"
    fi
    if ! $SKIP_S1; then
        schedule_phase "S1_stage1" \
            "S1_stage1|${TRAIN_GPU}|$PYTHON solutions/scripts/run_s1_reverse.py --stage1-only --dataset-tag $DATASET_TAG"
    fi
    log ""
else
    log_cyan "[Phase 1a] Parallel: S2_qk on GPU0 + S2_vo on GPU1"
    PHASE1A_CMDS=()
    if ! $SKIP_S2; then
        PHASE1A_CMDS+=("S2_qk|0|$PYTHON train.py --config solutions/configs/s2_param_orth_qk.json --dataset-tag $DATASET_TAG")
        PHASE1A_CMDS+=("S2_vo|1|$PYTHON train.py --config solutions/configs/s2_param_orth_vo.json --dataset-tag $DATASET_TAG")
    fi
    schedule_phase "Phase1a" "${PHASE1A_CMDS[@]}"
    log ""

    if ! $SKIP_S2; then
        log_cyan "[Phase 1b] Single: S2_full on GPU0"
        PHASE1B_CMDS=(
            "S2_full|0|$PYTHON train.py --config solutions/configs/s2_param_orth_full.json --dataset-tag $DATASET_TAG"
        )
        schedule_phase "Phase1b" "${PHASE1B_CMDS[@]}"
        log ""
    fi

    if ! $SKIP_S1; then
        log_cyan "[Phase 1c] Single: S1_stage1 on GPU0"
        PHASE1C_CMDS=(
            "S1_stage1|0|$PYTHON solutions/scripts/run_s1_reverse.py --stage1-only --dataset-tag $DATASET_TAG"
        )
        schedule_phase "Phase1c" "${PHASE1C_CMDS[@]}"
        log ""
    fi
fi
# Phase 2: S1 Stage 2 (depends on S1_stage1 success)
# ═══════════════════════════════════════════════════════════════════════════

S1_STAGE1_OK=false
for i in "${!PID_LABELS[@]}"; do
    if [[ "${PID_LABELS[$i]}" == "S1_stage1" && "${PID_STATUS[$i]}" -eq 0 ]]; then
        S1_STAGE1_OK=true
        break
    fi
done

# In dry-run mode or when S1 was skipped (results from previous run), skip Phase 2
if $SKIP_S1; then
    log_yellow "[Phase 2] SKIPPED (--skip-s1: S1 results from previous run)"
elif $DRY_RUN || $S1_STAGE1_OK; then
    S1_CKPT=$(find_s1_stage1_ckpt)

    if $EVAL_ONLY; then
        log_yellow "[Phase 2] SKIPPED (eval-only mode)"
    elif [[ -z "$S1_CKPT" ]] && ! $DRY_RUN; then
        log_red "[Phase 2] SKIPPED: S1_stage1 checkpoint not found"
        log_red "         Expected: solutions/output/other/${DATASET_TAG}/s1_clean_stage1_lora_only/"
    else
        log_cyan "┌─ Phase 2: S1 Stage 2 ─────────────────────────────────────────┐"

        if $DRY_RUN; then
            S1_CKPT="solutions/output/other/${DATASET_TAG}/s1_clean_stage1_lora_only"
        fi

        PHASE2_CMDS=(
            "S1_stage2|${TRAIN_GPU}|$PYTHON solutions/scripts/run_s1_reverse.py --stage2-only --dataset-tag $DATASET_TAG --stage1-checkpoint $S1_CKPT"
        )

        schedule_phase "Phase2" "${PHASE2_CMDS[@]}"

        log_cyan "└─ Phase 2 done ────────────────────────────────────────────────┘"
        log ""
    fi
else
    log_red "[Phase 2] SKIPPED: S1_stage1 failed (check ${LOG_DIR}/S1_stage1.log)"
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Evaluation (S4 CPD + S3 Attention)
# ═══════════════════════════════════════════════════════════════════════════

EVAL_CMDS=()

if ! $SKIP_S4; then
    # S4 diagnostic first (fast), then full eval if diagnostic passes
    EVAL_CMDS+=("S4_cpd_diag|0|$PYTHON solutions/scripts/run_s4_cpd_eval.py --checkpoint $CPD_CHECKPOINT --diagnostic-only --dataset-tag $DATASET_TAG --model-path $MODEL_PATH")
    # Note: S4 full eval runs after diagnostic, handled separately below
fi

if ! $SKIP_S3; then
    EVAL_CMDS+=("S3_attn|1|$PYTHON solutions/scripts/attn_compare_b3_b9.py --b3-checkpoint $B3_CHECKPOINT --b9-checkpoint $B9_CHECKPOINT --num-samples $ATTN_SAMPLES --dataset-tag $DATASET_TAG --model-path $MODEL_PATH")
fi

if [[ ${#EVAL_CMDS[@]} -gt 0 ]]; then
    log_cyan "┌─ Phase 3: Evaluation ─────────────────────────────────────────┐"

    schedule_phase "Phase3" "${EVAL_CMDS[@]}"

    log_cyan "└─ Phase 3 done ────────────────────────────────────────────────┘"
    log ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: S4 Full CPD (after diagnostic confirms feasibility)
# ═══════════════════════════════════════════════════════════════════════════

if ! $SKIP_S4; then
    S4_DIAG_OK=false
    for i in "${!PID_LABELS[@]}"; do
        if [[ "${PID_LABELS[$i]}" == "S4_cpd_diag" && "${PID_STATUS[$i]}" -eq 0 ]]; then
            S4_DIAG_OK=true
            break
        fi
    done

    if $DRY_RUN || $S4_DIAG_OK; then
        log_cyan "┌─ Phase 4: S4 Full CPD Evaluation ─────────────────────────────┐"

        PHASE4_CMDS=(
            "S4_cpd_full|0|$PYTHON solutions/scripts/run_s4_cpd_eval.py --checkpoint $CPD_CHECKPOINT --max-samples $CPD_SAMPLES --dataset-tag $DATASET_TAG --model-path $MODEL_PATH"
        )

        schedule_phase "Phase4" "${PHASE4_CMDS[@]}"

        log_cyan "└─ Phase 4 done ────────────────────────────────────────────────┘"
        log ""
    else
        log_red "[Phase 4] SKIPPED: S4 diagnostic failed (check ${LOG_DIR}/S4_cpd_diag.log)"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# Final Summary
# ═════════════════════════════════════════════════════════════════════════════

log ""
log "════════════════════════════════════════════════════════════════════"
# Phase 5: LLM strategy-control evaluation for newly trained solution runs
if ! $SKIP_STRATEGY_EVAL; then
    STRATEGY_EVAL_CMDS=()

    add_strategy_eval_job() {
        local label="$1"
        local gpu_id="$2"
        local exp_name="$3"
        local exp_dir="solutions/output/need/${DATASET_TAG}/${exp_name}"
        local jsonl_path="${exp_dir}/swap_samples_valid.jsonl"
        local config_path="${exp_dir}/run_config.json"
        local out_path="${exp_dir}/strategy_eval_llm.json"

        if $DRY_RUN || [[ -f "$jsonl_path" ]]; then
            STRATEGY_EVAL_CMDS+=("${label}|${gpu_id}|$PYTHON scripts/evaluate_strategy_control_llm.py --model-path $MODEL_PATH --config $config_path --jsonl $jsonl_path --out $out_path")
        else
            log_yellow "[Phase 5] SKIP $label: missing $jsonl_path"
        fi
    }

    if ! $SKIP_S1; then
        add_strategy_eval_job "Eval_S1_clean" 0 "s1_clean_stage2_prefix_on_frozen_lora"
    fi
    if ! $SKIP_S2; then
        add_strategy_eval_job "Eval_S2_qk_clean" 1 "s2_clean_param_orth_qk"
        add_strategy_eval_job "Eval_S2_vo_clean" 0 "s2_clean_param_orth_vo"
        add_strategy_eval_job "Eval_S2_full_clean" 1 "s2_clean_param_orth_full"
    fi

    if [[ ${#STRATEGY_EVAL_CMDS[@]} -gt 0 ]]; then
        log_cyan "Phase 5: LLM Strategy Evaluation"
        for eval_cmd in "${STRATEGY_EVAL_CMDS[@]}"; do
            IFS='|' read -r eval_label _ <<< "$eval_cmd"
            schedule_phase "Phase5_${eval_label}" "$eval_cmd"
        done
        log_cyan "Phase 5 done"
        log ""
    else
        log_yellow "[Phase 5] SKIPPED: no solution swap_samples found"
    fi
else
    log_yellow "[Phase 5] SKIPPED (--skip-strategy-eval)"
fi

log_cyan "  RUN_ALL FINISHED"
log "════════════════════════════════════════════════════════════════════"

total=0 ok=0 fail=0

for i in "${!PID_LABELS[@]}"; do
    label="${PID_LABELS[$i]}"
    rc="${PID_STATUS[$i]}"
    gpu="${PID_GPUS[$i]}"
    logf="${PID_LOG_FILES[$i]}"

    ((total++))
    if [[ $rc -eq 0 ]]; then
        log "  ${GREEN}✓${NC} $label  (GPU $gpu)  → $logf"
        ((ok++))
    else
        log "  ${RED}✗${NC} $label  (GPU $gpu, rc=$rc)  → $logf"
        ((fail++))
    fi
done

log ""
log "  Total: $total  |  ${GREEN}Passed: $ok${NC}  |  ${RED}Failed: $fail${NC}"
log "  Log directory: $LOG_DIR"
log ""

# Write machine-readable status
{
    echo "timestamp=$TIMESTAMP"
    echo "dataset_tag=$DATASET_TAG"
    echo "gpus=$GPUS"
    echo "total=$total"
    echo "ok=$ok"
    echo "fail=$fail"
    for i in "${!PID_LABELS[@]}"; do
        echo "job|${PID_LABELS[$i]}|${PID_STATUS[$i]}|${PID_GPUS[$i]}|${PID_LOG_FILES[$i]}"
    done
} > "$STATUS_FILE"

log "  Status file: $STATUS_FILE"
log "════════════════════════════════════════════════════════════════════"

# Exit with failure if any job failed
if [[ $fail -gt 0 ]]; then
    exit 1
fi
exit 0
