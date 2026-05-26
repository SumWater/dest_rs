#!/usr/bin/env bash
set -e

# 运行 make_followup_configs.py 生成的补充实验配置。
# 默认会运行 lambda_orth、orth_alpha、多 seed 三组实验。
# efficiency 目录中的 prefix length / LoRA rank 实验较多，默认不跑，可按需手动运行。

CONFIG_DIR="${1:-configs/followup}"

echo "============================================================"
echo "DEST-O 补充实验启动"
echo "配置目录：${CONFIG_DIR}"
echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

run_group() {
  local group_dir="$1"
  if [ ! -d "${group_dir}" ]; then
    echo "跳过：目录不存在 ${group_dir}"
    return
  fi

  for config in "${group_dir}"/*.json; do
    [ -e "${config}" ] || continue
    echo
    echo "============================================================"
    echo "运行配置：${config}"
    echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    python -u train.py --config "${config}"
    echo "完成配置：${config}"
  done
}

run_group "${CONFIG_DIR}/lambda_orth"
run_group "${CONFIG_DIR}/orth_alpha"
run_group "${CONFIG_DIR}/seeds"

echo
echo "============================================================"
echo "默认补充实验已完成"
echo "如需效率分析，请手动运行 ${CONFIG_DIR}/efficiency/*.json"
echo "结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
