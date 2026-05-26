#!/usr/bin/env bash
set -e

# 顺序运行新蓝图中的 B2-B5 主消融实验。
# B6 分类辅助头不再作为主方法，因此不放入本脚本。

echo "============================================================"
echo "DEST-O 主消融实验启动：B2-B5"
echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "当前目录：$(pwd)"
echo "Python 路径：$(which python)"
echo "Python 版本：$(python --version)"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "当前 GPU 状态："
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
fi

run_one() {
  local config="$1"
  local name
  name="$(basename "${config}" .json)"

  echo
  echo "============================================================"
  echo "准备运行：${name}"
  echo "配置文件：${config}"
  echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  if [ ! -f "${config}" ]; then
    echo "错误：找不到配置文件 ${config}"
    exit 1
  fi

  python -u train.py --config "${config}"

  echo "结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "实验 ${name} 已完成。"

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "运行后 GPU 状态："
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  fi
}

run_one "configs/b2_lora_only.json"
run_one "configs/b3_prefix_only.json"
run_one "configs/b4_prefix_lora.json"
run_one "configs/b5_prefix_lora_orth.json"

echo
echo "============================================================"
echo "B2-B5 主消融实验全部完成"
echo "结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
