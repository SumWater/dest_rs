#!/usr/bin/env bash
set -e

# 顺序运行 B2-B6 核心消融实验。
# 这个脚本不使用 tee 管道，尽量避免终端输出缓冲导致“看起来卡住”。

echo "============================================================"
echo "DEST-RS 核心消融实验启动"
echo "开始时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "当前目录：$(pwd)"
echo "Python 路径：$(which python)"
echo "Python 版本：$(python --version)"
echo "============================================================"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "当前 GPU 状态："
  nvidia-smi
else
  echo "未检测到 nvidia-smi，跳过 GPU 状态打印。"
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

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "运行前 GPU 状态："
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  fi

  python -u train.py --config "${config}"
  local code=$?

  echo "结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
  echo "退出码：${code}"

  if [ "${code}" -ne 0 ]; then
    echo "实验 ${name} 运行失败，脚本停止。"
    exit "${code}"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "运行后 GPU 状态："
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  fi

  echo "实验 ${name} 已完成。"
}

run_one "configs/b2_lora_only.json"
run_one "configs/b3_prefix_only.json"
run_one "configs/b4_prefix_lora.json"
run_one "configs/b5_prefix_lora_orth.json"
run_one "configs/b6_dest_rs.json"

echo
echo "============================================================"
echo "全部 B2-B6 核心消融实验已完成"
echo "结束时间：$(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
