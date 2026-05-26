# DEST-RS：面向谈判对话策略控制的混合 PEFT 实验工程

本目录是按照论文蓝图改造后的主实验工程。旧工程 `dest_strategy_peft` 只作为实现参考，新实验建议都在本目录中运行。

DEST-RS 的核心思想是：使用 **strategy-specific prefix bank** 承担高层策略 steering，使用 **shared LoRA branch** 承担共享生成适配，并通过 **局部-全局正交约束** 与 **策略分类辅助监督** 降低两类适配路径之间的干扰，从而提升策略可控性、生成质量和稳定性。

## 已实现方法

在配置文件中设置 `adapter_mode`：

| 模式 | Prefix | LoRA | Orth | Cls | 用途 |
|---|---:|---:|---:|---:|---|
| `lora_only` | 否 | 是 | 否 | 否 | B2：LoRA 单分支基线 |
| `prefix_only` | 是 | 否 | 否 | 否 | B3：Prefix 单分支基线 |
| `prefix_lora` | 是 | 是 | 否 | 否 | B4：Prefix+LoRA 混合基线 |
| `prefix_lora_orth` | 是 | 是 | 是 | 否 | B5：加入正交约束的消融版本 |
| `dest_rs` | 是 | 是 | 是 | 是 | B6：完整 DEST-RS 方法 |

数据处理默认使用 `multi_label_policy="drop"`，即只保留单策略标签样本，符合蓝图中“降低多标签噪声”的建议。

## 快速开始

先修改每个配置文件中的两个字段：

- `model_name_or_path`：本地大模型路径，例如本地 Qwen 路径。
- `dataset_dir`：CaSiNo 数据目录。代码支持两种目录结构：
  - `dataset_dir/casino_train.json`
  - `dataset_dir/split/casino_train.json`

先跑一个小规模冒烟实验：

```bash
python train.py --config configs/smoke_dest_rs.json
```

再运行核心消融实验：

```bash
python train.py --config configs/b2_lora_only.json
python train.py --config configs/b3_prefix_only.json
python train.py --config configs/b4_prefix_lora.json
python train.py --config configs/b5_prefix_lora_orth.json
python train.py --config configs/b6_dest_rs.json
```

## 输出文件

每次训练会保存：

- `tokenizer/`
- `lora_adapter/`
- `prefix_bank.pt`，其中包含 `prefix_bank` 和 `strategy_classifier`
- `label_map.json`
- `run_config.json`
- `metrics.json`
- `swap_samples_valid.jsonl`

## 损失函数

完整目标函数为：

```text
L = L_gen + lambda_orth * L_orth + lambda_cls * L_cls
```

其中：

- `L_gen`：自回归生成损失。
- `L_orth`：Prefix 增量表示与 LoRA 增量表示之间的局部-全局正交约束。
- `L_cls`：基于中间层 pooled hidden states 的策略分类辅助损失。

## 工具脚本

- `scripts/summarize_runs.py`：把多个训练目录中的 `metrics.json` 汇总为 CSV。
- `scripts/evaluate_generations.py`：计算 swap samples 的 distinct-n、平均长度和重复率。
- `scripts/dataset_stats.py`：统计当前标签策略下的 CaSiNo 策略分布。
- `scripts/make_ablation_configs.py`：根据一个模板批量生成消融实验配置。
- `scripts/make_followup_configs.py`：生成新蓝图需要的 lambda、alpha、多 seed 和效率分析配置。
- `scripts/analyze_orthogonality.py`：计算 Prefix/LoRA 表示增量的 cosine 与 Frobenius 指标。
- `scripts/build_judge_sheet.py`：导出 B4/B5 swap 样例对比表，供人工或 LLM judge 标注。
- `scripts/llm_judge_sheet.py`：调用 DeepSeek API 自动标注 B4/B5 judge 表格。
- `scripts/train_strategy_evaluator.py`：训练轻量外部策略评估器。
- `scripts/evaluate_strategy_control.py`：对 swap samples 计算 Strategy Accuracy 和 Macro-F1。
- `generate_swap_samples.py`：从 checkpoint 重新生成 swap-test 样例。
- `generate_interpolation_samples.py`：生成 prefix interpolation 定性分析样例。

## 新蓝图实验脚本

- `run_b2_b5_ablation.sh`：只运行 B2-B5 主消融实验。
- `run_followup_experiments.sh`：运行 `configs/followup/` 下的 lambda、alpha 和多 seed 补充实验。

生成或刷新补充实验配置：

```bash
python scripts/make_followup_configs.py \
  --out-dir configs/followup \
  --output-root ./outputs/followup
```

运行补充实验：

```bash
chmod +x run_followup_experiments.sh
./run_followup_experiments.sh
```

生成 B4/B5 judge 表格：

```bash
python scripts/build_judge_sheet.py \
  --b4-jsonl outputs/b4_prefix_lora/swap_samples_valid.jsonl \
  --b5-jsonl outputs/b5_prefix_lora_orth/swap_samples_valid.jsonl \
  --out outputs/judge_b4_vs_b5.csv
```

使用 DeepSeek API 自动标注 judge 表格：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"

python scripts/llm_judge_sheet.py \
  --input outputs/judge_b4_vs_b5.csv \
  --out outputs/judge_b4_vs_b5_labeled.csv \
  --model deepseek-v4-pro
```

先试跑 3 行：

```bash
python scripts/llm_judge_sheet.py \
  --input outputs/judge_b4_vs_b5.csv \
  --out outputs/judge_b4_vs_b5_labeled.csv \
  --model deepseek-v4-pro \
  --limit 3
```

计算正交性分析指标：

```bash
python scripts/analyze_orthogonality.py \
  --checkpoint-dir outputs/b5_prefix_lora_orth \
  --split test \
  --max-samples 64 \
  --out outputs/b5_orthogonality_test.json
```

## 推荐执行顺序

1. 运行 `configs/smoke_dest_rs.json`，确认模型路径、数据路径和依赖环境可用。
2. 运行 B2-B6 核心消融实验。
3. 使用 `scripts/summarize_runs.py` 汇总 `metrics.json`。
4. 使用 `scripts/evaluate_generations.py` 评估 `swap_samples_valid.jsonl` 的多样性和重复率。
5. 使用 `scripts/train_strategy_evaluator.py` 训练外部策略评估器，再用 `scripts/evaluate_strategy_control.py` 计算 Strategy Accuracy 和 Macro-F1。
