# DEST-RS：面向谈判对话策略控制的混合 PEFT 实验工程

DEST-RS 的核心思想是：使用 **strategy-specific prefix bank** 承担高层策略 steering，使用 **shared LoRA branch** 承担共享生成适配，并通过 **局部-全局正交约束** 与 **策略分类辅助监督** 降低两类适配路径之间的干扰，从而提升策略可控性、生成质量和稳定性。

## 快速开始

修改配置文件中的两个字段：

- `model_name_or_path`：本地大模型路径
- `dataset_dir`：CaSiNo 数据目录（支持 `dataset_dir/casino_train.json` 或 `dataset_dir/split/casino_train.json`）

冒烟测试：

```bash
python train.py --config configs/smoke_dest_rs.json
```

## 实验模式

| 模式 | Prefix | LoRA | Orth | Cls | 用途 |
|---|:---:|:---:|:---:|:---:|---|
| `lora_only` | | Y | | | B2：LoRA 单分支基线 |
| `prefix_only` | Y | | | | B3：Prefix 单分支基线 |
| `prefix_lora` | Y | Y | | | B4：Prefix+LoRA 混合基线 |
| `prefix_lora_orth` | Y | Y | Y | | B5：加入正交约束 |
| `dest_rs` | Y | Y | Y | Y | B6：完整 DEST-RS |

## 损失函数

```
L = L_gen + λ_orth * L_orth + λ_cls * L_cls
```

- `L_gen`：自回归生成损失
- `L_orth`：Prefix 与 LoRA 增量表示的局部-全局正交约束
- `L_cls`：基于 delta_prefix（prefix 增量表示）的策略分类辅助损失

## 输出文件

每次训练保存：`tokenizer/`、`lora_adapter/`、`prefix_bank.pt`、`label_map.json`、`run_config.json`、`metrics.json`、`swap_samples_valid.jsonl`

## 工具脚本

| 脚本 | 用途 |
|---|---|
| `scripts/summarize_runs.py` | 多个训练目录的 metrics 汇总为 CSV |
| `scripts/evaluate_generations.py` | 计算 distinct-n、平均长度和重复率 |
| `scripts/analyze_orthogonality.py` | 计算 Prefix/LoRA 增量的 cosine 与 Frobenius 指标 |
| `scripts/dataset_stats.py` | 统计 CaSiNo 策略分布 |
| `scripts/train_strategy_evaluator.py` | 训练轻量外部策略评估器 |
| `scripts/evaluate_strategy_control.py` | 计算 Strategy Accuracy 和 Macro-F1 |
| `scripts/build_judge_sheet.py` | 导出 swap 样例对比表 |
| `scripts/llm_judge_sheet.py` | 调用 DeepSeek API 自动标注 judge 表格 |
| `generate_swap_samples.py` | 从 checkpoint 重新生成 swap 样例 |
| `generate_interpolation_samples.py` | 生成 prefix interpolation 定性分析样例 |
