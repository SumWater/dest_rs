# Solutions 实验说明

这个文件夹用于放置 Prefix + LoRA 控制失败后的补充解决方案实验。核心目标是验证：

> 当共享 LoRA 修改了模型的注意力投影空间后，策略专属 Prefix 的控制信号还能否穿透修改后的模型，并在最终生成中保留策略区分度？

## 要解决的问题

任务是策略可控的谈判回复生成：给定对话上下文和 9 种谈判策略之一，生成符合该策略的回复。

已有结果显示：

- Prefix-only 的策略控制较好，但 PPL 较差，回复不够流畅。
- Prefix + LoRA 联合训练后，PPL 明显下降，回复更流畅，但策略准确率下降。
- 先训练 Prefix、冻结 Prefix、再训练 LoRA 后，Prefix 权重没变，但控制仍然下降。

这说明问题不只是训练时的梯度冲突，而是推理阶段的环境变化：Prefix 学到的策略信号依赖原始 attention 空间；LoRA 修改 Q/K/V/O 投影后，Prefix 信号经过新的 attention 层时被扭曲或淹没。

## 当前实现的方案

| 编号 | 实验 | 目的 |
|---|---|---|
| S1 | Reverse Curriculum | 先训练 LoRA，再冻结 LoRA 训练 Prefix，让 Prefix 从一开始就适应 LoRA 修改后的 attention 空间。 |
| S2 | Parameter-Level Orthogonality | 在参数交互层面约束 LoRA 更新，减少 LoRA 对 Prefix key/value 通道的干扰。 |
| S3 | Attention Analysis | 比较 B3/B9 中模型对 Prefix token 的注意力质量，判断是否存在 Prefix attention 被压缩。 |
| S4 | Contrastive Prefix Decoding | 推理期方法，用目标 Prefix logits 减去 neutral Prefix logits，放大策略差异信号。 |

注意：clean 版本实验都设置了：

```json
"inject_strategy_text": false
```

也就是说，策略控制应该来自 Prefix swap，而不是显式策略文本指令。

## 目录结构

```text
solutions/
  README.md
  configs/
    s1_reverse.json
    s2_param_orth_qk.json
    s2_param_orth_vo.json
    s2_param_orth_full.json
  scripts/
    run_all.sh
    run_s1_reverse.py
    run_s4_cpd_eval.py
    attn_compare_b3_b9.py
  src/
    losses_param_orth.py
    inference_cpd.py
    attention_utils.py
  output/
    need/
    other/
    logs/
```

## 推荐运行方式

在项目根目录运行：

```bash
nohup bash solutions/scripts/run_all.sh \
  --dataset-tag casino_augmented_new_fix_seed42 \
  --model-path /home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B \
  > run_all_nohup.log 2>&1 &
```

查看日志：

```bash
tail -f run_all_nohup.log
tail -f solutions/output/logs/*/run_all.log
```

正式运行前可以先 dry-run：

```bash
bash solutions/scripts/run_all.sh \
  --dataset-tag casino_augmented_new_fix_seed42 \
  --model-path /home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B \
  --dry-run
```

## 当前调度策略

`run_all.sh` 现在采用保守调度。脚本会：

- 创建 `solutions/output/run_all.lock`，防止同时启动多个 `run_all.sh`
- 每次启动任务前检查目标 GPU 是否已有 compute 进程
- 如果目标 GPU 已经忙碌，则拒绝继续启动，避免同一 GPU 叠多个实验导致 OOM

默认顺序：

```text
Phase 1a: GPU0 -> S2_qk,   GPU1 -> S2_vo
Phase 1b: GPU0 -> S2_full
Phase 1c: GPU0 -> S1_stage1
Phase 2:  GPU0 -> S1_stage2
Phase 3:  S4 diagnostic + S3 attention analysis
Phase 4:  S4 full CPD evaluation
Phase 5:  LLM strategy evaluation，逐个评估
```

常用参数：

```bash
--serial               所有训练任务串行执行
--train-gpu 0          指定串行任务和 S1 stage2 使用的 GPU
--skip-s1              跳过 S1
--skip-s2              跳过 S2
--skip-s3              跳过 S3 attention analysis
--skip-s4              跳过 S4 CPD
--skip-strategy-eval   跳过 LLM 策略评估
--eval-only            不训练，只运行评估相关脚本
```

如果异常退出后留下旧锁：

```bash
rm -rf solutions/output/run_all.lock
```

## 主要输出

重点看这些目录：

```text
solutions/output/need/<dataset_tag>/s1_clean_stage2_prefix_on_frozen_lora/
solutions/output/need/<dataset_tag>/s2_clean_param_orth_qk/
solutions/output/need/<dataset_tag>/s2_clean_param_orth_vo/
solutions/output/need/<dataset_tag>/s2_clean_param_orth_full/
```

每个训练目录通常包含：

```text
metrics.json
run_config.json
label_map.json
swap_samples_valid.jsonl
strategy_eval_llm.json
```

主要比较：

- `metrics.json` 里的 PPL
- `strategy_eval_llm.json` 里的 overall strategy accuracy
- S1/S2 是否能在保持 LoRA 低 PPL 的同时恢复 Prefix 控制能力

## 各方案说明

### S1: Reverse Curriculum

思路是反转训练顺序：

1. Stage 1：只训练共享 LoRA，让模型先学会流畅谈判回复。
2. Stage 2：冻结 base model 和 LoRA，只训练策略专属 Prefix。

这样 Prefix 不是在原始 attention 空间里学习，而是在 LoRA 已经修改后的 attention 空间里学习。它理论上更容易适应推理时真正面对的模型环境。

输出目录：

```text
s1_clean_stage1_lora_only
s1_clean_stage2_prefix_on_frozen_lora
```

### S2: Parameter-Level Orthogonality

S2 试图在参数交互层面减少 LoRA 对 Prefix 的干扰。

当前包括三个变体：

```text
s2_clean_param_orth_qk
s2_clean_param_orth_vo
s2_clean_param_orth_full
```

含义：

- `qk`：约束 LoRA Query 更新不要过度读取 Prefix key 子空间。
- `vo`：约束 LoRA Output 更新不要过度改写 Prefix value 贡献。
- `full`：同时使用 Q-K 和 V-O 两种约束。

实现注意：

- `losses_param_orth.py` 使用 LoRA 低秩因子计算，不显式构造 dense `B @ A` 矩阵。
- 对 4-bit 量化模型，base projection weight 可能是 packed weight，代码会先尝试 dequantize。
- `param_orth_every_n_steps` 默认不是每步计算，以降低显存和时间开销。

### S3: Attention Analysis

S3 不是主要训练方案，而是诊断工具。

它比较 B3 和 B9 的 Prefix attention，判断 LoRA 后模型是否：

- 压缩了 Prefix attention mass
- 重新分配了 Prefix attention
- 放大了 Prefix attention 但仍然控制失败

这个结果用于判断是否值得继续做 Prefix attention gating。

### S4: Contrastive Prefix Decoding

S4 是推理期方法，不额外训练。

每一步生成时计算：

```text
final_logits = target_logits + alpha * (target_logits - neutral_logits)
```

直觉是用 neutral Prefix 去消除通用流畅度成分，放大目标策略 Prefix 带来的差异。

它适合作为推理期补救和机制诊断，但不是训练阶段的根本修复。

## 手动运行命令

单独运行 S1：

```bash
python solutions/scripts/run_s1_reverse.py \
  --dataset-tag casino_augmented_new_fix_seed42
```

单独运行 S2：

```bash
python train.py --config solutions/configs/s2_param_orth_qk.json \
  --dataset-tag casino_augmented_new_fix_seed42

python train.py --config solutions/configs/s2_param_orth_vo.json \
  --dataset-tag casino_augmented_new_fix_seed42

python train.py --config solutions/configs/s2_param_orth_full.json \
  --dataset-tag casino_augmented_new_fix_seed42
```

单独对一个实验做 LLM 策略评估：

```bash
python scripts/evaluate_strategy_control_llm.py \
  --model-path /home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B \
  --config solutions/output/need/casino_augmented_new_fix_seed42/s1_clean_stage2_prefix_on_frozen_lora/run_config.json \
  --jsonl solutions/output/need/casino_augmented_new_fix_seed42/s1_clean_stage2_prefix_on_frozen_lora/swap_samples_valid.jsonl \
  --out solutions/output/need/casino_augmented_new_fix_seed42/s1_clean_stage2_prefix_on_frozen_lora/strategy_eval_llm.json
```

## 注意事项

- 当前项目里的 Prefix 实现是把策略专属 soft prefix 拼到 `inputs_embeds` 前面，更接近 strategy-specific soft prompt，而不是经典每层 KV Prefix Tuning。
- 不要把 `inject_strategy_text=true` 的结果和 clean Prefix 控制结果混在一起比较。
- 如果手动 kill 了实验，重新运行前检查：

```bash
ps -ef | grep -E "run_all|train.py|run_s1_reverse" | grep -v grep
nvidia-smi
rm -rf solutions/output/run_all.lock
```

- 如果从 Windows 同步 shell 脚本到 Linux，建议执行：

```bash
find solutions/scripts -name "*.sh" -exec sed -i 's/\r$//' {} \;
```
