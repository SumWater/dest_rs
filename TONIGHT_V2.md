# V2 实验计划（2026-05-28）

## 问题诊断

V1 实验中 prefix 未能学到策略信息，三个根因：

| 问题 | V1 现状 | 后果 |
|---|---|---|
| cls 读的是 h_both（混合表示） | losses.py:230 | orth 推远、cls 需要合作，两个目标矛盾 → B6 正交性反而比 B5 差 |
| λ_orth=0.05 太小 | 加权后仅占总 loss 0.07% | 正交约束对梯度几乎无影响 |
| orth_every_n_steps=20 | 95% 的步骤无 orth 梯度 | prefix/LoRA 在非 orth 步自由纠缠 |

## V2 改动

### 代码改动（1 行）

`src/losses.py` 第 230 行：

```python
# V1: cls 从 prefix+LoRA 混合表示学，与 orth 方向矛盾
cls_loss, cls_logits = classification_loss(hybrid, h_both, labels, strategy_ids)

# V2: cls 从 prefix 增量表示学，与 orth 方向一致
cls_loss, cls_logits = classification_loss(hybrid, delta_prefix, labels, strategy_ids)
```

### 超参数改动

| 参数 | V1 | V2 | 理由 |
|---|---|---|---|
| lambda_orth | 0.05 | 1.0 | 让 orth 占总 loss ~1-3% |
| lambda_cls | 0.2 | 2.0 | 让 cls 占总 loss ~5-7% |
| orth_every_n_steps | 20 | 1 | 每步都有正交信号 |
| orth_start_step | 50 | 0 | 从第一步就开始 |
| num_epochs | 2 | 3 | 给 prefix 更多学习时间 |

### 实验组

| 实验 | 配置文件 | 目的 |
|---|---|---|
| B5v2 | `configs/v2/b5v2_orth_strong.json` | 仅强化 orth（无 cls），验证正交约束是否真正生效 |
| B6v2 | `configs/v2/b6v2_dest_rs_fixed.json` | orth + 修复的 cls，验证 prefix 能否独立承载策略 |

## 运行指令

```bash
conda activate tuning
nohup bash run_tonight_v2.sh &
```

预计耗时约 8 小时（RTX 5880）。日志输出到 `outputs/v2/tonight_v2_log.txt`。

脚本自动执行：训练 B5v2 → 训练 B6v2 → 正交性分析 → 生成质量评估 → 产出检查。

## 第二天检查

### 1. 确认跑完

```bash
tail -30 outputs/v2/tonight_v2_log.txt
```

看末尾的产出文件检查表，所有项应显示 `[OK]`。

### 2. 正交性是否改善

```bash
cat outputs/v2/b5v2_orthogonality_test.json
cat outputs/v2/b6v2_orthogonality_test.json
```

对比基线：

| 指标 | V1 B4（无 orth） | V1 B5（弱 orth） | V1 B6（cls 冲突） | **期望 B5v2** | **期望 B6v2** |
|---|---|---|---|---|---|
| mean_cosine ↓ | 0.122 | 0.086 | 0.133 | **< 0.05** | **< B5v2** |

B6v2 的 mean_cosine 应 **低于** B5v2 — 说明修复后 cls 不再破坏正交性。

### 3. 生成质量是否保持

在 `tonight_v2_log.txt` 中搜索 `[eval]` 行：

```bash
grep '\[eval\]' outputs/v2/tonight_v2_log.txt
```

test PPL 应接近 8.0（V1 B2 lora_only = 8.08），PPL > 10 说明 orth 过强伤害了生成。

### 4. cls 是否在学习（仅 B6v2）

```bash
grep 'cls=' outputs/v2/tonight_v2_log.txt | tail -20
```

训练末期 cls loss 应明显低于训练初期，说明 prefix 正在学习策略信号。

## 文件结构

```
configs/v2/
  b5v2_orth_strong.json
  b6v2_dest_rs_fixed.json
outputs/v2/
  tonight_v2_log.txt          ← 完整日志
  b5v2_orth_strong/
    metrics.json              ← loss/PPL 历史
    swap_samples_valid.jsonl  ← 生成样本
  b6v2_dest_rs_fixed/
    metrics.json
    swap_samples_valid.jsonl
  b5v2_orthogonality_test.json ← cosine 等正交指标
  b6v2_orthogonality_test.json
```
