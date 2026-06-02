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

---

## 数据管线修复：multi_label_policy = duplicate（2026-06-02）

### 问题发现

分析 B8 策略控制结果时发现 5/9 个策略类别准确率为 0%，排查发现根因在数据管线：

`multi_label_policy: "drop"` 丢弃了所有多标签 utterance，造成三个问题：

#### 1. 30.9% 训练数据被浪费

CaSiNo 数据中策略标签天然多标签（一句话可以同时是 self-need + other-need），`drop` 策略直接丢弃这些样本：

| 策略 | 总标注 | 单标签保留 | 多标签丢弃 | **丢弃率** |
|---|---|---|---|---|
| other-need | 363 | 86 | 277 | **76.3%** |
| showing-empathy | 216 | 61 | 155 | **71.8%** |
| no-need | 166 | 56 | 110 | **66.3%** |
| uv-part | 122 | 46 | 76 | **62.3%** |
| self-need | 849 | 348 | 501 | **59.0%** |
| elicit-pref | 330 | 159 | 171 | 51.8% |
| promote-coordination | 500 | 239 | 261 | 52.2% |
| vouch-fair | 387 | 226 | 161 | 41.6% |
| small-talk | 924 | 689 | 235 | 25.4% |

**small-talk 只丢了 25%，其他策略丢了 50-76%。** 训练集被系统性偏向 small-talk（占 36%）。

#### 2. 训练集极端不均衡

```
small-talk            689  (36.1%)  ████████████████████
self-need             348  (18.2%)  ██████████
promote-coordination  239  (12.5%)  ███████
vouch-fair            226  (11.8%)  ██████
elicit-pref           159  (8.3%)   ████
other-need             86  (4.5%)   ██
showing-empathy        61  (3.2%)   █
no-need                56  (2.9%)   █
uv-part                46  (2.4%)   █
```

最多（small-talk: 689）和最少（uv-part: 46）相差 **15 倍**。

#### 3. 最 informative 的样本被丢弃

真实对话中策略天然组合出现：
- `other-need` + `self-need` 共现 111 次
- `promote-coordination` + `vouch-fair` 共现 58 次
- `self-need` + `small-talk` 共现 56 次

`drop` 把这些共现样本全部丢弃，模型学到的策略边界是人为割裂的，训练时从未见过策略组合，推理时却要区分它们——导致大量策略被混淆为 `self-need` 或 `small-talk`。

### 修复方案

**`multi_label_policy` 从 `"drop"` 改为 `"duplicate"`。**

当 utterance 有多个策略标签时，复制多份，每份用不同 primary_strategy。例如 `"我需要水，也希望你拿到柴"` 标注为 `[self-need, other-need]`，会产生两条训练样本：
- 同一条文本，primary_strategy = `self-need`
- 同一条文本，primary_strategy = `other-need`

这样 prefix 学到的是"同一句话可以服务于不同策略意图"，而非"不同策略必须产生不同的话"。

#### 数据量变化

| 策略 | drop 后 | duplicate 后 | 增幅 |
|---|---|---|---|
| other-need | 86 | 363 | **+322%** |
| showing-empathy | 61 | 216 | **+254%** |
| no-need | 56 | 166 | **+196%** |
| uv-part | 46 | 122 | **+165%** |
| 总训练样本 | 1,910 | ~3,616 | **+89%** |

#### 为什么不选其他方案

| 方案 | 判断 |
|---|---|
| `first` | 只取第一个标签，不解决数据浪费，只是换了种丢弃方式 |
| 4-class 合并 | 减少混淆但没解决数据利用率问题，可以后续叠加 |
| 加权 loss | 只缓解不均衡，不解决多标签被丢弃，且需调参 |

`duplicate` 是成本最低、信息保留最完整的修改——**只改一行配置**。

### B8-dup 实验

在 B8 架构上验证 duplicate 的效果：

```bash
bash run_b8_dup.sh
```

```
configs/b8_dup/
  b8_dup_p1_lora.json      # Phase 1: LoRA only, duplicate
  b8_dup_p2_prefix.json    # Phase 2: prefix on frozen LoRA, duplicate
outputs/b8_dup/
  p1_lora_only/
  p2_prefix_frozen_lora/
```

预期：低频策略（no-need, uv-part, other-need, showing-empathy）的逐类准确率应有明显提升，4-class 准确率应接近或超越 B3（0.622）。
