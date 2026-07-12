# 策略可控谈判对话生成：后续实验完整执行方案

## 0. 文档目的

本文档用于直接交给 Codex 执行。目标是在保留现有有效实验的基础上，判断当前瓶颈究竟来自：

1. 数据标签和任务定义不匹配；
2. 输入信息不足；
3. 类别不平衡；
4. LoRA 对 Prefix 控制的覆盖；
5. 标准 LM loss 缺乏输出级策略监督。

最终目标是形成一篇可以投稿的小论文，并作为硕士毕业论文的核心方法章与实验章。

---

# 1. 核心研究问题

任务定义：

```text
输入：谈判对话上下文 + 当前发言人 profile + 目标策略
输出：符合目标策略、上下文合理、语言流畅的回复
```

当前主要矛盾：

```text
Prefix-only：
策略控制相对较好，但 PPL 较差。

Prefix + LoRA：
PPL 改善，但策略控制显著下降。
```

新的研究假设：

```text
仅依靠标准 token-level LM loss，
无法稳定地区分“策略正确的回复”和“策略错误但流畅的回复”。

原始观察性数据也没有提供：
同一个上下文在不同策略干预下应该产生什么不同回复。
```

因此，后续主线为：

```text
真实数据审计
→ 修正评估
→ 构造同上下文多策略反事实数据
→ 冻结 Base 和 LoRA
→ 使用序列级策略排序损失优化 Prefix
→ 做完整消融和人工评估
```

---

# 2. 旧实验是否保留

## 2.1 必须保留的有效实验

| 实验 | 状态 | 后续用途 |
|---|---|---|
| B3 Prefix-only | 有效 | 控制能力基线 |
| B4 Prefix+LoRA 联合训练 | 有效 | 流畅度基线和冲突基线 |
| B9 先训 Prefix、再冻 Prefix 训 LoRA | 有效 | 证明问题不是单纯梯度覆盖；后续主方法初始化 |
| S1 Reverse Curriculum | 有效负结果 | 证明让 Prefix 适应固定 LoRA 空间仍不够 |
| S2_qk v3 | 有效负结果 | 证明 QK 参数正交不能恢复最终输出控制 |
| Gold judge calibration | 有效诊断 | 证明单标签 judge 与原始标签存在明显不一致 |

这些实验不能删除。它们共同构成论文中的“问题发现和机制诊断”。

## 2.2 暂不作为结论的实验

| 实验 | 状态 | 处理方式 |
|---|---|---|
| S3 Attention Analysis | 无有效结论 | 不是主线，不继续投入；有时间再做可视化 |
| 修复后的 S4 | 尚未完整重跑 | 作为低成本 decoding baseline 小规模重跑 |
| hidden-state 正交、辅助分类、梯度路由 | 视实现和日志决定 | 若代码、loss 和日志可验证，则作为负结果；否则只放附录或不使用 |

## 2.3 作废实验

| 实验 | 原因 |
|---|---|
| 旧版 S4 | temperature=0.0 时发生除零，生成结果无效 |
| 旧版未实际接入的 S2 | 4-bit base weight 处理错误，正交 loss 被全部 skip |

作废的是对应的“结果”，不是整个思路。论文中不要引用这些错误结果。

---

# 3. 之前的数据增强如何处理

之前通过扩充不同策略的数据，使各类别数量平衡，这部分不作废。

将其定义为：

```text
D1：Class-Balanced Augmented Dataset
```

它验证的是：

```text
类别不平衡是否导致模型偏向 self-need、small-talk 等高频策略。
```

但它不能解决以下问题：

1. 同一上下文通常只有一个实际回复；
2. 模型没有看到“同一上下文切换策略后，回复应该怎样变化”；
3. 生成样本可能仍然存在多策略混合；
4. 不同类别样本可能来自完全不同的上下文分布；
5. LoRA 仍可以根据上下文统计规律预测高频、低风险回复；
6. 独立扩充样本不构成严格的正负策略对。

所以，旧数据增强应作为一个正式数据版本和消融基线保留，而不是直接丢弃。

需要对旧增强数据做审计：

```text
- 每类原始样本数
- 每类增强样本数
- 增强来源
- 是否由同一个生成模型产生
- 是否包含重复或近重复
- 是否包含策略名或模板线索
- 是否改变了人物事实
- 是否严格符合目标策略
- 是否混入其他策略
- 是否和测试集上下文重复
```

---

# 4. 工程准备工作

## 4.1 冻结当前代码和结果

Codex 执行：

```text
1. 创建 git tag：
   before_counterfactual_strategy_experiments

2. 备份：
   - 当前训练配置
   - 当前评估脚本
   - B3/B4/B9 checkpoint
   - S1/S2_qk checkpoint
   - 所有生成结果 JSONL
   - 所有训练日志
   - 当前数据集版本
   - 当前 judge prompt

3. 生成环境快照：
   - pip freeze
   - CUDA version
   - PyTorch version
   - transformers version
   - peft version
   - GPU 型号
```

建议目录：

```text
project/
├── configs/
├── data/
│   ├── raw/
│   ├── old_augmented/
│   ├── processed/
│   ├── counterfactual/
│   └── testsets/
├── scripts/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── generation/
├── checkpoints/
├── generations/
├── reports/
├── logs/
└── experiment_registry.csv
```

## 4.2 建立实验注册表

创建：

```text
experiment_registry.csv
```

字段：

```text
experiment_id
date
git_commit
data_version
model_init
trainable_modules
losses
seed
config_path
checkpoint_path
generation_path
evaluation_path
notes
status
```

所有实验必须通过 registry 追踪，禁止仅靠文件夹名字判断。

## 4.3 随机种子策略

探索阶段：

```text
seed = 42
```

最终主结果：

```text
seed = 42, 43, 44
```

最终报告均值和标准差。低成本诊断实验可以先只跑一个 seed。

---

# 5. 第一阶段：完整数据审计

## 5.1 统一数据格式

所有数据转换为 JSONL，每条至少包含：

```json
{
  "sample_id": "unique_id",
  "dialogue_id": "dialogue_id",
  "turn_id": 10,
  "context": ["utterance 1", "utterance 2"],
  "speaker": "A",
  "speaker_profile": {
    "food_priority": "high",
    "water_priority": "medium",
    "firewood_priority": "low",
    "reasons": []
  },
  "original_response": "gold utterance",
  "strategy_labels": ["self-need", "elicit-pref"],
  "primary_strategy": "self-need",
  "source": "real",
  "is_augmented": false
}
```

如果当前数据只有单标签，也要保留：

```text
strategy_labels
```

字段，避免后续继续被单标签格式限制。

## 5.2 数据划分检查

必须按：

```text
dialogue_id
```

划分 train/dev/test，不能按 utterance 随机划分。

检查：

```text
train dialogue_id ∩ dev dialogue_id = empty
train dialogue_id ∩ test dialogue_id = empty
dev dialogue_id ∩ test dialogue_id = empty
```

同时检查旧增强数据是否基于 test 上下文生成。如果是，必须从训练数据中移除。

## 5.3 输出数据审计报告

实现：

```text
scripts/audit_dataset.py
```

输出：

```text
reports/data_audit.json
reports/data_audit.md
reports/strategy_counts.csv
reports/strategy_cooccurrence.csv
reports/duplicate_samples.csv
reports/dialogue_split_leakage.csv
```

统计内容：

1. 总样本数；
2. 每类策略数量；
3. 单标签和多标签比例；
4. 策略共现矩阵；
5. 每类平均回复长度；
6. 每类对话轮次分布；
7. 每类 train/dev/test 数量；
8. profile 缺失比例；
9. 完全重复；
10. 近重复；
11. 类别间模板重复；
12. 旧增强数据与真实数据的文本相似度；
13. 增强数据是否存在固定开头、固定句式；
14. 各策略中包含其他策略关键词或行为的比例。

近重复可使用：

```text
normalized edit similarity
或
sentence embedding cosine similarity
```

第一版可使用简单的字符或 token Jaccard，后续再升级。

---

# 6. 建立数据版本

## D0：Raw Original

```text
当前原始训练数据。
```

用途：复现原始结果。

## D1：Old Balanced Augmentation

```text
之前已经做过的类别平衡增强数据。
```

用途：判断单纯类别平衡是否有效。

## D2：Atomic Clean

筛选规则：

```text
1. 优先保留单标签样本；
2. 删除策略语义明显不清晰的样本；
3. 删除 profile 与回复矛盾的样本；
4. 删除模板化严重和重复样本；
5. 保留原始真实上下文；
6. 按 dialogue_id 划分。
```

第一轮可以先构建 Clean-6：

```text
small-talk
self-need
other-need
no-need
uv-part
elicit-pref
```

暂时排除边界较模糊的：

```text
showing-empathy
promote-coordination
vouch-fair
```

Clean-6 只用于诊断，不代表最终任务永久变成 6 类。

## D3：Atomic Balanced + Profile

在 D2 基础上：

```text
1. 类别均衡；
2. 输入当前发言人 profile；
3. 输入物品优先级和需求理由；
4. 保持真实回复。
```

## D4：Counterfactual Synthetic

同一真实上下文下，生成 2～4 个可行策略回复。

特点：

```text
same context
same profile
same speaker
different feasible strategy
different strategy-specific response
```

## D5：Real + Counterfactual Mixed

训练主数据：

```text
真实数据
+
经过审核的反事实正样本
+
同上下文策略 preference pairs
+
B9 hard negatives
```

---

# 7. 第二阶段：低成本诊断实验

先做这些实验，再决定是否大规模生成数据。

## 7.1 Context-only Strategy Prediction

实现：

```text
scripts/train_strategy_classifier.py
```

三个版本：

```text
C1：context → strategy
C2：response → strategy
C3：context + response → strategy
```

指标：

```text
accuracy
macro-F1
per-class precision/recall/F1
confusion matrix
```

解释：

```text
如果 C1 很高：
说明上下文和策略高度相关，LoRA 可能学会绕过 Prefix。

如果 C2 很低：
说明标签不能从回复文本中稳定识别，数据边界有问题。

如果 C3 仍然不高：
说明标签噪声或 judge 定义存在严重问题。
```

## 7.2 Profile Ablation

固定同一模型和数据，比较：

```text
P0：context + strategy
P1：profile + context + strategy
```

优先观察：

```text
self-need
other-need
no-need
uv-part
vouch-fair
```

## 7.3 Text Verbalizer Upper Bound

不使用 soft Prefix，显式输入策略定义。

版本：

```text
V0：context
V1：context + strategy name
V2：context + strategy definition
V3：profile + context + strategy definition
V4：profile + context + strategy definition + one example
```

目的：

```text
判断数据和上下文是否足以支持策略控制。
```

如果 V3/V4 也很差，说明数据、输入或评估存在根本问题，不应立即扩大 Prefix 训练。

## 7.4 Oracle@K 与 Rerank@K

对 B9 每个样本生成：

```text
K = 4, 8
```

计算：

```text
Top-1 Target Presence
Oracle@4
Oracle@8
Judge Rerank@4
Judge Rerank@8
```

采样配置至少保存：

```text
temperature
top_p
top_k
max_new_tokens
repetition_penalty
seed
```

解释：

```text
Top-1 低，Oracle@8 高：
正确策略在分布中，但排名低。适合 ranking/reranking。

Oracle@8 也低：
模型基本生成不出目标策略。先补 profile、verbalizer 和反事实数据。

Oracle@8 高，Rerank@8 低：
评估器或 judge 有问题。
```

---

# 8. 第三阶段：反事实数据生成

## 8.1 第一批规模

不要直接生成数千条。

第一批：

```text
50 个真实上下文
每个上下文 3 个可行策略
共约 150 条反事实回复
```

通过后扩大为：

```text
100～200 个上下文
每个上下文 3～4 个策略
共约 300～800 条回复
```

## 8.2 上下文选择

优先选择：

```text
1. profile 完整；
2. 对话历史足以理解当前谈判状态；
3. 至少存在 2 个合理策略；
4. 不处于对话完全结束状态；
5. 不包含严重歧义；
6. 不与最终测试集重叠。
```

## 8.3 策略可行性判断

每个上下文先生成：

```json
{
  "context_id": "xxx",
  "feasible_strategies": [
    "self-need",
    "elicit-pref",
    "promote-coordination"
  ],
  "infeasible_strategies": [
    "small-talk",
    "no-need"
  ],
  "reasoning_summary": {
    "self-need": "speaker profile supports expressing own need",
    "no-need": "conflicts with high-priority item"
  }
}
```

注意：

```text
不要强制每个上下文覆盖 9 个策略。
```

可行性结果应支持人工修改。

## 8.4 生成输入模板

生成器输入必须包含：

```text
- 当前发言人 profile
- 物品优先级
- 需求理由
- 对方已表达的需求
- 对话历史
- 目标策略名称
- 目标策略定义
- 易混淆策略定义
- 禁止混入的行为
- 长度限制
- 事实一致性要求
```

模板示例：

```text
You are generating one natural negotiation reply.

Speaker profile:
{profile}

Dialogue context:
{context}

Target strategy:
{target_strategy}

Target strategy definition:
{target_definition}

Do not mainly express the following strategies:
{confusing_strategy_definitions}

Requirements:
1. The reply must clearly express the target strategy.
2. Do not mention the strategy name.
3. Do not invent facts.
4. Do not contradict the speaker profile.
5. Do not produce a full allocation unless the target strategy requires it.
6. Keep the reply between 15 and 35 English words.
7. Output only the reply.
```

## 8.5 同一生成器原则

同一组反事实回复应尽量使用：

```text
同一个生成模型
同一套 system prompt
同一套 temperature/top_p
同一批次生成
```

避免 chosen 和 rejected 因模型风格不同产生捷径。

## 8.6 数据验证

实现：

```text
scripts/validate_counterfactual_data.py
```

规则过滤：

```text
- 空回复
- 超长或过短
- 重复文本
- 含策略名称
- 明显模板开头
- profile 矛盾
- 非法格式
- 和原始回复近重复
```

Judge 过滤字段：

```json
{
  "target_strategy_present": true,
  "primary_strategy": "self-need",
  "off_target_strategies": [],
  "context_consistent": true,
  "profile_consistent": true,
  "naturalness": 4,
  "specificity": 4,
  "accept": true
}
```

第一批必须人工审核全部 150 条。

扩大后：

```text
随机人工审核 10%～20%
低一致率类别提高抽检比例
```

---

# 9. Preference Pair 构造

每条 pair：

```json
{
  "pair_id": "pair_xxx",
  "context_id": "ctx_xxx",
  "profile": {},
  "context": [],
  "target_strategy": "self-need",
  "chosen": "reply expressing self-need",
  "rejected": "fluent reply expressing other-need",
  "negative_type": "counterfactual_confusion",
  "chosen_source": "synthetic_same_generator",
  "rejected_source": "synthetic_same_generator"
}
```

## 9.1 负样本来源

混合比例建议：

```text
50%：
同一上下文反事实组中的易混淆策略回复

30%：
B9 在目标 Prefix 下生成的高概率错误回复

20%：
最小策略改写得到的 hard negative
```

## 9.2 易混淆策略图

```text
self-need ↔ other-need
no-need ↔ uv-part
showing-empathy ↔ small-talk
promote-coordination ↔ vouch-fair
elicit-pref ↔ small-talk / promote-coordination
```

## 9.3 Pair 质量要求

chosen 和 rejected 应满足：

```text
- 上下文相同
- profile 相同
- 语言质量相近
- 长度尽量接近
- 不使用明显不同的模板
- 核心差异是策略
```

---

# 10. 第四阶段：基线重评估

## 10.1 不一定需要重新训练旧模型

如果 B3/B4/B9 的生成结果仍保存：

```text
先用新评估器重新评估旧 generations。
```

只有在以下情况才重训：

```text
1. 数据划分存在泄漏；
2. 旧训练数据版本无法确认；
3. 旧代码存在实现错误；
4. profile 输入发生变化；
5. 旧 checkpoint 无法正确加载；
6. 需要三随机种子最终结果。
```

## 10.2 必须重新计算的指标

```text
Target Strategy Presence
Primary Strategy Accuracy
Macro-F1
Per-class F1
Off-target Strategy Count
Strategy Purity
Prediction Distribution
PPL
Repetition Rate
Distinct-1/2
Average Length
```

---

# 11. 第五阶段：主训练方法

## 11.1 初始化

使用：

```text
B9 checkpoint
```

冻结：

```text
Base model
LoRA
```

只训练：

```text
Strategy Prefix
```

第一版不加入新的 LoRA，不重新联合训练。

## 11.2 序列得分

长度归一化：

```text
S(y | x, s)
=
sum(log p(y_t | x, s, y_<t)) / number_of_valid_response_tokens
```

注意：

```text
只统计 response token；
不统计 prompt、context、profile token；
padding token 不参与；
chosen 和 rejected 分别 forward；
必须返回每个样本的 sequence score。
```

## 11.3 损失函数

真实数据 LM：

```text
L_real_lm
```

反事实正样本 LM：

```text
L_syn_lm
```

序列级 ranking：

```text
L_rank =
max(0, margin + S(rejected) - S(chosen))
```

总损失：

```text
L =
lambda_real * L_real_lm
+ lambda_syn * L_syn_lm
+ lambda_rank * L_rank
```

第一轮：

```text
lambda_real = 1.0
lambda_syn = 0.5
lambda_rank = 1.0
margin = 0.3
```

这些只是起点，必须配置化。

## 11.4 Batch 组成

第一版：

```text
50% 真实样本
25% 合成反事实正样本
25% preference pairs
```

也可以实现两阶段训练：

```text
Stage A：
真实 + 合成正样本 LM

Stage B：
真实 LM + ranking
```

两种方式都保留配置入口。

## 11.5 训练日志

每 N step 记录：

```text
train_total_loss
train_real_lm
train_syn_lm
train_rank_loss
rank_active_ratio
chosen_score
rejected_score
score_margin
prefix_grad_norm
learning_rate
```

重要诊断：

```text
rank_active_ratio 长期接近 0：
负样本太简单或 margin 太小。

chosen_score 长期低于 rejected_score：
实现错误或数据标签错误。

PPL 快速恶化：
lambda_rank 太大或缺少 LM 锚定。
```

---

# 12. 主实验矩阵

## 12.1 数据诊断实验

| ID | 数据 | 模型 | 目的 |
|---|---|---|---|
| DEXP-01 | D0 | B3 | 原始 Prefix-only |
| DEXP-02 | D0 | B9 | 原始冲突 |
| DEXP-03 | D1 | B3 | 旧类别平衡对 Prefix-only 的影响 |
| DEXP-04 | D1 | B9 | 旧类别平衡能否缓解 LoRA 覆盖 |
| DEXP-05 | D2 Clean-6 | B3 | 干净标签环境控制上限 |
| DEXP-06 | D2 Clean-6 | B9 | 干净标签下冲突是否仍存在 |
| DEXP-07 | D3 + profile | B3 | profile 对控制的影响 |
| DEXP-08 | D3 + profile | B9 | profile 是否缓解冲突 |

## 12.2 生成和诊断

| ID | 方法 | 目的 |
|---|---|---|
| GEN-01 | B9 Top-1 | 当前生成基线 |
| GEN-02 | B9 Oracle@4 | 候选空间诊断 |
| GEN-03 | B9 Oracle@8 | 候选空间诊断 |
| GEN-04 | B9 Judge Rerank@8 | 无训练输出级 baseline |
| GEN-05 | Text Verbalizer | 显式控制上界 |

## 12.3 主方法

| ID | 数据/损失 | 说明 |
|---|---|---|
| M-01 | D4 synthetic + LM | 仅合成 SFT |
| M-02 | D5 real+synthetic + LM | 真实和合成混合 |
| M-03 | D5 + LM + ranking | 第一主方法 |
| M-04 | M-03 + hard negatives | 验证模型错误驱动负样本 |
| M-05 | M-04 + Prefix orthogonality | 可选，不作为默认主方法 |
| M-06 | M-04 with random negatives | 负样本难度消融 |
| M-07 | M-04 without real LM | 流畅度锚定消融 |
| M-08 | M-04 without synthetic LM | 合成正样本 LM 消融 |
| M-09 | DPO/SimPO version | 后续方法对比，不是第一优先级 |

## 12.4 S4 baseline

只先在 30～50 条上测试：

```text
alpha = 0.25, 0.5, 1.0
temperature = greedy or 0.7
```

检查无数值异常后再决定是否完整评估。

---

# 13. 评估协议

## 13.1 自动策略评估必须改为多标签

Judge 输出：

```json
{
  "present_strategies": ["self-need", "promote-coordination"],
  "target_present": true,
  "primary_strategy": "self-need",
  "confidence": 0.85
}
```

不要再只允许九选一。

## 13.2 主要指标

### Target Strategy Presence

```text
目标策略是否出现在回复中。
```

### Primary Strategy Accuracy

```text
主要策略是否为目标策略。
```

### Macro-F1

对多标签或 primary label 分别计算，必须明确版本。

### Strategy Purity

```text
目标策略存在，但混入多少非目标策略。
```

可定义：

```text
Purity = 1 - off_target_count / 8
```

### Counterfactual Separation

同一上下文分别使用不同 Prefix 生成回复。

构造矩阵：

```text
M[i][j] = 目标为策略 i 时，生成回复中策略 j 的得分
```

报告：

```text
diagonal mean
off-diagonal mean
diagonal margin
```

### 流畅度和退化

```text
PPL
repetition rate
distinct-1
distinct-2
average length
invalid generation rate
```

## 13.3 人工评估

最终主模型至少抽取：

```text
100～200 条生成结果
```

建议维度：

```text
1. Target strategy correctness
2. Context relevance
3. Profile consistency
4. Naturalness
5. Overall usability
```

最好双人标注，并报告一致率。

若只能一人标注，必须明确写为人工审查而非严格 inter-annotator evaluation。

## 13.4 测试集

至少包括：

```text
T_real：
原始真实测试上下文

T_cf_human：
人工审核的同上下文多策略反事实测试集

T_hard：
易混淆策略对测试集
```

T_cf_human 和 T_hard 不能参与训练。

---

# 14. 实验决策门槛

## 14.1 第一批 150 条反事实数据是否扩展

满足以下大部分条件才扩展：

```text
1. M-03 明显高于 B9；
2. Macro-F1 至少有多个类别共同改善；
3. self-need/small-talk 预测集中度下降；
4. PPL 没有明显退化；
5. 人工检查没有明显模板化；
6. 同一上下文切换 Prefix 后，回复行为明显变化；
7. ranking active ratio 正常；
8. chosen score 稳定高于 rejected score。
```

如果完全无改善，先检查：

```text
- response token mask
- sequence score 计算
- Prefix 是否真的 trainable
- optimizer 是否包含 Prefix
- chosen/rejected 是否放反
- 数据标签是否错误
- profile 是否缺失
- judge 是否不可靠
```

不要直接扩大到几千条。

## 14.2 是否加入正交约束

只有在 M-04 已有效后再测试 M-05。

若：

```text
M-05 > M-04
```

可将正交作为辅助贡献。

若：

```text
M-05 <= M-04
```

不要强行保留正交约束，作为负结果即可。

## 14.3 是否做 DPO

只有在以下条件满足后：

```text
1. preference pair 质量已验证；
2. sequence ranking 已有效；
3. judge 噪声可控；
4. 有稳定的 chosen/rejected 数据。
```

再实现 DPO/SimPO。

---

# 15. Codex 需要实现的脚本

## 数据

```text
scripts/audit_dataset.py
scripts/build_atomic_dataset.py
scripts/build_balanced_dataset.py
scripts/add_profile_fields.py
scripts/check_split_leakage.py
scripts/find_duplicates.py
scripts/select_counterfactual_contexts.py
scripts/generate_feasible_strategies.py
scripts/generate_counterfactual_responses.py
scripts/validate_counterfactual_data.py
scripts/build_preference_pairs.py
scripts/mine_b9_hard_negatives.py
```

## 诊断

```text
scripts/train_strategy_classifier.py
scripts/eval_profile_ablation.py
scripts/eval_verbalizer_upper_bound.py
scripts/generate_oracle_candidates.py
scripts/eval_oracle_rerank.py
```

## 训练

```text
scripts/train_prefix_lm.py
scripts/train_prefix_sequence_ranking.py
scripts/train_prefix_dpo.py
```

## 评估

```text
scripts/eval_multilabel_strategy.py
scripts/eval_generation_quality.py
scripts/eval_counterfactual_separation.py
scripts/eval_ppl.py
scripts/build_human_eval_sheet.py
scripts/summarize_experiments.py
```

---

# 16. 代码实现要求

## 16.1 所有配置化

禁止硬编码：

```text
data path
checkpoint path
strategy map
prefix length
margin
loss weights
generation parameters
judge provider
seed
```

使用 YAML。

## 16.2 可恢复训练

保存：

```text
model state
optimizer state
scheduler state
global step
random states
best metric
```

## 16.3 严格参数检查

训练启动时输出：

```text
all trainable parameter names
number of trainable parameters
Prefix parameter count
LoRA parameter count
Base parameter count
```

在主方法中必须断言：

```text
LoRA trainable count == 0
Base trainable count == 0
Prefix trainable count > 0
```

## 16.4 数值安全

生成和训练均检查：

```text
NaN
Inf
temperature <= 0
empty response
all-masked response
zero response token count
```

greedy decoding 不要走 temperature 除法。

## 16.5 输出可复现

每个 generation JSONL 保存：

```json
{
  "experiment_id": "M-03",
  "sample_id": "xxx",
  "seed": 42,
  "context": [],
  "profile": {},
  "target_strategy": "self-need",
  "generated_response": "...",
  "generation_config": {},
  "checkpoint": "...",
  "git_commit": "..."
}
```

---

# 17. 推荐执行顺序

## P0：必须先做

```text
1. 冻结代码和环境
2. 数据审计
3. 检查 dialogue split 泄漏
4. 审计旧类别平衡增强数据
5. 重新评估 B3/B4/B9 的多标签指标
6. Profile ablation
7. Text verbalizer upper bound
8. B9 Oracle@8
```

## P1：最小反事实实验

```text
9. 选 50 个上下文
10. 为每个上下文选择 3 个可行策略
11. 生成约 150 条回复
12. 人工审核全部 150 条
13. 构造 150～300 个 preference pairs
14. 训练 M-01、M-02、M-03
15. 与 B3、B9 比较
```

## P2：验证有效后扩大

```text
16. 扩展到 300～800 条反事实回复
17. 加入 B9 hard negatives
18. 跑 M-04
19. 三随机种子
20. 人工评估
21. 构建 hard-confusion test
```

## P3：论文增强实验

```text
22. 正交辅助项 M-05
23. DPO/SimPO 对比
24. 修复后的 S4 baseline
25. OOD 或跨模型验证
26. 效率和显存分析
```

---

# 18. 最终论文实验故事

论文应按以下逻辑组织：

```text
1. 发现 Prefix-only 与 Prefix+LoRA 存在控制-流畅度冲突；
2. B9 证明不是单纯训练时梯度覆盖；
3. S1、S2_qk 证明仅修复内部通道和参数空间不够；
4. 数据审计证明原始观察性数据缺乏策略干预监督；
5. 类别平衡增强只能解决频率问题，不能解决同上下文策略区分；
6. 构造上下文锚定的多策略反事实数据；
7. 通过序列级 ranking 直接优化输出行为；
8. 在保持 PPL 的同时提高策略控制；
9. 使用多标签、反事实和人工评估验证效果。
```

---

# 19. 最终结论

旧实验不作废，应分为：

```text
有效基线
有效负结果
实现错误导致的无效结果
尚无结论的探索实验
```

之前的类别平衡增强也不作废，保留为 D1。它回答：

```text
类别频率是否导致策略偏置。
```

新反事实数据回答：

```text
同一个上下文切换目标策略时，
模型是否能产生功能上不同的回复。
```

两者不是重复关系，而是不同层次的数据干预。

后续最重要的实验不是继续堆正交约束，而是：

```text
先审计旧增强数据
→ 跑 Oracle@8 和 verbalizer 上界
→ 生成 150 条高质量反事实回复
→ 使用 LM + sequence ranking 只训练 Prefix
→ 验证有效后再扩大。
```
