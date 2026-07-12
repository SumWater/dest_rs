# 策略可控谈判生成：后续实验阶段性总报告

更新时间：2026-07-11

## 1. 当前研究问题

任务是根据谈判上下文、当前发言人 Profile 和目标策略生成回复。已有实验的核心矛盾是：

- B3 Prefix-only：策略控制较好，但 PPL 较差；
- B4 Prefix+LoRA：PPL 改善，但策略控制显著下降；
- B9 先训练 Prefix、再冻结 Prefix 训练 LoRA：部分缓解冲突，但没有恢复到 B3。

当前假设是：token-level LM loss 无法稳定区分“目标策略正确的回复”和“语言流畅但策略错误的回复”。后续主线因此是同上下文反事实数据与回复级 Sequence Ranking，而不是继续默认增加正交损失。

## 2. 冻结与可复现性状态

- 本机保存代码、配置、数据、generation 和 evaluation；
- checkpoint 与原训练环境保存在实验主机；
- 已采集实验主机 checkpoint SHA-256、代码/结果哈希、Python/CUDA/GPU 和依赖版本；
- 原环境：Python 3.10.20、PyTorch 2.12.0+cu126、Transformers 5.8.1、PEFT 0.19.1、RTX 5880 Ada；
- B3/B4/B9 seed 42/43/44 checkpoint 均存在并已登记哈希；
- Git 冻结标签尚未创建，但基准提交已记录为 `4c10480bbdba29172ddcc51cc9c6cb750da16fef`。

## 3. 数据审计

D1（旧类别平衡增强数据）正式划分：train/dev/test = 610/41/41 个 dialogue。

- 标注发言总数：8167；
- 真实发言：4615；增强发言：3552；
- 多标签发言：968（约 11.9%）；
- Profile 缺失：0；需求理由缺失：0；
- train/dev/test 的 `dialogue_id` 交叉：0；
- 增强训练 dialogue 与正式 test 重合：0；
- 完全重复文本组：108；
- 模型输入包含当前发言人的 Profile、物品优先级与需求理由。

结论：正式 dialogue-level 划分没有发现泄漏，但旧单标签 Judge 与数据的多标签属性不匹配；近重复、模板化和未标注策略混入仍需人工或语义审计。

## 4. B3/B4/B9 多标签重评估

三随机种子均值 ± 样本标准差：

| 实验 | Target Presence | Primary Accuracy | Macro-F1 | Off-target | Test PPL | 完全重复率 |
|---|---:|---:|---:|---:|---:|---:|
| B3 | 58.27 ± 4.06% | 52.84 ± 3.24% | 50.22 ± 3.01% | 0.74 ± 0.04 | 10.84 ± 0.09 | 12.10 ± 2.80% |
| B4 | 38.89 ± 1.70% | 35.19 ± 3.57% | 33.29 ± 2.37% | 0.83 ± 0.08 | 8.16 ± 0.16 | 33.33 ± 8.34% |
| B9 | 42.96 ± 1.92% | 39.26 ± 3.23% | 37.70 ± 2.18% | 0.79 ± 0.06 | 8.29 ± 0.13 | 24.57 ± 1.71% |

结论：B4 用策略控制和多样性换取 PPL；B9 相对 B4 有部分恢复且 PPL 代价很小，因此适合作为后续方法初始化，但它仍明显弱于 B3。

持续困难的策略包括 `uv-part`、`promote-coordination` 和 `other-need`。

## 5. 诊断一：Context/Response 策略可预测性

使用 D0 真实数据、固定 dialogue-level 划分和相同轻量多标签分类器：

| 输入 | 单标签 Accuracy | 多标签 Macro-F1 | Exact Match |
|---|---:|---:|---:|
| Context-only | 27.95% | 28.85% | 0.98% |
| Response-only | 71.62% | 51.70% | 43.28% |
| Context+Response | 69.87% | 46.71% | 48.20% |

结论：上下文不能很好地直接预测策略，不支持“模型主要依靠上下文绕过 Prefix”的强假设。回复中存在明显策略信号，但 `uv-part`、`no-need`、`showing-empathy` 和 `other-need` 的边界较弱。

## 6. 诊断二：B9 Profile 输入消融

seed=42、30 个相同 valid 上下文、9 个目标策略；P0/P1 使用相同 B9 checkpoint 和 greedy decoding，Judge 对两组均隐藏 Profile。

| 条件 | Target Presence | Primary Accuracy | Macro-F1 | Off-target |
|---|---:|---:|---:|---:|
| P0 无 Profile | 51.11% | 46.30% | 45.62% | 0.719 |
| P1 有 Profile | 41.11% | 38.89% | 37.09% | 0.774 |

Profile 使 18 条 Target Presence 从错变对，却使 45 条从对变错。退化主要集中在：

- `vouch-fair`：−29.29 pp；
- `other-need`：−21.01 pp；
- `promote-coordination`：−10.29 pp；
- `no-need`：−10.26 pp。

结论：当前瓶颈不是缺少 Profile。Profile 中显著的自我需求信息可能与目标 Prefix 竞争。但 Profile 对事实一致性仍然必要，后续应保留 Profile，并通过排序损失约束目标策略优先。

证据等级：单 seed 机制诊断，不作为跨种子最终结论。

## 7. 诊断三：显式策略 Verbalizer 上界

冻结 Base Qwen3-8B，不使用 Prefix/LoRA；同一批 30 个 valid 上下文，Judge 对全部条件隐藏 Profile。

| 条件 | Target Presence | Primary Accuracy | Macro-F1 | Off-target |
|---|---:|---:|---:|---:|
| V0 Context | 17.04% | 11.11% | 9.40% | 1.396 |
| V1 + Strategy Name | 36.67% | 27.04% | 24.76% | 1.115 |
| V2 + Strategy Definition | 78.89% | 72.59% | 65.78% | 0.611 |
| V3 + Profile + Definition | 70.74% | 63.70% | 56.82% | 0.830 |
| V4 + Profile + Definition + Example | 87.78% | 86.30% | 81.43% | 0.289 |

V2 相对 V1 的 Target Presence 提升 42.22 pp，说明明确行为定义远比策略名称有效。V3 再次显示 Profile 会干扰控制。V4 是最高提示上界，但完全重复率为 29.63%，存在示例模仿，不能作为干净方法基线。

结论：Base 模型能够理解并执行策略，瓶颈在 soft Prefix/LM-only 训练未稳定形成输出级策略映射，而不是模型能力不足。

## 8. 诊断四：B9 Top-K Pilot

seed=42，10 个对话轮次上下文 × 9 个策略，K=8，共 90 个任务、720 条候选：

| 指标 | Target Presence |
|---|---:|
| Top-1 | 43.33% |
| Oracle@4 | 65.56% |
| Oracle@8 | 75.56% |
| Judge Rerank@8 | 75.56% |

Oracle@8 − Top-1 = +32.22 pp，说明正确策略回复经常存在于 B9 的生成分布中，但默认排名较低，直接支持 Sequence Ranking。

- `promote-coordination`、`vouch-fair`：更接近排序问题；
- `other-need`、`uv-part`：Oracle@8 仍只有 40%/30%，还存在生成能力与数据覆盖问题。

限制：pilot 上下文主要来自同一 dialogue；Judge Rerank 使用同一个 Judge 选择并判定，因此是乐观上界。该结果只用于方法决策，不作为最终论文统计。

## 9. 综合判断

现有证据共同支持以下机制解释：

1. 类别不平衡不是唯一原因；
2. 上下文泄露不是主要瓶颈；
3. Base 模型具备较高显式策略执行上限；
4. Profile 会诱导默认的自我需求表达，与目标 Prefix 竞争；
5. B9 分布中经常存在正确候选，但 LM-only 目标没有把它排到前面；
6. `other-need`、`uv-part` 等类别还需要同上下文反事实正样本；
7. 下一步应优先使用反事实数据 + Sequence Ranking 优化 Prefix，不应继续默认增加正交约束。

## 10. 下一阶段执行顺序

1. 从非测试集选择 50 个真实上下文，每个上下文选择 3 个合理策略；
2. 生成约 150 条同上下文反事实回复；
3. 人工审核全部 150 条：目标策略、off-target、上下文、Profile、自然度、模板化和事实编造；
4. 构造 150～300 个同上下文 Preference Pair；
5. 使用 B9 checkpoint，冻结 Base 和 LoRA，只训练 Strategy Prefix；
6. 首轮只运行 M-01、M-02、M-03；
7. M-03 使用回复 token 的长度归一化序列 log-probability及 margin=0.3 的 Sequence Ranking；
8. 只有 M-03 多类别共同改善、类别塌缩减弱且 PPL 未明显恶化，才扩大数据并考虑正交消融、DPO 或 SimPO。

## 11. 相关报告

- `reports/data_audit.md`
- `reports/multilabel_baseline_summary.md`
- `reports/profile_ablation/summary.md`
- `reports/verbalizer_upper_bound/summary.md`
- `reports/b9_topk_pilot/summary.md`
- `reports/freeze_manifest.md`

