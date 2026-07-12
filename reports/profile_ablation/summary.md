# B9 Profile 输入消融（seed=42）

生成条件严格配对：相同 B9 checkpoint、30 个 valid 上下文、9 个目标策略、Prefix/LoRA 和 greedy decoding；唯一生成侧差异为是否输入当前发言人的 Profile。Judge 对 P0/P1 均移除 Profile，只观察相同类型的 dialogue history 与 candidate utterance。

| 条件 | Target Presence | Primary Accuracy | Macro-F1 | Off-target Count |
|---|---:|---:|---:|---:|
| P0：无 Profile | 51.11% | 46.30% | 45.62% | 0.719 |
| P1：有 Profile | 41.11% | 38.89% | 37.09% | 0.774 |
| P1 − P0 | -10.00 pp | -7.41 pp | -8.53 pp | +0.056 |

配对变化：

- Target Presence：P0 错/P1 对 18 条；P0 对/P1 错 45 条。
- Primary Accuracy：P0 错/P1 对 19 条；P0 对/P1 错 39 条。

## 每类 F1

| 策略 | P0 | P1 | P1 − P0 |
|---|---:|---:|---:|
| self-need | 34.53% | 29.35% | -5.18 pp |
| other-need | 30.77% | 9.76% | -21.01 pp |
| no-need | 76.92% | 66.67% | -10.26 pp |
| uv-part | 10.81% | 13.64% | +2.83 pp |
| vouch-fair | 50.91% | 21.62% | -29.29 pp |
| small-talk | 45.28% | 50.91% | +5.63 pp |
| elicit-pref | 67.61% | 59.15% | -8.45 pp |
| showing-empathy | 57.89% | 57.14% | -0.75 pp |
| promote-coordination | 35.82% | 25.53% | -10.29 pp |

## 解释边界

在 B9 seed=42 上，Profile 没有改善策略控制，反而把输出推向 Profile 中显著的自我需求信息，可能干扰目标 Prefix，尤其伤害 `other-need` 与 `vouch-fair`。这说明问题不是“缺少 Profile”这么简单。

该结果只覆盖一个 checkpoint seed 和 30 个上下文；同一上下文的 9 个策略输出也不是完全独立样本。因此它是机制诊断，不作为跨种子最终显著性结论。后续不应删除 Profile，而应在 Sequence Ranking 中显式约束模型利用 Profile 时仍服从目标策略。
