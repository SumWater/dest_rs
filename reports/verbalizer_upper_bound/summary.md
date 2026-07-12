# 显式策略 Verbalizer 上界（Base Qwen3-8B）

冻结基础模型，不使用 Prefix/LoRA；30 个相同 valid 上下文、每个上下文 9 个目标策略、greedy decoding。Judge 对所有条件隐藏 Profile。

| 条件 | Target Presence | Primary Accuracy | Macro-F1 | Off-target Count | 完全重复率 |
|---|---:|---:|---:|---:|---:|
| V0 Context | 17.04% | 11.11% | 9.40% | 1.396 | 99.26% |
| V1 + Strategy Name | 36.67% | 27.04% | 24.76% | 1.115 | 12.59% |
| V2 + Strategy Definition | 78.89% | 72.59% | 65.78% | 0.611 | 9.26% |
| V3 + Profile + Definition | 70.74% | 63.70% | 56.82% | 0.830 | 0.00% |
| V4 + Profile + Definition + Example | 87.78% | 86.30% | 81.43% | 0.289 | 29.63% |

V0 对同一上下文的九个目标策略没有收到任何目标信息，因此生成相同回复是预期行为；它只表示无控制条件的机会水平，不与其他条件比较语言多样性。

## 关键边际变化

- V2 − V1：Target Presence +42.22 pp，Macro-F1 +41.02 pp。策略名称远不如明确行为定义有效。
- V3 − V2：Target Presence −8.15 pp，Macro-F1 −8.96 pp，Off-target +0.219。Profile 再次干扰目标策略控制。
- V4 − V3：Target Presence +17.04 pp，Macro-F1 +24.61 pp，但完全重复率增加到 29.63%。

V4 与提供的单个示例有 4 条完全复制、5 条 token Jaccard ≥ 0.8、32 条 ≥ 0.6。因此 V4 是提示/模仿上界，不是干净的生成方法基线。

## 结论

数据上下文和 Base Qwen3-8B 足以支持较高策略控制；瓶颈不是模型无法理解策略，而是 soft Prefix 与 LM-only 训练没有稳定地把策略定义映射到输出。V2 已显著超过 B3/B9，支持继续采用输出级 Sequence Ranking，而不是继续增加正交约束。

Profile 对事实一致性有用，但会把生成拉向显著的自我需求内容。后续保留 Profile，同时用反事实偏好对约束“目标策略优先于 Profile 诱导的默认策略”。
