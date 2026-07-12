# B9 Top-K 诊断 Pilot

设置：B9 seed=42，10 个对话轮次上下文 × 9 个目标策略，K=8，temperature=0.8，top_p=0.9。共 90 个任务、720 条候选；Judge 解析失败 2 条。

| 指标 | Target Presence |
|---|---:|
| Top-1（greedy） | 43.33% |
| Oracle@4 | 65.56% |
| Oracle@8 | 75.56% |
| Judge Rerank@8 | 75.56% |

Oracle@8 − Top-1 = +32.22 pp，表明正确策略回复经常位于 B9 的采样分布中，但默认排序较低。这直接支持训练 Sequence Ranking。

Judge Rerank@8 使用同一个 Judge 的目标存在/主策略判断进行候选选择和结果判定，因此等于 Oracle@8；它是目标感知 Judge reranker 的乐观上界，不是独立无偏评估。后续若将 reranker 作为正式基线，需要用独立标注或交叉 Judge 评估。

## 每类 Oracle

| 策略 | Oracle@4 | Oracle@8 |
|---|---:|---:|
| elicit-pref | 100% | 100% |
| no-need | 100% | 100% |
| other-need | 20% | 40% |
| promote-coordination | 40% | 70% |
| self-need | 100% | 100% |
| showing-empathy | 80% | 80% |
| small-talk | 80% | 80% |
| uv-part | 20% | 30% |
| vouch-fair | 50% | 80% |

`other-need` 与 `uv-part` 在 K=8 下仍很弱，需要增加同上下文反事实正样本；`promote-coordination` 与 `vouch-fair` 的 Oracle 增益更像排序问题。

## 限制

前 10 个数据项是对话轮次而非 10 个独立 dialogue，主要集中在同一 dialogue_id，统计相关性较强。该实验仅作为是否继续 Sequence Ranking 的方向性 pilot，不作为论文最终置信区间或显著性结果。完整实验可在第一轮方法有效后补跑，并按 dialogue_id 均匀抽样。
