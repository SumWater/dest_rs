# 今晚实验计划
  ./run_all_tonight.sh 2>&1 | tee outputs/tonight_log.txt

  如果要用 LLM Judge，运行前先加一行：

  export DEEPSEEK_API_KEY="你的key"

  不加的话脚本会自动跳过那一步，不影响其他实验。第二天看 outputs/tonight_log.txt 末尾的文件检查表就知道哪些跑成功了。
## 当前状态

已完成的实验（followup 目录下，有完整 metrics + swap samples）：
- B4 × 3 seeds (42/43/44)
- B5 × 3 seeds (42/43/44)
- B5 lambda 消融 (0.01 / 0.05 / 0.10)
- B5 alpha 消融 (local_only / local_global / global_only)
- B4 vs B5 正交性分析 (seed42, test split)

**缺失的关键数据：**

| 缺失项 | 影响 |
|---|---|
| B2 (LoRA-only) 训练结果 | 消融表不完整，无法证明"双路径优于单路径" |
| B3 (Prefix-only) 训练结果 | 同上 |
| B6 (完整 DEST-RS) 训练结果 | 无法展示完整方法的效果，论文核心方法无数据 |
| Strategy Accuracy / Macro-F1 | 无法定量证明策略可控性——论文最核心的贡献 |
| B6 正交性分析 | 无法比较分类监督对表示分离的影响 |
| LLM Judge 标注 | judge 表格已生成但未标注 |

---

## 实验脚本说明（run_all_tonight.sh）

### 阶段 1：补齐核心消融实验（约 3-5 小时）

**运行内容：** B2 (lora_only) → B3 (prefix_only) → B6 (dest_rs)

**为什么要跑：**
- B4 和 B5 已有 followup/b4_seed_42 和 followup/b5_seed_42 的结果，
  配置与主实验完全一致，可以直接复用，不需要再跑。
- B2 和 B3 是单分支基线，用于证明 Prefix+LoRA 双路径组合优于任一单路径。
- B6 是完整的 DEST-RS 方法（正交约束 + 分类监督），是论文提出的方法，
  必须有结果才能写论文。
- 三者跑完后，配合已有的 B4/B5 数据，消融表就完整了：
  B2 → B3 → B4 → B5 → B6，逐步叠加组件。

### 阶段 2：正交性分析（约 10 分钟）

**运行内容：** 对 B6 checkpoint 计算 prefix/LoRA 表示增量的 cosine 和 Frobenius 指标。

**为什么要跑：**
- 已有 B4 和 B5 的正交性数据，加上 B6 可以展示分类监督是否进一步改善表示分离。
- 这是论文中"正交约束有效性"部分的定量证据。

### 阶段 3：策略可控性评估（约 10-15 分钟）

**运行内容：**
1. 在 CaSiNo 训练集上训练一个轻量 BoW 策略分类器（strategy evaluator）。
2. 用该分类器评估 B2/B3/B4/B5/B6 各自 swap samples 中的生成文本，
   计算 Strategy Accuracy 和 Macro-F1。

**为什么要跑：**
- 这是论文最核心的评估指标。你的论文要证明"切换 prefix 能控制生成策略"，
  必须有一个独立的分类器来定量验证。
- 仅靠 perplexity 无法说明策略可控性，审稿人一定会问。

### 阶段 4：LLM Judge 标注（约 15-30 分钟，需要 API Key）

**运行内容：** 调用 DeepSeek API 自动标注 B4 vs B5 的 judge 表格。

**为什么要跑：**
- judge 表格已经生成（outputs/judge_b4_vs_b5_seed42.csv），但 b4_match/b5_match
  等列全部为空。
- LLM Judge 可以作为"准人工评估"，补充自动指标的不足。
- 如果没有 DEEPSEEK_API_KEY，这一步会自动跳过，不影响前面的实验。

---

## 跑完之后

所有结果写入 outputs/ 目录。脚本结束后会打印汇总信息。
拿到结果后可以整理论文中的以下表格：

1. **Table: 主消融实验** — B2/B3/B4/B5/B6 的 valid_loss、test_ppl、Strategy Acc、Macro-F1
2. **Table: 正交性分析** — B4/B5/B6 的 mean_cosine、cosine²、prefix_delta_norm
3. **Table: Lambda 消融** — λ_orth = 0.01/0.05/0.10
4. **Table: Alpha 消融** — local_only / mixed / global_only
5. **Table: 多 seed 稳定性** — B4/B5 × 3 seeds 的均值和标准差
6. **Table: LLM Judge** — B4 vs B5 策略匹配率和胜负统计
