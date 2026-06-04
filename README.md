# DeSTRS：面向谈判对话策略可控生成的混合 PEFT 方法

DeSTRS 的核心思想：使用 **strategy-specific prefix bank** 承担高层策略 steering，使用 **shared LoRA branch** 承担共享生成适配，并通过 **局部-全局正交约束** 与 **策略分类辅助监督** 降低两类适配路径之间的干扰，从而提升策略可控性和生成质量。

## 项目结构

```
configs/                 # 实验配置（每个实验一个 JSON）
src/
  config.py              # TrainConfig 数据类，含 need_dir/other_dir 路径派生
  casino_dataset.py      # 数据加载、prompt 构建、Dataset/Collator
  modeling.py            # HybridStrategyModel、LoRA/Prefix 构建
  losses.py              # 损失函数（gen + orth + cls）
  evaluate.py            # 评估与 swap sample 生成
scripts/
  generate_augmented_data.py     # 策略驱动对话数据生成（Qwen3.5-9B）
  resplit_casino.py              # CaSiNo 数据集重划分（分层抽样）
  analyze_orthogonality.py       # Prefix/LoRA 增量正交性分析
  evaluate_generations.py        # distinct-n、平均长度、重复率
  evaluate_strategy_control.py   # Strategy Accuracy / Macro-F1
  evaluate_strategy_control_llm.py  # LLM-based 策略分类评估
train.py                 # 训练入口
run_all.sh               # 一键运行全部实验
```

## 输出目录结构

训练产物按用途分流，方便 git 管理：

```
output/
├── need/{dataset_tag}/{experiment_name}/   ← git tracked（分析数据）
│   ├── metrics.json
│   ├── strategy_eval_llm.json
│   ├── swap_samples_valid.jsonl
│   ├── run_config.json
│   └── label_map.json
│
├── other/{dataset_tag}/{experiment_name}/  ← git ignored（模型权重）
│   ├── prefix_bank.pt
│   ├── lora_adapter/
│   └── tokenizer/
```

## 实验矩阵

### Phase 1：基线实验（无依赖）

| 实验 | 配置 | adapter_mode | 说明 |
|---|---|---|---|
| B2 | `b2_lora_only.json` | lora_only | LoRA 单分支基线 |
| B3 | `b3_prefix_only.json` | prefix_only | Prefix 单分支基线 |
| B4 | `b4_prefix_lora.json` | prefix_lora | Prefix + LoRA 混合 |
| B5 | `b5_prefix_lora_orth.json` | prefix_lora_orth | + 正交约束 |
| B6 | `b6_dest_rs.json` | dest_rs | 完整 DeSTRS（+ 分类器） |

### Phase 2：依赖实验

| 实验 | 配置 | 依赖 | 说明 |
|---|---|---|---|
| B7 | `b7_dest_rs_warm.json` | B3 的 prefix | warm-start DeSTRS |
| B8 | `b8_lora_then_prefix.json` | B2 的 LoRA | 两阶段：先 LoRA → 再 Prefix |
| B9 | `b9_prefix_then_lora.json` | B3 的 prefix | 两阶段：先 Prefix → 再 LoRA |

### 实验递进关系

```
B2 (LoRA only)  ──────────────────→ B8 (冻结B2的LoRA，训Prefix)
B3 (Prefix only) ─┬──→ B7 (用B3的prefix初始化，训完整DeSTRS)
                   └──→ B9 (冻结B3的prefix，训LoRA)
B4 (Prefix+LoRA)
B5 (+ 正交约束)
B6 (完整 DeSTRS)
```

## 快速开始

1. 修改配置文件中 `model_name_or_path` 为本地模型路径
2. 运行全部实验：

```bash
# 使用默认数据集（casino_original）
./run_all.sh

# 使用其他数据集
DATASET_TAG=casino_augmented DATASET_DIR=./augmented_data ./run_all.sh
```

3. 单独运行某个实验：

```bash
python train.py --config configs/b6_dest_rs.json
python train.py --config configs/b6_dest_rs.json --dataset-tag casino_augmented --dataset-dir ./augmented_data
```

## 损失函数

```
L = L_gen + λ_orth * L_orth + λ_cls * L_cls
```

- `L_gen`：自回归生成损失
- `L_orth`：Prefix 与 LoRA 增量表示的局部-全局正交约束
- `L_cls`：基于 delta_prefix 的策略分类辅助损失

## 数据集说明

### CaSiNo 原始数据集

- 来源：Chawla et al., 2021
- 总计 1030 个对话，仅 396 个有策略标注
- 9 种谈判策略：elicit-pref, self-need, other-need, no-need, promote-coordination, showing-empathy, small-talk, uv-part, vouch-fair
- 已知局限：标注覆盖率低、类别严重不均衡（最大类 vs 最小类 ≈ 15:1）、约 30% 样本为多标签

### 增强数据集（LLM 生成）

基于 CaSiNo 的 9 种策略体系，使用 Qwen3.5-9B 逐轮生成策略驱动的谈判对话，解决原始数据集的标注稀疏和类别不均衡问题。

生成设计：
- **逐轮生成**：每轮指定一个策略，prompt 包含策略定义、示例和对话历史，保证单标签、策略信号强
- **策略序列规划**：对话流模板模拟自然对话节奏（寒暄 → 信息交换 → 推动成交），优先补充缺口大的策略
- **目标**：每类策略补到 500 条训练样本，总计生成约 2883 条新数据（约 300 个对话）
- **测试集**：CaSiNo 原始 test set（可比性）+ 生成数据留 10%（新数据评估）

## 数据生成

```bash
# 1. 生成增强数据
python scripts/generate_augmented_data.py \
    --model-path /home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3.5-9B \
    --casino-train CaSiNo-main/data/split/casino_train.json \
    --casino-valid CaSiNo-main/data/split/casino_valid.json \
    --casino-test CaSiNo-main/data/split/casino_test.json \
    --output-dir ./augmented_data/split \
    --target-per-strategy 500 \
    --seed 42

# 2. 在增强数据上运行全部实验
DATASET_TAG=casino_augmented DATASET_DIR=./augmented_data ./run_all.sh
```

生成输出：

```
augmented_data/
└── split/
    ├── casino_train.json      # CaSiNo 原始训练集 + 生成数据 90%
    ├── casino_valid.json      # CaSiNo 原始 valid set（不变）
    ├── casino_test.json       # CaSiNo 原始 test set（不变）
    ├── generated_test.json    # 生成数据 10%（第二测试集）
    └── generation_stats.json  # 生成统计
```

## 实验流程总览

```bash
# 第一组：CaSiNo 原始数据
./run_all.sh

# 第二组：增强数据（生成完成后）
DATASET_TAG=casino_augmented DATASET_DIR=./augmented_data ./run_all.sh
```

两组实验结果分别保存在 `output/need/casino_original/` 和 `output/need/casino_augmented/` 下，可直接对比。
