# 论文配图规格说明

## 通用要求
- 所有图需同时生成 PNG（dpi≥250）和 PDF 矢量版本
- 配色方案（五类）：baseline 灰 #b0b0b0 / LoRA 蓝 #4c72b0 / Prefix 橙 #dd8452 / Joint 绿 #55a868 / Diagnostic 红 #c44e52
- IEEEtran 双栏，图宽适应单栏（columnwidth），字体大小以图中可读为准
- 中文标注用英文（论文是英文）

---

## Fig 1: 策略准确率 vs 话术质量 双轴柱状图

### 目标
展示 6 个核心配置中策略控制（准确率）与话术质量（PPL）的 trade-off。诊断变体不放入此图。

### 数据

| 配置 | 策略准确率 (%) | Acc Std | PPL | PPL Std |
|---|---|---|---|---|
| Frozen Base + Strategy Text | 27.41 | — | — | — |
| LoRA + Strategy Text | 42.22 | — | 7.81 | — |
| Prefix-only (B3) | 55.87 | 1.28 | 10.54 | 0.12 |
| Prefix+LoRA (B4) | 37.20 | 3.78 | 7.93 | 0.17 |
| Warm-start (B7) | 40.49 | 2.19 | 8.18 | 0.24 |
| Prefix→LoRA (B9) | 40.00 | 2.98 | 8.07 | 0.10 |

### 图表设计
- **类型**：分组柱状图（grouped bar chart），每组两根柱子并排
- **X 轴**：6 个配置名（按上表顺序）
- **左 Y 轴**：策略准确率 (%)，范围 0–65，每格 10
- **右 Y 轴**：PPL，范围 7–11，每格 1（注意 PPL 越低越好，所以 Y 轴方向可以和直觉相反标注）
- **柱子颜色**：准确率柱用类别色（B3=橙, B4/B7/B9=绿, baselines=灰/蓝），PPL 柱统一用浅色或斜线纹理区分
- **误差棒**：B3/B4/B7/B9 两根柱子都标注误差棒（Acc Std 和 PPL Std）
- **辅助线**：在准确率 11.11% 处画虚线标注 "Random (11.1%)"
- **图例**：Acc vs PPL 两根柱的图例
- **尺寸**：宽约 6.5 inch，高约 3.5 inch（或适应 IEEEtran 单栏宽度）

---

## Fig 2: 逐策略准确率对比

### 目标
展示 B3（Prefix-only）和 B4（Prefix+LoRA）在 9 个策略上的逐类准确率，并对比评估器校准准确率。

### 数据

| 策略 | B3 Acc (%) | B4 Acc (%) | 评估器校准 Acc (%) |
|---|---|---|---|
| elicit-pref | 78.95 | 78.95 | 84.2 |
| self-need | 56.76 | 35.14 | 51.4 |
| other-need | 60.00 | 0.00 | 30.0 |
| no-need | 0.00 | 50.00 | 0.0 |
| promote-coordination | 62.07 | 24.14 | 51.7 |
| showing-empathy | 50.00 | 10.00 | 40.0 |
| small-talk | 59.15 | 45.07 | 57.7 |
| uv-part | 50.00 | 25.00 | 25.0 |
| vouch-fair | 71.43 | 64.29 | 35.7 |

### 图表设计
- **类型**：分组柱状图，每组三根柱子并排
- **X 轴**：9 个策略名（缩写或用完整名，可 45° 倾斜），按 B3 准确率降序排列
- **Y 轴**：准确率 (%)，范围 0–100
- **颜色**：B3=橙, B4=绿, Evaluator=灰
- **图例**：三根柱的图例
- **尺寸**：宽约 columnwidth，高约 3.5 inch

### 注意事项
- no-need 的 B3=0% 和 B4=50% 是真实数据，不是错误；评估器对 no-need 的校准也是 0%，这是需要标注的关键发现

---

## Fig 3: 诊断框架总览图

### 目标
展示 Prefix-LoRA 混合 PEFT 的架构和诊断变体的信息流。

### 已生成文件
- `paper_figures/fig3_framework.png` / `.pdf`（已定稿，无需修改）
- 学术线框图风格，黑/白/灰，用 FancyBboxPatch 绘制模块框
- 展示：Input → Prefix Bank + LoRA → Frozen Qwen3-8B → Output
- 虚线标注诊断信号：L_orth, L_cls, L_contrastive, gradient routing

---

## Fig 4: 评估器混淆矩阵

### 目标
展示 LLM 评估器在 200 条人工标注验证集上的 9×9 混淆矩阵。

### 数据来源
- `output/evaluator_calibration_results.json`（200 条，字段：gold, pred, utterance）
- 9 个策略类：elicit-pref, small-talk, promote-coordination, self-need, showing-empathy, vouch-fair, other-need, uv-part, no-need

### 汇总数据（可直接用，与 JSON 一致）

| 真实 \ 预测 | elicit | smallTalk | promCoord | selfNeed | empathy | vouchFair | otherNeed | uvPart | noNeed | 总计 |
|---|---|---|---|---|---|---|---|---|---|---|
| elicit-pref | 16 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 19 |
| small-talk | 2 | 41 | 5 | 3 | 12 | 1 | 1 | 2 | 4 | 71 |
| promote-coord | 0 | 1 | 15 | 9 | 2 | 0 | 1 | 1 | 0 | 29 |
| self-need | 3 | 1 | 2 | 19 | 1 | 5 | 2 | 3 | 1 | 37 |
| showing-empathy | 0 | 0 | 3 | 1 | 4 | 2 | 0 | 0 | 0 | 10 |
| vouch-fair | 0 | 1 | 1 | 5 | 2 | 5 | 0 | 0 | 0 | 14 |
| other-need | 1 | 0 | 0 | 5 | 0 | 1 | 3 | 0 | 0 | 10 |
| uv-part | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 4 |
| no-need | 0 | 0 | 0 | 3 | 0 | 1 | 0 | 1 | 1 | 6 |

注：可用 `output/evaluator_calibration_results.json` 精确生成。

### 图表设计
- **类型**：热力图（heatmap），YlOrRd 色阶
- **行**：真实标签（Gold），列：预测标签（Predicted）
- **单元格标注**：计数 + 行百分比，如 "16\n(84%)"
- **右侧额外列**：每类准确率百分比
- **标题/副标题**：Overall accuracy: 52.0% | Total: 200 samples
- **尺寸**：宽约 7 inch，高约 6 inch

### 已生成文件
- `paper_figures/fig4_confusion.png` / `.pdf`（已定稿可参考，如需微调可重新生成）

---

## 补充说明
- Fig1 和 Fig2 的绘图脚本在 `scripts/paper_figures.py`，可作为参考但图稿师不需拘泥于 matplotlib 限制
- 所有实验数据详见 `实验完整记录.md`
- 论文最终编译用 `paper_figures/paper.tex`，Overleaf 上传时需把所有 `.png` 图放入根目录
