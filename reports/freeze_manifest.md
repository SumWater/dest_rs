# 实验冻结清单

- 冻结基准提交：`4c10480bbdba29172ddcc51cc9c6cb750da16fef`（`main`）
- 计划标签：`before_counterfactual_strategy_experiments`
- 未跟踪用户文件：`当前实验意图与结果总览.md`（保留，不纳入冻结提交）
- 配置：`configs/`
- 训练/评估代码：`train.py`、`src/`、`scripts/`
- D0：`CaSiNo-main/data/split/`
- D1：`augmented_data/split/`
- 生成与评估结果：`output/need/`
- checkpoint 预期目录：`output/other/`（实际权重保存在另一台实验主机，本机只保留结果侧文件）

## 冻结完整性结论

本机可以冻结代码、配置、数据和已保存的 generation/evaluation。B3/B4/B9、S1、S2_qk 的 checkpoint 与原训练环境位于另一台实验主机，应在该主机原地生成清单和哈希；不要求将全部大权重复制到本机。后续 M-01～M-03 仍应在实验主机使用登记后的 B9 checkpoint。

本机 Python/CUDA 状态不属于历史实验环境，不写入可复现性结论。
