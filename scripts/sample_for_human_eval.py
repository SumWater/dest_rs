#!/usr/bin/env python3
"""从 swap_samples 中抽样，生成人工评估标注表格。

用法：
  python scripts/sample_for_human_eval.py \
      --swap output/need/casino_augmented/b3_prefix_only/swap_samples_valid.jsonl \
      --swap output/need/casino_augmented/b4_prefix_lora/swap_samples_valid.jsonl \
      --swap output/need/casino_augmented_fix_b6/b7_ep2/swap_samples_valid.jsonl \
      --swap output/need/casino_augmented_fix_b6/b9_prefix_then_lora/swap_samples_valid.jsonl \
      --labels B3 B4 B7 B9 \
      --contexts-per-model 5 \
      --out human_eval_samples.csv

输出 CSV 列：
  model, dialogue_id, turn_index, gold_strategy, target_strategy,
  context, utterance, fluency(留空), strategy_match(留空),
  is_template(留空), notes(留空)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


COLUMNS = [
    "model",
    "dialogue_id",
    "turn_index",
    "gold_strategy",
    "target_strategy",
    "context",
    "utterance",
    "fluency",          # 1-5，留空给标注者填
    "strategy_match",   # yes/no，留空给标注者填
    "is_template",      # yes/no，留空给标注者填
    "notes",            # 自由文本，留空给标注者填
]


def load_swap(path: str) -> list[dict]:
    """加载 swap_samples JSONL。"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="抽样生成人工评估表格")
    parser.add_argument("--swap", "-s", required=True, nargs="+", help="swap_samples JSONL 文件路径")
    parser.add_argument("--labels", "-l", required=True, nargs="+", help="每个 swap 文件的模型标签")
    parser.add_argument("--contexts-per-model", "-n", type=int, default=5, help="每模型抽样 context 数")
    parser.add_argument("--out", "-o", default="human_eval_samples.csv", help="输出 CSV 路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    if len(args.swap) != len(args.labels):
        print(f"错误: --swap ({len(args.swap)} 个) 和 --labels ({len(args.labels)} 个) 数量不匹配")
        sys.exit(1)

    import random
    rng = random.Random(args.seed)

    rows = []
    for swap_path, label in zip(args.swap, args.labels):
        records = load_swap(swap_path)
        if not records:
            print(f"警告: {swap_path} 为空")
            continue

        # 按 dialogue_id 分组
        by_dialogue: dict[int, list[dict]] = {}
        for r in records:
            did = r.get("dialogue_id", 0)
            by_dialogue.setdefault(did, []).append(r)

        # 随机选 N 个 dialogue
        dialogue_ids = list(by_dialogue.keys())
        rng.shuffle(dialogue_ids)
        selected_dids = dialogue_ids[: args.contexts_per_model]

        print(f"\n{label} ({Path(swap_path).parent.name}):")
        print(f"  总 dialogues: {len(dialogue_ids)}, 抽样: {len(selected_dids)}")

        for did in sorted(selected_dids):
            for rec in by_dialogue[did]:
                generated = rec.get("generated_by_strategy") or {}
                for strategy_name, utterance in generated.items():
                    utterance = (utterance or "").strip()
                    if len(utterance) < 2:
                        continue
                    context = (rec.get("prompt") or "").strip()
                    # 截断过长上下文
                    if len(context) > 800:
                        context = context[:800] + "..."
                    rows.append({
                        "model": label,
                        "dialogue_id": str(did),
                        "turn_index": str(rec.get("turn_index", "")),
                        "gold_strategy": rec.get("gold_strategy", ""),
                        "target_strategy": strategy_name,
                        "context": context,
                        "utterance": utterance,
                        "fluency": "",
                        "strategy_match": "",
                        "is_template": "",
                        "notes": "",
                    })

    # 打乱顺序避免标注偏置
    rng.shuffle(rows)

    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"\n共 {total} 条标注样本，已保存至 {args.out}")
    print(f"标注维度: fluency (1-5), strategy_match (yes/no), is_template (yes/no), notes")
    print(f"每个 context × 9 strategies ≈ {9 * args.contexts_per_model} 条/模型")


if __name__ == "__main__":
    main()
