#!/usr/bin/env python3
"""从已有 checkpoint 重新生成 swap samples（不需要重训，支持 5.1 扩大评估）。

用法：
  python scripts/regenerate_swap_samples.py \
      --checkpoint-dir output/other/casino_augmented/b3_prefix_only \
      --need-dir output/need/casino_augmented/b3_prefix_only \
      --num-examples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.casino_dataset import CasinoStrategyDataset, StrategyDataCollator, StrategyLabelSpace, load_split_examples
from src.config import load_config
from src.evaluate import save_swap_samples
from src.modeling import build_hybrid_model


def main() -> None:
    parser = argparse.ArgumentParser(description="从已有 checkpoint 重新生成 swap samples")
    parser.add_argument("--checkpoint-dir", required=True, help="模型权重目录 (output/other/...)")
    parser.add_argument("--need-dir", default=None, help="输出目录 (output/need/...)，默认从 checkpoint-dir 推导")
    parser.add_argument("--config", default=None, help="覆盖 run_config.json")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--num-examples", type=int, default=100, help="生成 context 数量")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    need_dir = Path(args.need_dir) if args.need_dir else Path(
        str(checkpoint_dir).replace("output/other/", "output/need/")
    )
    config_path = Path(args.config) if args.config else need_dir / "run_config.json"
    if not config_path.exists():
        # 尝试从 checkpoint_dir 找
        alt = checkpoint_dir / "run_config.json"
        if alt.exists():
            config_path = alt

    cfg = load_config(str(config_path))
    cfg.warm_start_dir = str(checkpoint_dir)
    cfg.eval_batch_size = args.batch_size

    # 临时改 demo_num_examples
    cfg.demo_num_examples = args.num_examples

    with (need_dir / "label_map.json").open("r", encoding="utf-8") as f:
        label_space = StrategyLabelSpace.from_json(json.load(f))

    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    examples = load_split_examples(cfg, args.split)
    dataset = CasinoStrategyDataset(examples, label_space)

    output_path = str(need_dir / f"swap_samples_{args.split}.jsonl")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Split:      {args.split}")
    print(f"Contexts:   {min(args.num_examples, len(dataset))}")
    print(f"Output:     {output_path}")
    print()

    save_swap_samples(
        hybrid=hybrid,
        tokenizer=tokenizer,
        dataset=dataset,
        label_space=label_space,
        cfg=cfg,
        output_path=output_path,
        num_examples=args.num_examples,
        split_name=args.split,
    )

    total = min(args.num_examples, len(dataset)) * len(label_space.labels)
    print(f"完成，共 {total} 条生成")

    catcher.remove()


if __name__ == "__main__":
    main()
