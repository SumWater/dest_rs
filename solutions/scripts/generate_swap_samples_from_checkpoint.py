#!/usr/bin/env python3
"""Regenerate prefix-swap samples from an existing checkpoint.

Use this when a training run only saved 5 validation records but we need the
same 30-record/270-swap evaluation size as the main baselines.  It does not
train; it loads the checkpoint from solutions/output/other/... and writes a
fresh swap_samples_*.jsonl file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOLUTIONS_ROOT = Path(__file__).resolve().parents[1]
if str(SOLUTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTIONS_ROOT))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.casino_dataset import CasinoStrategyDataset, StrategyLabelSpace, load_split_examples  # noqa: E402
from src.config import load_config, resolve_warm_start_dir  # noqa: E402
from src.evaluate import save_swap_samples  # noqa: E402
from src.modeling import build_hybrid_model, freeze_for_adapter_mode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate swap samples from a saved checkpoint")
    parser.add_argument("--config", required=True, help="Config matching the trained experiment")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir containing prefix_bank.pt")
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--split", default="valid", choices=["valid", "test"])
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--out", required=True, help="Output JSONL path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_tag:
        cfg.dataset_tag = args.dataset_tag
    cfg.warm_start_dir = args.checkpoint
    cfg.warm_start_prefix = True
    cfg.warm_start_lora = True
    cfg.demo_num_examples = args.num_examples
    resolve_warm_start_dir(cfg)

    train_examples = load_split_examples(cfg, "train")
    label_space = StrategyLabelSpace.fit(train_examples)
    split_examples = load_split_examples(cfg, args.split)
    dataset = CasinoStrategyDataset(split_examples, label_space)

    print(f"[swap-gen] loading checkpoint: {cfg.warm_start_dir}")
    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    catcher.remove()
    freeze_for_adapter_mode(hybrid, cfg)
    hybrid.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        save_swap_samples(
            hybrid=hybrid,
            tokenizer=tokenizer,
            dataset=dataset,
            label_space=label_space,
            cfg=cfg,
            output_path=str(out),
            num_examples=args.num_examples,
            split_name=args.split,
        )
    print(f"[swap-gen] saved {min(args.num_examples, len(dataset))} records to {out}")


if __name__ == "__main__":
    main()



