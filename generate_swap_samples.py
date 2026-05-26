from __future__ import annotations

import argparse
import json
import os

from torch.utils.data import DataLoader

from src.config import load_config
from src.casino_dataset import CasinoStrategyDataset, StrategyLabelSpace, load_split_examples
from src.evaluate import save_swap_samples
from src.modeling import build_hybrid_model, load_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--num_examples", type=int, default=5)
    args = parser.parse_args()

    config_path = args.config or os.path.join(args.checkpoint_dir, "run_config.json")
    cfg = load_config(config_path)
    cfg.warm_start_dir = args.checkpoint_dir
    cfg.output_dir = args.checkpoint_dir

    tokenizer = load_tokenizer(cfg)
    with open(os.path.join(args.checkpoint_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label_space = StrategyLabelSpace.from_json(json.load(f))

    split_examples = load_split_examples(cfg, args.split)
    dataset = CasinoStrategyDataset(split_examples, label_space)
    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))

    output_path = os.path.join(args.checkpoint_dir, f"swap_samples_{args.split}.jsonl")
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
    catcher.remove()
    print(f"[done] 已写入 {output_path}")


if __name__ == "__main__":
    main()
