from __future__ import annotations

import argparse
import json
import os

import torch

from src.config import load_config
from src.casino_dataset import CasinoStrategyDataset, StrategyLabelSpace, load_split_examples
from src.evaluate import greedy_generate
from src.modeling import build_hybrid_model, get_embed_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--strategy_a", default="self-need")
    parser.add_argument("--strategy_b", default="promote-coordination")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--num_examples", type=int, default=3)
    parser.add_argument("--betas", default="0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    cfg = load_config(os.path.join(args.checkpoint_dir, "run_config.json"))
    cfg.warm_start_dir = args.checkpoint_dir
    cfg.output_dir = args.checkpoint_dir

    with open(os.path.join(args.checkpoint_dir, "label_map.json"), "r", encoding="utf-8") as f:
        label_space = StrategyLabelSpace.from_json(json.load(f))
    if args.strategy_a not in label_space.label_to_id or args.strategy_b not in label_space.label_to_id:
        raise ValueError(f"未知策略组合。可用策略标签：{label_space.labels}")

    split_examples = load_split_examples(cfg, args.split)
    dataset = CasinoStrategyDataset(split_examples, label_space)
    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    device = get_embed_device(hybrid.peft_model)

    id_a = label_space.label_to_id[args.strategy_a]
    id_b = label_space.label_to_id[args.strategy_b]
    betas = [float(x.strip()) for x in args.betas.split(",") if x.strip()]

    records = []
    with torch.no_grad():
        prefix_a = hybrid.prefix_bank[id_a].detach().to(device)
        prefix_b = hybrid.prefix_bank[id_b].detach().to(device)
        for idx in range(min(args.num_examples, len(dataset))):
            item = dataset[idx]
            generated = {}
            for beta in betas:
                mixed_prefix = beta * prefix_a + (1.0 - beta) * prefix_b
                generated[str(beta)] = greedy_generate(
                    hybrid=hybrid,
                    tokenizer=tokenizer,
                    prompt=item["prompt"],
                    strategy_id=id_a,
                    cfg=cfg,
                    prefix_override=mixed_prefix,
                )
            records.append(
                {
                    "split": args.split,
                    "dialogue_id": item["dialogue_id"],
                    "turn_index": item["turn_index"],
                    "strategy_a": args.strategy_a,
                    "strategy_b": args.strategy_b,
                    "gold_strategy": item["primary_strategy"],
                    "gold_target": item["target"],
                    "prompt": item["prompt"],
                    "generated_by_beta": generated,
                }
            )

    output_path = os.path.join(args.checkpoint_dir, f"interpolation_{args.strategy_a}_to_{args.strategy_b}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    catcher.remove()
    print(f"[done] 已写入 {output_path}")


if __name__ == "__main__":
    main()
