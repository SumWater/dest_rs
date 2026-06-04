from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.casino_dataset import CasinoStrategyDataset, StrategyDataCollator, StrategyLabelSpace, load_split_examples
from src.config import load_config
from src.losses import pool_target_tokens
from src.modeling import build_hybrid_model, get_embed_device, lora_disabled_ctx


@torch.no_grad()
def collect_batch_metrics(hybrid, catcher, batch: dict, cfg) -> dict[str, float]:
    device = get_embed_device(hybrid.peft_model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    strategy_ids = batch["strategy_id"].to(device)

    hybrid.eval()

    # base-only：关闭 Prefix 与 LoRA。
    catcher.reset()
    with lora_disabled_ctx(hybrid.peft_model):
        _ = hybrid(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            labels=None,
            prefix_on=False,
            prefix_scale=cfg.prefix_scale_eval,
            use_cache=False,
        )
        h_base = catcher.pop().detach()

    # prefix-only：开启 Prefix，关闭 LoRA。
    catcher.reset()
    with lora_disabled_ctx(hybrid.peft_model):
        _ = hybrid(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            labels=None,
            prefix_on=True,
            prefix_scale=cfg.prefix_scale_eval,
            use_cache=False,
        )
        h_prefix = catcher.pop()
        h_prefix = hybrid.slice_real_tokens(h_prefix, prefix_on=True)

    # lora-only：关闭 Prefix，开启 LoRA。
    catcher.reset()
    _ = hybrid(
        input_ids=input_ids,
        attention_mask=attention_mask,
        strategy_ids=strategy_ids,
        labels=None,
        prefix_on=False,
        prefix_scale=cfg.prefix_scale_eval,
        use_cache=False,
    )
    h_lora = catcher.pop()
    h_lora = hybrid.slice_real_tokens(h_lora, prefix_on=False)

    delta_prefix = h_prefix - h_base
    delta_lora = h_lora - h_base
    target_mask = labels.ne(-100).to(delta_prefix.device)

    if target_mask.sum().item() == 0:
        return {
            "tokens": 0.0,
            "mean_cosine": 0.0,
            "mean_cosine_squared": 0.0,
            "frobenius_corr": 0.0,
            "prefix_delta_norm": 0.0,
            "lora_delta_norm": 0.0,
        }

    indices = torch.nonzero(target_mask, as_tuple=False)
    prefix_vectors = delta_prefix[indices[:, 0], indices[:, 1], :]
    lora_vectors = delta_lora[indices[:, 0], indices[:, 1], :]
    cosine = F.cosine_similarity(prefix_vectors, lora_vectors, dim=-1)

    prefix_pooled = pool_target_tokens(delta_prefix, target_mask)
    lora_pooled = pool_target_tokens(delta_lora, target_mask)
    prefix_norm = F.normalize(prefix_pooled, dim=-1)
    lora_norm = F.normalize(lora_pooled, dim=-1)
    cross_corr = torch.matmul(prefix_norm.transpose(0, 1), lora_norm) / max(1, prefix_norm.size(0))

    return {
        "tokens": float(indices.size(0)),
        "mean_cosine": float(cosine.mean().item()),
        "mean_abs_cosine": float(cosine.abs().mean().item()),
        "mean_cosine_squared": float((cosine**2).mean().item()),
        "frobenius_corr": float((torch.norm(cross_corr, p="fro") ** 2).item()),
        "prefix_delta_norm": float(prefix_vectors.norm(dim=-1).mean().item()),
        "lora_delta_norm": float(lora_vectors.norm(dim=-1).mean().item()),
    }


def weighted_average(rows: list[dict[str, float]], key: str) -> float:
    total_weight = sum(row.get("tokens", 0.0) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(row[key] * row.get("tokens", 0.0) for row in rows) / total_weight


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 checkpoint 中 Prefix/LoRA 表示增量的重合程度。")
    parser.add_argument("--checkpoint-dir", required=True, help="模型权重目录 (output/other/...)。")
    parser.add_argument("--need-dir", default=None, help="分析数据目录 (output/need/...)。默认从 checkpoint-dir 推导。")
    parser.add_argument("--config", default=None, help="可选：覆盖 checkpoint 中的 run_config.json。")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"], help="分析哪个数据划分。")
    parser.add_argument("--max-samples", type=int, default=64, help="最多分析多少条样本。")
    parser.add_argument("--batch-size", type=int, default=1, help="分析 batch size。")
    parser.add_argument("--out", default=None, help="可选：输出 JSON 路径。")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    need_dir = Path(args.need_dir) if args.need_dir else Path(str(checkpoint_dir).replace("output/other/", "output/need/"))
    config_path = Path(args.config) if args.config else need_dir / "run_config.json"
    cfg = load_config(str(config_path))
    cfg.warm_start_dir = str(checkpoint_dir)
    cfg.eval_batch_size = args.batch_size

    with (need_dir / "label_map.json").open("r", encoding="utf-8") as f:
        label_space = StrategyLabelSpace.from_json(json.load(f))

    tokenizer = None
    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    examples = load_split_examples(cfg, args.split)
    if args.max_samples and args.max_samples > 0:
        examples = examples[: args.max_samples]
    dataset = CasinoStrategyDataset(examples, label_space)
    collator = StrategyDataCollator(tokenizer, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    rows = []
    for batch in loader:
        rows.append(collect_batch_metrics(hybrid, catcher, batch, cfg))

    catcher.remove()
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "split": args.split,
        "samples": len(dataset),
        "batches": len(rows),
        "tokens": sum(row.get("tokens", 0.0) for row in rows),
        "mean_cosine": weighted_average(rows, "mean_cosine"),
        "mean_abs_cosine": weighted_average(rows, "mean_abs_cosine"),
        "mean_cosine_squared": weighted_average(rows, "mean_cosine_squared"),
        "frobenius_corr": weighted_average(rows, "frobenius_corr"),
        "prefix_delta_norm": weighted_average(rows, "prefix_delta_norm"),
        "lora_delta_norm": weighted_average(rows, "lora_delta_norm"),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
