#!/usr/bin/env python3
"""分析 LoRA 对 Prefix 策略信号的影响（支撑 5.5 表示空间分析）。

核心问题：B9 冻结 B3 Prefix 后只训 LoRA，策略控制仍下降 4.3pp。
假设 LoRA 改变了 Q/K/V/O 投影，使 Prefix 在"脏环境"中失效。

本脚本对指定 checkpoint 的同一批输入，分别计算：
  1. delta_clean  = h(prefix_on, lora_off) - h(prefix_off, lora_off)
  2. delta_lora   = h(prefix_on, lora_on)  - h(prefix_off, lora_on)

然后比较：
  - cos(delta_clean, delta_lora)：方向一致性（高=信号方向不变但可能减弱）
  - ||delta_lora|| / ||delta_clean||：LoRA 是否削弱 prefix effect
  - 按策略分组，看 other-need、uv-part 等是否被 LoRA 明显抹掉

用法：
  # 分析 B9（冻结 B3 Prefix + 训练的 LoRA）
  python scripts/analyze_lora_prefix_interference.py \
      --checkpoint-dir output/other/casino_augmented_fix_b6/b9_prefix_then_lora \
      --need-dir output/need/casino_augmented_fix_b6/b9_prefix_then_lora \
      --split valid \
      --max-samples 60

  # 也用于 B4（联合训练的 Prefix+LoRA）
  python scripts/analyze_lora_prefix_interference.py \
      --checkpoint-dir output/other/casino_augmented/b4_prefix_lora \
      --need-dir output/need/casino_augmented/b4_prefix_lora \
      --split valid \
      --max-samples 60
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.casino_dataset import CasinoStrategyDataset, StrategyDataCollator, StrategyLabelSpace, load_split_examples
from src.config import load_config
from src.modeling import build_hybrid_model, get_embed_device, lora_disabled_ctx

ALL_STRATEGIES = [
    "elicit-pref", "self-need", "other-need", "no-need",
    "promote-coordination", "showing-empathy", "small-talk",
    "uv-part", "vouch-fair",
]


@torch.no_grad()
def compute_prefix_effects(
    hybrid, catcher, input_ids, attention_mask, strategy_ids, cfg
) -> dict:
    """对一个 batch 计算 clean 和 lora 环境下的 prefix delta。"""
    device = get_embed_device(hybrid.peft_model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    strategy_ids = strategy_ids.to(device)

    hybrid.eval()

    # ── Clean 环境：LoRA off ──
    with lora_disabled_ctx(hybrid.peft_model):
        # Clean base
        catcher.reset()
        _ = hybrid(
            input_ids=input_ids, attention_mask=attention_mask,
            strategy_ids=strategy_ids, labels=None,
            prefix_on=False, prefix_scale=cfg.prefix_scale_eval, use_cache=False,
        )
        h_base_clean = catcher.pop().detach()

        # Clean prefix
        catcher.reset()
        _ = hybrid(
            input_ids=input_ids, attention_mask=attention_mask,
            strategy_ids=strategy_ids, labels=None,
            prefix_on=True, prefix_scale=cfg.prefix_scale_eval, use_cache=False,
        )
        h_prefix_clean = catcher.pop().detach()

    h_prefix_clean = hybrid.slice_real_tokens(h_prefix_clean, prefix_on=True)

    # ── LoRA 环境：LoRA on ──
    # LoRA base
    catcher.reset()
    _ = hybrid(
        input_ids=input_ids, attention_mask=attention_mask,
        strategy_ids=strategy_ids, labels=None,
        prefix_on=False, prefix_scale=cfg.prefix_scale_eval, use_cache=False,
    )
    h_base_lora = catcher.pop().detach()

    # LoRA prefix
    catcher.reset()
    _ = hybrid(
        input_ids=input_ids, attention_mask=attention_mask,
        strategy_ids=strategy_ids, labels=None,
        prefix_on=True, prefix_scale=cfg.prefix_scale_eval, use_cache=False,
    )
    h_prefix_lora = catcher.pop().detach()

    h_base_lora = hybrid.slice_real_tokens(h_base_lora, prefix_on=False)
    h_prefix_lora = hybrid.slice_real_tokens(h_prefix_lora, prefix_on=True)

    delta_clean = h_prefix_clean - h_base_clean
    delta_lora = h_prefix_lora - h_base_lora

    # ── 逐 token 计算指标 ──
    # 展平 (B*T, D)
    bsz, seq_len, dim = delta_clean.shape
    dc = delta_clean.reshape(-1, dim)
    dl = delta_lora.reshape(-1, dim)

    dc_norm = dc.norm(dim=-1)
    dl_norm = dl.norm(dim=-1)

    # 过滤零向量
    valid = (dc_norm > 1e-8) & (dl_norm > 1e-8)

    if valid.sum() == 0:
        return {
            "n_tokens": 0,
            "mean_cosine": 0.0,
            "mean_norm_ratio": 1.0,
            "mean_clean_norm": 0.0,
            "mean_lora_norm": 0.0,
        }

    cos = F.cosine_similarity(dc[valid], dl[valid], dim=-1)

    return {
        "n_tokens": int(valid.sum().item()),
        "mean_cosine": float(cos.mean().item()),
        "mean_norm_ratio": float((dl_norm[valid] / dc_norm[valid].clamp_min(1e-8)).mean().item()),
        "mean_clean_norm": float(dc_norm[valid].mean().item()),
        "mean_lora_norm": float(dl_norm[valid].mean().item()),
    }


def weighted_avg(rows: list[dict], key: str) -> float:
    total_w = sum(r.get("n_tokens", 0) for r in rows)
    if total_w <= 0:
        return 0.0
    return sum(r[key] * r.get("n_tokens", 0) for r in rows) / total_w


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 LoRA 对 Prefix 策略信号的影响"
    )
    parser.add_argument("--checkpoint-dir", required=True, help="模型权重目录")
    parser.add_argument("--need-dir", default=None, help="分析数据目录")
    parser.add_argument("--config", default=None, help="覆盖 run_config.json")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--max-samples", type=int, default=100, help="最多分析样本数")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--out", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    need_dir = Path(args.need_dir) if args.need_dir else Path(
        str(checkpoint_dir).replace("output/other/", "output/need/")
    )
    config_path = Path(args.config) if args.config else need_dir / "run_config.json"
    cfg = load_config(str(config_path))
    cfg.warm_start_dir = str(checkpoint_dir)
    cfg.eval_batch_size = args.batch_size

    with (need_dir / "label_map.json").open("r", encoding="utf-8") as f:
        label_space = StrategyLabelSpace.from_json(json.load(f))

    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    examples = load_split_examples(cfg, args.split)
    if args.max_samples and args.max_samples > 0:
        examples = examples[: args.max_samples]
    dataset = CasinoStrategyDataset(examples, label_space)
    collator = StrategyDataCollator(tokenizer, cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # ── 收集结果 ──
    all_rows: list[dict] = []
    per_strategy_rows: dict[str, list[dict]] = defaultdict(list)

    for batch in loader:
        row = compute_prefix_effects(
            hybrid, catcher,
            batch["input_ids"], batch["attention_mask"],
            batch["strategy_id"], cfg,
        )
        row["strategy"] = batch["meta"][0].get("primary_strategy", "unknown") if batch.get("meta") else "unknown"
        all_rows.append(row)
        per_strategy_rows[row["strategy"]].append(row)

    catcher.remove()

    # ── 汇总 ──
    print(f"\n{'═' * 70}")
    print(f"  LoRA 对 Prefix 策略信号的影响分析")
    print(f"  Checkpoint: {checkpoint_dir}")
    print(f"  Samples: {len(all_rows)}")
    print(f"{'═' * 70}\n")

    # Overall
    print("── Overall ──")
    print(f"  mean cosine similarity:      {weighted_avg(all_rows, 'mean_cosine'):.4f}")
    print(f"  mean norm ratio (lora/clean): {weighted_avg(all_rows, 'mean_norm_ratio'):.4f}")
    print(f"  mean clean prefix norm:       {weighted_avg(all_rows, 'mean_clean_norm'):.4f}")
    print(f"  mean lora prefix norm:        {weighted_avg(all_rows, 'mean_lora_norm'):.4f}")

    # 解释
    cos = weighted_avg(all_rows, "mean_cosine")
    nr = weighted_avg(all_rows, "mean_norm_ratio")
    if cos > 0.9 and nr < 0.85:
        print("\n  → Prefix 信号方向基本不变 (cos>0.9)，但强度被 LoRA 削弱 (ratio<0.85)")
        print("    支持'LoRA 改变 Prefix 运行环境但未完全破坏信号方向'的解释")
    elif cos < 0.7:
        print("\n  → Prefix 信号方向被 LoRA 显著改变 (cos<0.7)")
        print("    LoRA 不仅改变了信号强度，还改变了 Prefix 的影响方向")
    else:
        print(f"\n  → cos={cos:.3f}, ratio={nr:.3f}，信号方向部分保持，强度有变化")

    # Per strategy
    print("\n── Per strategy ──")
    print(f"  {'Strategy':<28s} {'N':>5s} {'cos':>8s} {'norm_ratio':>10s} {'clean_norm':>10s} {'lora_norm':>10s}")
    print("  " + "-" * 75)
    strategies_with_data = []
    for s in ALL_STRATEGIES:
        if s not in per_strategy_rows:
            continue
        rows_s = per_strategy_rows[s]
        n = sum(r.get("n_tokens", 0) for r in rows_s)
        cos_s = weighted_avg(rows_s, "mean_cosine")
        nr_s = weighted_avg(rows_s, "mean_norm_ratio")
        cn = weighted_avg(rows_s, "mean_clean_norm")
        ln = weighted_avg(rows_s, "mean_lora_norm")
        strategies_with_data.append((s, n, cos_s, nr_s, cn, ln))
        print(f"  {s:<28s} {n:>5d} {cos_s:>8.4f} {nr_s:>10.4f} {cn:>10.4f} {ln:>10.4f}")

    # 找出受 LoRA 影响最大的策略
    print("\n── LoRA 影响最大的策略（按 norm_ratio 偏离 1.0 排序）──")
    sorted_by_impact = sorted(strategies_with_data, key=lambda x: abs(x[3] - 1.0), reverse=True)
    for s, n, cos_s, nr_s, cn, ln in sorted_by_impact:
        direction = "削弱" if nr_s < 1.0 else "增强"
        print(f"  {s:<28s} norm_ratio={nr_s:.4f} ({direction}), cos={cos_s:.4f}")

    # ── 保存 ──
    if args.out:
        output = {
            "checkpoint_dir": str(checkpoint_dir),
            "split": args.split,
            "n_samples": len(all_rows),
            "overall": {
                "mean_cosine": weighted_avg(all_rows, "mean_cosine"),
                "mean_norm_ratio": weighted_avg(all_rows, "mean_norm_ratio"),
                "mean_clean_norm": weighted_avg(all_rows, "mean_clean_norm"),
                "mean_lora_norm": weighted_avg(all_rows, "mean_lora_norm"),
            },
            "per_strategy": {
                s: {
                    "n_samples": len(per_strategy_rows.get(s, [])),
                    "mean_cosine": weighted_avg(per_strategy_rows.get(s, []), "mean_cosine"),
                    "mean_norm_ratio": weighted_avg(per_strategy_rows.get(s, []), "mean_norm_ratio"),
                    "mean_clean_norm": weighted_avg(per_strategy_rows.get(s, []), "mean_clean_norm"),
                    "mean_lora_norm": weighted_avg(per_strategy_rows.get(s, []), "mean_lora_norm"),
                }
                for s in ALL_STRATEGIES
                if s in per_strategy_rows
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(f"\n结果已保存至 {args.out}")

    print()


if __name__ == "__main__":
    main()
