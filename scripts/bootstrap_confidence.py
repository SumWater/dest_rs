#!/usr/bin/env python3
"""Bootstrap 置信区间分析。

读取 strategy_eval_llm.json，计算：
  - overall accuracy 的 bootstrap 95% CI
  - 逐策略 accuracy 的 bootstrap 95% CI
  - 两个实验差异的 bootstrap 置信区间（如 B3 vs B4）

用法：
  # 单个实验
  python scripts/bootstrap_confidence.py \
      --input output/need/casino_augmented/b3_prefix_only/strategy_eval_llm.json

  # 两个实验对比
  python scripts/bootstrap_confidence.py \
      --input output/need/casino_augmented/b3_prefix_only/strategy_eval_llm.json \
      --compare output/need/casino_augmented/b4_prefix_lora/strategy_eval_llm.json \
      --label-a B3 --label-b B4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

ALL_STRATEGIES = [
    "elicit-pref", "self-need", "other-need", "no-need",
    "promote-coordination", "showing-empathy", "small-talk",
    "uv-part", "vouch-fair",
]


def load_details(path: str) -> tuple[list[int], dict[str, list[int]], int]:
    """返回 (all_corrects, per_strategy_corrects, total_samples)。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    details = data["details"]
    all_corrects = [d["correct"] for d in details]

    per = defaultdict(list)
    for d in details:
        per[d["target_strategy"]].append(d["correct"])

    return all_corrects, dict(per), len(details)


def bootstrap_ci(corrects: list[int], n_bootstrap: int = 10000, ci: float = 95.0, seed: int = 42) -> dict:
    """Bootstrap 置信区间。"""
    rng = np.random.RandomState(seed)
    n = len(corrects)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(corrects, size=n, replace=True)
        means.append(np.mean(sample))

    alpha = (100.0 - ci) / 2.0
    return {
        "mean": float(np.mean(corrects)),
        "ci_low": float(np.percentile(means, alpha)),
        "ci_high": float(np.percentile(means, 100.0 - alpha)),
        "n": n,
    }


def bootstrap_difference(
    corrects_a: list[int],
    corrects_b: list[int],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """Bootstrap 两组差异的置信区间（配对）。"""
    rng = np.random.RandomState(seed)
    n_a, n_b = len(corrects_a), len(corrects_b)
    if n_a == 0 or n_b == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_positive": 0.5}

    diffs = []
    for _ in range(n_bootstrap):
        s_a = rng.choice(corrects_a, size=n_a, replace=True)
        s_b = rng.choice(corrects_b, size=n_b, replace=True)
        diffs.append(np.mean(s_a) - np.mean(s_b))

    diffs = np.array(diffs)
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_a_greater_b": float(np.mean(diffs > 0)),
        "p_b_greater_a": float(np.mean(diffs < 0)),
    }


def print_ci(label: str, ci: dict, width: int = 28) -> None:
    print(f"  {label:<{width}s} {ci['mean']:.4f}  [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  (n={ci['n']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap 置信区间分析")
    parser.add_argument("--input", "-i", required=True, nargs="+", help="strategy_eval_llm.json 文件（可多个，按顺序合并）")
    parser.add_argument("--compare", "-c", nargs="+", default=None, help="对比的 strategy_eval_llm.json（与 --input 同等结构）")
    parser.add_argument("--label-a", default="A", help="--input 的标签")
    parser.add_argument("--label-b", default="B", help="--compare 的标签")
    parser.add_argument("--n-bootstrap", type=int, default=10000, help="bootstrap 迭代次数")
    parser.add_argument("--ci", type=float, default=95.0, help="置信水平")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="输出 JSON 路径")
    args = parser.parse_args()

    # ── 加载主实验 ──
    all_corrects: list[int] = []
    all_per: dict[str, list[int]] = defaultdict(list)
    total = 0
    for path in args.input:
        c, per, n = load_details(path)
        all_corrects.extend(c)
        for s, vals in per.items():
            all_per[s].extend(vals)
        total += n

    print(f"\n{'═' * 70}")
    print(f"  Bootstrap 分析: {args.label_a}")
    print(f"  文件: {', '.join(Path(p).name for p in args.input)}")
    print(f"  样本数: {total}")
    print(f"  Bootstrap: {args.n_bootstrap} 次, {args.ci}% CI")
    print(f"{'═' * 70}\n")

    # ── Overall ──
    overall_ci = bootstrap_ci(all_corrects, args.n_bootstrap, args.ci, args.seed)
    print("── Overall ──")
    print_ci(f"{args.label_a} accuracy", overall_ci)

    # ── Per-strategy ──
    print("\n── Per strategy ──")
    for s in ALL_STRATEGIES:
        if s in all_per:
            ci = bootstrap_ci(all_per[s], args.n_bootstrap, args.ci, args.seed)
            print_ci(f"  {s}", ci)
        else:
            print(f"  {s:<28s} (no data)")

    # ── 对比 ──
    output = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "overall_a": overall_ci,
        "per_strategy": {},
        "difference": None,
    }

    if args.compare:
        b_corrects: list[int] = []
        b_per: dict[str, list[int]] = defaultdict(list)
        b_total = 0
        for path in args.compare:
            c, per, n = load_details(path)
            b_corrects.extend(c)
            for s, vals in per.items():
                b_per[s].extend(vals)
            b_total += n

        print(f"\n{'═' * 70}")
        print(f"  Bootstrap 对比: {args.label_a} vs {args.label_b}")
        print(f"  {args.label_a}: n={total}")
        print(f"  {args.label_b}: n={b_total}")
        print(f"{'═' * 70}\n")

        b_overall = bootstrap_ci(b_corrects, args.n_bootstrap, args.ci, args.seed)
        print_ci(f"{args.label_a}", overall_ci)
        print_ci(f"{args.label_b}", b_overall)

        # ── Overall difference ──
        diff_ci = bootstrap_difference(all_corrects, b_corrects, args.n_bootstrap, args.seed)
        print(f"\n── Difference ({args.label_a} - {args.label_b}) ──")
        print(f"  mean diff: {diff_ci['mean_diff']:.4f}  [{diff_ci['ci_low']:.4f}, {diff_ci['ci_high']:.4f}]")
        print(f"  P({args.label_a} > {args.label_b}): {diff_ci['p_a_greater_b']:.4f}")
        print(f"  P({args.label_b} > {args.label_a}): {diff_ci['p_b_greater_a']:.4f}")

        # 显著性判断
        if diff_ci["ci_low"] > 0:
            print(f"  → {args.label_a} 显著优于 {args.label_b} (CI 不跨 0)")
        elif diff_ci["ci_high"] < 0:
            print(f"  → {args.label_b} 显著优于 {args.label_a} (CI 不跨 0)")
        else:
            print(f"  → 差异不显著 (CI 跨越 0)")

        output["overall_b"] = b_overall
        output["difference"] = diff_ci

        # ── Per-strategy difference ──
        print("\n── Per-strategy difference ──")
        for s in ALL_STRATEGIES:
            if s in all_per and s in b_per:
                d = bootstrap_difference(all_per[s], b_per[s], args.n_bootstrap, args.seed)
                sig = "✓ sig" if (d["ci_low"] > 0 or d["ci_high"] < 0) else ""
                print(f"  {s:<28s} Δ={d['mean_diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] {sig}")
                output["per_strategy"][s] = d

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2))
        print(f"\n结果已保存至 {args.out}")

    print()


if __name__ == "__main__":
    main()
