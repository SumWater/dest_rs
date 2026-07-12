"""Aggregate B3/B4/B9 multi-label judge results across seeds."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

SEEDS = (42, 43, 44)
EXPERIMENTS = ("b3_prefix_only", "b4_prefix_lora", "b9_prefix_then_lora")
DISPLAY = {"b3_prefix_only": "B3", "b4_prefix_lora": "B4", "b9_prefix_then_lora": "B9"}


def avg_sd(values):
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0, "values": values}


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def generation_stats(path):
    responses = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            generated = record.get("generated_by_strategy") or {}
            responses.extend(str(v) for v in generated.values())
    normalized = [norm(x) for x in responses]
    counts = Counter(normalized)
    repeated_samples = sum(n for text, n in counts.items() if text and n > 1)
    tokens = [re.findall(r"[a-z0-9']+", x.lower()) for x in responses]
    unigrams = [t for row in tokens for t in row]
    bigrams = [tuple(row[i:i+2]) for row in tokens for i in range(len(row) - 1)]
    return {
        "samples": len(responses),
        "exact_repetition_rate": repeated_samples / len(responses) if responses else 0,
        "duplicate_group_count": sum(n > 1 for n in counts.values()),
        "distinct_1": len(set(unigrams)) / len(unigrams) if unigrams else 0,
        "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else 0,
        "average_words": mean(map(len, tokens)) if tokens else 0,
    }


def main():
    root = Path("output/need")
    runs = []
    parse_failures = []
    for seed in SEEDS:
        for exp in EXPERIMENTS:
            directory = root / f"casino_augmented_new_fix_seed{seed}" / exp
            result = json.loads((directory / "strategy_eval_multilabel.json").read_text(encoding="utf-8"))
            train_metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            best_epoch = train_metrics["best_epoch"]
            best = next(x for x in train_metrics["history"] if x["epoch"] == best_epoch)
            m = result["metrics"]
            gen = generation_stats(directory / "swap_samples_valid.jsonl")
            row = {
                "experiment": DISPLAY[exp], "seed": seed,
                "target_presence": m["target_strategy_presence"],
                "primary_accuracy": m["primary_strategy_accuracy"],
                "macro_f1": m["macro_f1"],
                "off_target_count": m["mean_off_target_strategy_count"],
                "parse_failures": m["parse_failures"],
                "valid_ppl": best["valid_ppl"], "test_ppl": best["test_ppl"],
                **gen,
                "per_class": m["per_class"],
                "prediction_distribution": m["prediction_distribution"],
                "primary_prediction_distribution": m["primary_prediction_distribution"],
            }
            runs.append(row)
            parse_failures.extend(
                {"experiment": DISPLAY[exp], "seed": seed, **d}
                for d in result["details"] if d.get("parse_failed")
            )

    scalar = ["target_presence", "primary_accuracy", "macro_f1", "off_target_count",
              "valid_ppl", "test_ppl", "exact_repetition_rate", "distinct_1",
              "distinct_2", "average_words"]
    summary = {}
    for exp in ("B3", "B4", "B9"):
        selected = [r for r in runs if r["experiment"] == exp]
        summary[exp] = {key: avg_sd([r[key] for r in selected]) for key in scalar}
        labels = selected[0]["per_class"]
        summary[exp]["per_class_f1"] = {
            label: avg_sd([r["per_class"][label]["f1"] for r in selected]) for label in labels
        }
        total_predictions = Counter()
        primary_predictions = Counter()
        for r in selected:
            total_predictions.update(r["prediction_distribution"])
            primary_predictions.update(r["primary_prediction_distribution"])
        summary[exp]["prediction_distribution_total"] = dict(total_predictions)
        summary[exp]["primary_prediction_distribution_total"] = dict(primary_predictions)

    out = Path("reports")
    out.mkdir(exist_ok=True)
    payload = {"summary": summary, "runs": runs, "parse_failures": parse_failures}
    (out / "multilabel_baseline_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "multilabel_baseline_runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["experiment", "seed", *scalar, "parse_failures"]
        w = csv.DictWriter(f, fields, extrasaction="ignore"); w.writeheader(); w.writerows(runs)

    def cell(metric, exp, pct=False):
        x = summary[exp][metric]
        scale = 100 if pct else 1
        return f"{x['mean']*scale:.2f} ± {x['std']*scale:.2f}"
    lines = ["# B3/B4/B9 多标签重评估", "", "三随机种子均值 ± 样本标准差。", "",
             "| 实验 | Target Presence (%) | Primary Accuracy (%) | Macro-F1 (%) | Off-target Count | Test PPL | Exact Repetition (%) | Distinct-1 | Distinct-2 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exp in ("B3", "B4", "B9"):
        lines.append(f"| {exp} | {cell('target_presence', exp, True)} | {cell('primary_accuracy', exp, True)} | {cell('macro_f1', exp, True)} | {cell('off_target_count', exp)} | {cell('test_ppl', exp)} | {cell('exact_repetition_rate', exp, True)} | {cell('distinct_1', exp)} | {cell('distinct_2', exp)} |")
    lines += ["", "## 每类 F1 (%)", "", "| 策略 | B3 | B4 | B9 |", "|---|---:|---:|---:|"]
    for label in summary["B3"]["per_class_f1"]:
        vals = []
        for exp in ("B3", "B4", "B9"):
            x = summary[exp]["per_class_f1"][label]
            vals.append(f"{x['mean']*100:.2f} ± {x['std']*100:.2f}")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines += ["", f"解析失败：{len(parse_failures)}/2430。失败样本保留在 JSON 报告中，不静默删除。", ""]
    (out / "multilabel_baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
