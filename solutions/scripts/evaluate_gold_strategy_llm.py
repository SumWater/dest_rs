#!/usr/bin/env python3
"""Evaluate the LLM strategy judge on gold human/reference utterances.

This is a sanity check for data and evaluator reliability.  It uses the same
few-shot judge implementation as scripts/evaluate_strategy_control_llm.py, but
classifies the dataset's gold target utterance instead of model generations.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
SOLUTIONS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SOLUTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTIONS_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from evaluate_strategy_control_llm import (  # noqa: E402
    classify_batch,
    load_few_shot_examples,
)
from src.casino_dataset import CasinoStrategyDataset, StrategyLabelSpace, load_split_examples  # noqa: E402
from src.config import load_config  # noqa: E402


ALL_STRATEGIES = [
    "elicit-pref", "self-need", "other-need", "no-need",
    "promote-coordination", "showing-empathy", "small-talk",
    "uv-part", "vouch-fair",
]


def load_judge(model_path: str, use_4bit: bool):
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config"):
        model.generation_config.enable_thinking = False
    if hasattr(model, "config") and hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold utterance LLM-judge calibration")
    parser.add_argument("--model-path", required=True, help="Path to Qwen judge model")
    parser.add_argument("--config", required=True, help="Training config for dataset loading")
    parser.add_argument("--dataset-tag", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--few-shot-per-strategy", type=int, default=2)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset_tag:
        cfg.dataset_tag = args.dataset_tag
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir

    train_examples = load_split_examples(cfg, "train")
    label_space = StrategyLabelSpace.fit(train_examples)
    examples = load_split_examples(cfg, args.split)
    dataset = CasinoStrategyDataset(examples, label_space)

    few_shot_examples = load_few_shot_examples(
        train_examples, label_space, args.few_shot_per_strategy
    )
    samples = []
    for item in dataset:
        if item["primary_strategy"] not in ALL_STRATEGIES:
            continue
        samples.append(
            {
                "dialogue_id": item["dialogue_id"],
                "turn_index": item["turn_index"],
                "gold_strategy": item["primary_strategy"],
                "target_strategy": item["primary_strategy"],
                "context": item["prompt"],
                "utterance": item["target"],
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break

    print(f"[gold-eval] split={args.split} samples={len(samples)}")
    print(f"[gold-eval] loading judge: {args.model_path}")
    model, tokenizer = load_judge(args.model_path, args.use_4bit)
    preds = classify_batch(model, tokenizer, samples, few_shot_examples, ALL_STRATEGIES)

    rows = []
    for sample, pred in zip(samples, preds):
        sample["pred_strategy"] = pred
        sample["correct"] = int(pred == sample["target_strategy"]) if pred else 0
        sample["raw_response"] = sample.get("raw_response", "")
        rows.append(sample)

    valid = [r for r in rows if r["pred_strategy"] is not None]
    correct = sum(r["correct"] for r in valid)
    overall_acc = correct / len(valid) if valid else 0.0
    null_count = len(rows) - len(valid)

    by_target = {}
    grouped = defaultdict(list)
    pred_dist = defaultdict(int)
    for row in rows:
        grouped[row["target_strategy"]].append(row)
        pred_dist[row["pred_strategy"]] += 1
    for strategy in ALL_STRATEGIES:
        group = [r for r in grouped[strategy] if r["pred_strategy"] is not None]
        by_target[strategy] = {
            "n": len(group),
            "accuracy": sum(r["correct"] for r in group) / len(group) if group else 0.0,
        }

    payload = {
        "mode": "gold_utterance_judge_calibration",
        "split": args.split,
        "overall_accuracy": overall_acc,
        "null_predictions": null_count,
        "total_samples": len(rows),
        "by_target_strategy": by_target,
        "predicted_distribution": dict(sorted(pred_dist.items(), key=lambda x: str(x[0]))),
        "details": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gold-eval] accuracy={correct}/{len(valid)} = {overall_acc:.4f}")
    if null_count:
        print(f"[gold-eval] null predictions={null_count}/{len(rows)}")
    print(f"[gold-eval] saved: {out}")


if __name__ == "__main__":
    main()



