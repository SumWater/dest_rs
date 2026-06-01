"""
Diagnostic 1: How well does the LLM classifier perform on HUMAN-WRITTEN gold targets?
This tells us the ceiling of the evaluation task.

Diagnostic 2: How many distinct strategies can humans even distinguish?
Group strategies into coarse clusters and re-evaluate.

Usage:
    python scripts/diagnose_baseline.py \
        --model-path /path/to/Qwen3-8B \
        --config configs/v3/b6v3_dest_rs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.casino_dataset import load_split_examples, StrategyLabelSpace
from src.config import load_config

# Reuse the evaluator's prompt and logic
from evaluate_strategy_control_llm import (
    SYSTEM, STRATEGY_DEFINITIONS, FEW_SHOT_HEADER, CLASSIFICATION_PROMPT,
    build_message, extract_strategy, classify_batch,
)

ALL_STRATEGIES = [
    "elicit-pref", "self-need", "other-need", "no-need",
    "promote-coordination", "showing-empathy", "small-talk",
    "uv-part", "vouch-fair",
]

# Coarse strategy clusters
COARSE_MAP = {
    "elicit-pref": "information-seeking",
    "self-need": "need-expression",
    "other-need": "need-expression",
    "no-need": "need-expression",
    "promote-coordination": "collaboration",
    "showing-empathy": "rapport",
    "small-talk": "rapport",
    "uv-part": "need-expression",
    "vouch-fair": "collaboration",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded.\n")

    # Load data
    cfg = load_config(args.config)
    train_examples = load_split_examples(cfg, "train")
    valid_examples = load_split_examples(cfg, "valid")
    test_examples = load_split_examples(cfg, "test")

    from evaluate_strategy_control_llm import load_few_shot_examples
    label_space = StrategyLabelSpace.fit(train_examples)
    few_shot = load_few_shot_examples(train_examples, label_space, samples_per_strategy=2)

    # ── Diagnostic 1: LLM classifier on HUMAN text ──
    print("=" * 60)
    print(" Diagnostic 1: LLM classifier accuracy on HUMAN-WRITTEN gold text")
    print("=" * 60)

    for split_name, examples in [("train", train_examples[:50]),
                                  ("valid", valid_examples),
                                  ("test", test_examples)]:
        samples = []
        for ex in examples:
            samples.append({
                "dialogue_id": "?",
                "turn_index": 0,
                "gold_strategy": ex.primary_strategy,
                "target_strategy": ex.primary_strategy,
                "context": ex.prompt,
                "utterance": ex.target,
            })

        preds = classify_batch(model, tokenizer, samples, few_shot, ALL_STRATEGIES, max_new_tokens=64)
        correct = sum(1 for s, p in zip(samples, preds) if p == s["target_strategy"])
        nulls = sum(1 for p in preds if p is None)
        print(f"  {split_name}: {correct}/{len(samples)} = {correct/len(samples):.4f}  (null: {nulls})")

    # ── Diagnostic 2: Coarse-grained strategy clustering ──
    print()
    print("=" * 60)
    print(" Diagnostic 2: Coarse-grained (4-class) re-evaluation")
    print("=" * 60)

    # Load the LLM eval results
    eval_file = Path("outputs/v3/llm_strategy_eval.json")
    if eval_file.exists():
        with eval_file.open() as f:
            eval_data = json.load(f)

        for level_name, mapping in [
            ("4-class", {
                "elicit-pref": "elicit-pref",  # keep separate (asking questions)
                "self-need": "need-expression",
                "other-need": "need-expression",
                "no-need": "need-expression",
                "promote-coordination": "collaboration",
                "showing-empathy": "rapport",
                "small-talk": "rapport",
                "uv-part": "need-expression",
                "vouch-fair": "collaboration",
            }),
            ("3-class", {
                "elicit-pref": "information",
                "self-need": "need-expression",
                "other-need": "need-expression",
                "no-need": "need-expression",
                "promote-coordination": "collaboration",
                "showing-empathy": "rapport",
                "small-talk": "rapport",
                "uv-part": "need-expression",
                "vouch-fair": "collaboration",
            }),
        ]:
            correct = 0
            total = 0
            for row in eval_data["details"]:
                if row["pred_strategy"] is None:
                    continue
                mapped_target = mapping.get(row["target_strategy"], row["target_strategy"])
                mapped_pred = mapping.get(row["pred_strategy"], row["pred_strategy"])
                if mapped_target == mapped_pred:
                    correct += 1
                total += 1

            print(f"  {level_name}: {correct}/{total} = {correct/total:.4f}")

        # By experiment with coarse mapping
        print()
        print("  By experiment (4-class):")
        by_exp = defaultdict(list)
        for row in eval_data["details"]:
            src = row.get("source_file", "")
            exp = src.split("/")[-2] if "/" in src else src
            if row["pred_strategy"] is None:
                continue
            mapping = {
                "elicit-pref": "elicit-pref",
                "self-need": "need-expression",
                "other-need": "need-expression",
                "no-need": "need-expression",
                "promote-coordination": "collaboration",
                "showing-empathy": "rapport",
                "small-talk": "rapport",
                "uv-part": "need-expression",
                "vouch-fair": "collaboration",
            }
            mapped_target = mapping.get(row["target_strategy"], row["target_strategy"])
            mapped_pred = mapping.get(row["pred_strategy"], row["pred_strategy"])
            by_exp[exp].append(int(mapped_target == mapped_pred))

        for exp in sorted(by_exp.keys()):
            vals = by_exp[exp]
            print(f"    {exp}: {sum(vals)/len(vals):.4f}")

    # ── Diagnostic 3: Per-strategy accuracy on HUMAN text ──
    print()
    print("=" * 60)
    print(" Diagnostic 3: Per-strategy LLM accuracy on HUMAN text (valid + test)")
    print("=" * 60)

    all_human = []
    for ex in valid_examples + test_examples:
        all_human.append({
            "dialogue_id": "?",
            "turn_index": 0,
            "gold_strategy": ex.primary_strategy,
            "target_strategy": ex.primary_strategy,
            "context": ex.prompt,
            "utterance": ex.target,
        })

    preds = classify_batch(model, tokenizer, all_human, few_shot, ALL_STRATEGIES, max_new_tokens=64)

    by_strat = defaultdict(list)
    for s, p in zip(all_human, preds):
        by_strat[s["target_strategy"]].append(int(p == s["target_strategy"]))

    print(f"{'Strategy':<25s} {'N':>5s} {'LLM Acc':>10s}")
    print("-" * 42)
    for s in ALL_STRATEGIES:
        vals = by_strat.get(s, [])
        acc = sum(vals) / len(vals) if vals else 0.0
        print(f"{s:<25s} {len(vals):>5d} {acc:>10.4f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
