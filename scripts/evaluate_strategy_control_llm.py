"""
LLM-based strategy control evaluator.

Uses Qwen3-8B with few-shot prompting to classify generated utterances
by negotiation strategy. Replaces the BOW linear evaluator.

Usage:
    python scripts/evaluate_strategy_control_llm.py \
        --model-path /path/to/Qwen3-8B \
        --jsonl outputs/v3/b6v3_dest_rs/swap_samples_valid.jsonl \
        --out outputs/v3/b6v3_dest_rs/strategy_eval_llm.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.casino_dataset import load_split_examples, StrategyLabelSpace
from src.config import load_config

# ── few-shot prompt template ──────────────────────────────────────────

SYSTEM = (
    "You are an expert in negotiation dialogue analysis. "
    "Your task is to classify the negotiation strategy used in an utterance."
)

STRATEGY_DEFINITIONS = """Negotiation strategies (from CaSiNo):
- elicit-pref: Ask about the other party's preferences, priorities, or situation
- self-need: Express or emphasize your own needs, wants, or requirements
- other-need: Acknowledge, discuss, or accommodate the other party's needs
- no-need: Downplay or deny needing something; signal flexibility
- promote-coordination: Propose collaboration, compromise, or working together
- showing-empathy: Express understanding, support, or emotional connection
- small-talk: Casual conversation, greetings, chit-chat unrelated to negotiation
- uv-part: Emphasize unique value of items; justify why something matters to you
- vouch-fair: Appeal to fairness, equity, or balanced outcomes"""

FEW_SHOT_HEADER = """Here are examples of negotiation utterances and their strategies:"""

CLASSIFICATION_PROMPT = """Now, given the following dialogue context and utterance, which strategy is being used?

Dialogue context:
{context}

Utterance: "{utterance}"

Which strategy does this utterance use? Reply with exactly the strategy name (one of: elicit-pref, self-need, other-need, no-need, promote-coordination, showing-empathy, small-talk, uv-part, vouch-fair)."""


def build_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def extract_strategy(text: str, strategies: list[str]) -> str | None:
    """Extract strategy label from model output."""
    text = text.strip().lower()
    # Try exact match first
    for s in strategies:
        if text == s.lower():
            return s
    # Try finding strategy name in text
    for s in strategies:
        if s.lower() in text:
            return s
    # Try regex
    pattern = r"(elicit-pref|self-need|other-need|no-need|promote-coordination|showing-empathy|small-talk|uv-part|vouch-fair)"
    m = re.search(pattern, text)
    if m:
        return m.group(1)
    return None


def load_few_shot_examples(train_examples, label_space, samples_per_strategy: int = 2):
    """Pick representative few-shot examples from training data, balanced by strategy."""
    from random import Random
    rng = Random(42)

    by_strategy = defaultdict(list)
    for ex in train_examples:
        label = ex.primary_strategy
        if label in label_space.label_to_id:
            by_strategy[label].append(ex)

    selected = []
    for label in sorted(by_strategy.keys()):
        candidates = by_strategy[label]
        rng.shuffle(candidates)
        for ex in candidates[:samples_per_strategy]:
            # Truncate long targets
            target = ex.target
            if len(target) > 200:
                target = target[:200] + "..."
            selected.append({"strategy": label, "utterance": target})

    rng.shuffle(selected)
    return selected


def classify_batch(
    model,
    tokenizer,
    samples: list[dict],
    few_shot_examples: list[dict],
    strategies: list[str],
    max_new_tokens: int = 128,
) -> list[str | None]:
    """Classify a batch of (context, utterance) pairs."""
    results: list[str | None] = []

    for sample in samples:
        context = sample["context"]
        utterance = sample["utterance"]

        # Build messages
        messages = [build_message("system", SYSTEM)]

        # Add strategy definitions
        user_parts = [STRATEGY_DEFINITIONS, FEW_SHOT_HEADER]
        for ex in few_shot_examples:
            user_parts.append(f'Strategy: {ex["strategy"]}\nUtterance: "{ex["utterance"]}"')

        user_parts.append(
            CLASSIFICATION_PROMPT.format(context=context[:1500], utterance=utterance[:500])
        )
        user_msg = "\n\n".join(user_parts)

        messages.append(build_message("user", user_msg))

        # Apply chat template (disable Qwen3 thinking mode)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.size(1):],
            skip_special_tokens=True,
        )
        label = extract_strategy(response, strategies)
        # Store raw response for debugging
        sample["raw_response"] = response
        results.append(label)

        # Print first 3 for debugging
        if len([r for r in results if r is not None]) <= 0 and len(results) <= 3:
            print(f"\n[DEBUG] raw response: {repr(response[:300])}")
            print(f"[DEBUG] parsed label: {label}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based strategy control evaluator")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3-8B model")
    parser.add_argument("--config", help="Training config (to load dataset)")
    parser.add_argument("--jsonl", required=True, nargs="+", help="swap_samples JSONL file(s)")
    parser.add_argument("--out", default=None, help="Output JSON file for detailed results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--few-shot-per-strategy", type=int, default=2)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-think", action="store_true", default=True, help="Disable Qwen3 thinking mode")
    args = parser.parse_args()

    # ── load model ──
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
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable Qwen3 thinking mode
    if args.no_think and hasattr(model, "generation_config"):
        model.generation_config.enable_thinking = False
    if hasattr(model, "config") and hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False

    model.eval()
    print("Model loaded.\n")

    # ── get strategies ──
    all_strategies = [
        "elicit-pref", "self-need", "other-need", "no-need",
        "promote-coordination", "showing-empathy", "small-talk",
        "uv-part", "vouch-fair",
    ]

    # ── load few-shot examples from training data ──
    few_shot_examples = []
    if args.config:
        cfg = load_config(args.config)
        train_examples = load_split_examples(cfg, "train")
        label_space = StrategyLabelSpace.fit(train_examples)
        few_shot_examples = load_few_shot_examples(
            train_examples, label_space, args.few_shot_per_strategy
        )
        print(f"Loaded {len(few_shot_examples)} few-shot examples from training data.\n")
    else:
        print("Warning: no --config provided, running zero-shot.\n")

    # ── evaluate ──
    all_rows = []
    for jsonl_path in args.jsonl:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            print(f"[SKIP] {jsonl_path} not found")
            continue

        print(f"Evaluating {jsonl_path}...")
        samples_to_classify = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("prompt") or ""
                for target_strategy, generated in (
                    record.get("generated_by_strategy") or {}
                ).items():
                    if target_strategy not in all_strategies:
                        continue
                    generated = generated or ""
                    if len(generated.strip()) < 2:
                        continue
                    samples_to_classify.append(
                        {
                            "dialogue_id": record.get("dialogue_id"),
                            "turn_index": record.get("turn_index"),
                            "gold_strategy": record.get("gold_strategy"),
                            "target_strategy": target_strategy,
                            "context": prompt,
                            "utterance": generated,
                        }
                    )

        if args.max_samples:
            samples_to_classify = samples_to_classify[: args.max_samples]

        preds = classify_batch(
            model, tokenizer, samples_to_classify, few_shot_examples, all_strategies
        )

        for sample, pred in zip(samples_to_classify, preds):
            sample["pred_strategy"] = pred
            sample["correct"] = int(pred == sample["target_strategy"]) if pred else 0
            sample["source_file"] = str(jsonl_path)
            sample["raw_response"] = sample.get("raw_response", "")
            all_rows.append(sample)

    # ── compute metrics ──
    print(f"\nTotal classified: {len(all_rows)}")

    # Overall
    valid = [r for r in all_rows if r["pred_strategy"] is not None]
    null_count = len(all_rows) - len(valid)
    correct = sum(r["correct"] for r in valid)
    overall_acc = correct / len(valid) if valid else 0.0
    print(f"Overall accuracy: {correct}/{len(valid)} = {overall_acc:.4f}")
    if null_count:
        print(f"  (null predictions: {null_count}/{len(all_rows)})")

    # By turn
    by_turn = defaultdict(list)
    for r in all_rows:
        by_turn[r["turn_index"]].append(r)

    print("\n── By turn ──")
    print(f"{'Turn':<6s} {'N':>5s} {'Correct':>8s} {'Accuracy':>10s}")
    print("-" * 32)
    for turn in sorted(by_turn.keys()):
        rows = by_turn[turn]
        valid_turn = [r for r in rows if r["pred_strategy"] is not None]
        correct_turn = sum(r["correct"] for r in valid_turn)
        acc = correct_turn / len(valid_turn) if valid_turn else 0.0
        print(f"{turn:<6d} {len(valid_turn):>5d} {correct_turn:>8d} {acc:>10.4f}")

    # Per-target-strategy
    by_target = defaultdict(list)
    for r in all_rows:
        by_target[r["target_strategy"]].append(r)

    print("\n── Per target strategy ──")
    print(f"{'Strategy':<25s} {'N':>5s} {'Accuracy':>10s}")
    print("-" * 42)
    for s in all_strategies:
        rows = by_target.get(s, [])
        valid_s = [r for r in rows if r["pred_strategy"] is not None]
        acc = sum(r["correct"] for r in valid_s) / len(valid_s) if valid_s else 0.0
        print(f"{s:<25s} {len(valid_s):>5d} {acc:>10.4f}")

    # By source file
    if len(args.jsonl) > 1:
        print("\n── By experiment ──")
        for jsonl_path in args.jsonl:
            path_str = str(jsonl_path)
            exp_rows = [r for r in all_rows if r.get("source_file") == path_str]
            valid_exp = [r for r in exp_rows if r["pred_strategy"] is not None]
            acc = sum(r["correct"] for r in valid_exp) / len(valid_exp) if valid_exp else 0.0
            print(f"  {Path(jsonl_path).parent.name}: {acc:.4f} ({len(valid_exp)} samples)")

    # ── confusion matrix summary ──
    print("\n── Top confusions (pred → target mismatches) ──")
    confusion = defaultdict(int)
    for r in valid:
        if not r["correct"] and r["pred_strategy"]:
            pair = f"{r['target_strategy']} → {r['pred_strategy']}"
            confusion[pair] += 1
    for pair, count in sorted(confusion.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pair}: {count}")

    # ── save ──
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "overall_accuracy": overall_acc,
            "null_predictions": null_count,
            "total_samples": len(all_rows),
            "by_turn": {
                str(t): {
                    "n": len([r for r in by_turn[t] if r["pred_strategy"] is not None]),
                    "accuracy": (
                        sum(r["correct"] for r in [x for x in by_turn[t] if x["pred_strategy"] is not None])
                        / len([x for x in by_turn[t] if x["pred_strategy"] is not None])
                        if [x for x in by_turn[t] if x["pred_strategy"] is not None]
                        else 0.0
                    ),
                }
                for t in sorted(by_turn.keys())
            },
            "details": all_rows,
        }
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
