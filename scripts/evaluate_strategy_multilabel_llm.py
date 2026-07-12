"""Multi-label Qwen3 judge for existing controlled-generation JSONL files.

The gold label set for a controlled generation is the requested target strategy.
The judge may return multiple strategies plus one primary strategy, allowing target
presence and strategy mixing to be measured separately.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.strategy_labels import CANONICAL_LABELS


STRATEGIES = list(CANONICAL_LABELS)

DEFINITIONS = {
    "small-talk": "social greeting or casual conversation not directly negotiating needs",
    "self-need": "establishes the current speaker's own personal need or reason for an item",
    "other-need": "establishes an item need for someone associated with the current speaker other than the speaker, such as the speaker's children, family, friends, or group members",
    "no-need": "states that the current speaker does not need, has low need for, or already has enough of an item; concession alone is insufficient",
    "uv-part": "undervaluing the partner's allocation or arguing that the partner needs an item less",
    "elicit-pref": "attempts to discover the partner's preference order or item priorities; generic questions are insufficient",
    "showing-empathy": "positively acknowledges the partner's personal context; a context-free formulaic acknowledgment is insufficient",
    "promote-coordination": "promotes an explicit trade, mutual concession, exchange, or joint effort to reach a deal",
    "vouch-fair": "appeals to fairness or calls out an allocation imbalance for personal benefit; compromise alone is insufficient",
}

SYSTEM = """You are an expert annotator of negotiation dialogue strategies.
An utterance may express zero, one, or multiple strategies. Identify every strategy
that is clearly expressed, then choose the single most central strategy as primary.
Do not infer a strategy only from the dialogue context; it must be expressed in the
candidate utterance. Return JSON only."""


def prompt(context: str, utterance: str) -> str:
    definitions = "\n".join(f"- {k}: {v}" for k, v in DEFINITIONS.items())
    return f"""Strategy definitions:
{definitions}

Dialogue context:
{context}

Candidate utterance:
{utterance}

Return exactly one JSON object with this schema:
{{"present_strategies": ["strategy-name"], "primary_strategy": "strategy-name"}}

Rules:
1. present_strategies may contain multiple labels, but only labels clearly expressed.
2. Use [] and null if no listed strategy is expressed.
3. primary_strategy must be one member of present_strategies.
4. Use only the nine strategy names listed above."""


def context_from_record(record: dict, blind_profile: bool = False) -> str:
    # Existing swap files store the full model prompt; it includes profile and history.
    value = str(record.get("prompt") or record.get("context") or "")
    if blind_profile:
        value = re.sub(
            r"\n*Target speaker private profile:\n.*?(?=\nDialogue history:)",
            "",
            value,
            flags=re.S,
        )
    return value.rsplit("Response:", 1)[0].strip()


def parse_answer(text: str) -> tuple[list[str], str | None]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if not match:
        return [], None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [], None
    values = data.get("present_strategies", [])
    if isinstance(values, str):
        values = [values]
    present = []
    for value in values if isinstance(values, list) else []:
        label = str(value).strip().lower()
        if label in STRATEGIES and label not in present:
            present.append(label)
    primary = data.get("primary_strategy")
    primary = str(primary).strip().lower() if primary is not None else None
    if primary not in STRATEGIES:
        primary = None
    if primary and primary not in present:
        present.append(primary)
    return present, primary


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    item["source_file"] = str(path)
                    records.append(item)
    return records


def expand_records(records: list[dict]) -> list[dict]:
    expanded = []
    for r in records:
        generated = r.get("generated_by_strategy")
        if isinstance(generated, dict):
            for target, utterance in generated.items():
                if target in STRATEGIES:
                    expanded.append({**r, "target_strategy": target, "utterance": str(utterance)})
        elif r.get("target_strategy") in STRATEGIES:
            expanded.append({**r, "utterance": str(r.get("utterance") or r.get("generated") or "")})
    return expanded


def compute_metrics(details: list[dict]) -> dict:
    per_class = {}
    f1_values = []
    for strategy in STRATEGIES:
        tp = sum(d["target_strategy"] == strategy and strategy in d["present_strategies"] for d in details)
        fp = sum(d["target_strategy"] != strategy and strategy in d["present_strategies"] for d in details)
        fn = sum(d["target_strategy"] == strategy and strategy not in d["present_strategies"] for d in details)
        support = sum(d["target_strategy"] == strategy for d in details)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[strategy] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": support, "tp": tp, "fp": fp, "fn": fn,
        }
    prediction_counts = Counter(s for d in details for s in d["present_strategies"])
    primary_counts = Counter(d["primary_strategy"] or "null" for d in details)
    n = len(details)
    return {
        "total_samples": n,
        "parse_failures": sum(d["parse_failed"] for d in details),
        "target_strategy_presence": sum(d["target_present"] for d in details) / n if n else 0,
        "primary_strategy_accuracy": sum(d["primary_correct"] for d in details) / n if n else 0,
        "macro_f1": sum(f1_values) / len(f1_values) if n else 0,
        "per_class": per_class,
        "mean_off_target_strategy_count": (
            sum(d["off_target_strategy_count"] for d in details) / n if n else 0
        ),
        "prediction_distribution": dict(prediction_counts),
        "primary_prediction_distribution": dict(primary_counts),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--jsonl", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--blind-profile", action="store_true",
                   help="Remove private profile from judge context for a blinded ablation")
    args = p.parse_args()

    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} gpu_count={torch.cuda.device_count()}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Refusing to run the 8B judge on CPU. "
            "Activate the original training environment and check CUDA_VISIBLE_DEVICES."
        )
    print(f"gpu_0={torch.cuda.get_device_name(0)}", flush=True)

    records = expand_records(load_records(args.jsonl))
    if args.max_samples is not None:
        records = records[: args.max_samples]

    done = []
    if args.resume and args.out.exists():
        done = json.loads(args.out.read_text(encoding="utf-8")).get("details", [])
    start = len(done)

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    print(f"model_device_map={getattr(model, 'hf_device_map', None)}", flush=True)
    device_map = getattr(model, "hf_device_map", {}) or {}
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"Model was partially offloaded to CPU/disk: {device_map}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    details = list(done)
    for index, record in enumerate(records[start:], start=start):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt(context_from_record(record, args.blind_profile), record["utterance"])},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        present, primary = parse_answer(raw)
        target = record["target_strategy"]
        detail = {
            "dialogue_id": record.get("dialogue_id"),
            "turn_index": record.get("turn_index"),
            "target_strategy": target,
            "utterance": record["utterance"],
            "present_strategies": present,
            "primary_strategy": primary,
            "target_present": int(target in present),
            "primary_correct": int(primary == target),
            "off_target_strategies": [s for s in present if s != target],
            "off_target_strategy_count": sum(s != target for s in present),
            "parse_failed": int(not present and primary is None and "{}" not in raw),
            "raw_response": raw,
            "source_file": record.get("source_file"),
        }
        for field in ("candidate_rank", "candidate_seed", "sequence_logprob", "token_count"):
            if field in record:
                detail[field] = record[field]
        details.append(detail)
        if (index + 1) % 25 == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"metrics": compute_metrics(details), "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{index + 1}/{len(records)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"metrics": compute_metrics(details), "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(compute_metrics(details), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
