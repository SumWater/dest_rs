#!/usr/bin/env python3
"""校准 LLM 评估器：对验证集的人工标注回复做策略分类，计算评估器自身的准确率上限。"""

from __future__ import annotations

import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── prompt（复用评估器模板） ──

SYSTEM = "You are an expert in negotiation dialogue analysis. Your task is to classify the negotiation strategy used in an utterance."

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

FEW_SHOT_TEMPLATE = """Here are examples of negotiation utterances and their strategies:
{few_shot}"""

CLASSIFICATION_PROMPT = """Now, given the following dialogue context and utterance, which strategy is being used?

Dialogue context:
{context}

Utterance: "{utterance}"

Which strategy does this utterance use? Reply with exactly the strategy name (one of: elicit-pref, self-need, other-need, no-need, promote-coordination, showing-empathy, small-talk, uv-part, vouch-fair)."""

ALL_STRATEGIES = [
    "elicit-pref", "self-need", "other-need", "no-need",
    "promote-coordination", "showing-empathy", "small-talk",
    "uv-part", "vouch-fair",
]


def extract_strategy(text: str) -> str | None:
    text = text.strip().lower()
    for s in ALL_STRATEGIES:
        if text == s.lower():
            return s
    for s in ALL_STRATEGIES:
        if s.lower() in text:
            return s
    pattern = r"(elicit-pref|self-need|other-need|no-need|promote-coordination|showing-empathy|small-talk|uv-part|vouch-fair)"
    m = re.search(pattern, text)
    return m.group(1) if m else None


def build_context(dialogue_item, turn_idx, context_turns=6):
    """构建对话上下文（复用 casino_dataset 的逻辑）。"""
    chat_logs = dialogue_item["chat_logs"]
    # 当前 turn 之前的对话历史
    start = max(0, turn_idx - context_turns)
    lines = []
    for i in range(start, turn_idx):
        msg = chat_logs[i]
        speaker = msg.get("speaker", msg.get("role", "unknown"))
        text = msg.get("text", msg.get("content", ""))
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def load_few_shot(train_data, n_per_strategy=2):
    """从训练集选 few-shot 样本。"""
    from random import Random
    rng = Random(42)
    by_strat = defaultdict(list)
    for item in train_data:
        for ann in item.get("annotations", []):
            lbl = ann[1] if isinstance(ann, list) and len(ann) > 1 else ann
            labels = [s.strip() for s in lbl.split(",")]
            if len(labels) != 1 or labels[0] not in ALL_STRATEGIES:
                continue
            text = ann[0] if isinstance(ann, list) else ""
            if text and text.strip():
                by_strat[labels[0]].append(text.strip())

    selected = []
    for s in ALL_STRATEGIES:
        cand = by_strat.get(s, [])
        rng.shuffle(cand)
        for t in cand[:n_per_strategy]:
            if len(t) > 200:
                t = t[:200] + "..."
            selected.append({"strategy": s, "utterance": t})
    rng.shuffle(selected)
    return selected


def main():
    model_path = "/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B"
    data_dir = "augmented_data/split"
    max_samples = 200  # 限制数量，避免跑太久

    print("Loading model...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config"):
        model.generation_config.enable_thinking = False
    if hasattr(model.config, "enable_thinking"):
        model.config.enable_thinking = False
    model.eval()
    print("Model loaded.\n")

    # 加载训练集（用于 few-shot）
    with open(f"{data_dir}/casino_train.json") as f:
        train_data = json.load(f)
    few_shot = load_few_shot(train_data)
    fs_text = "\n".join(f'Strategy: {ex["strategy"]}\nUtterance: "{ex["utterance"]}"' for ex in few_shot)

    # 加载验证集，提取单标签样本
    with open(f"{data_dir}/casino_valid.json") as f:
        valid_data = json.load(f)

    samples = []
    for item in valid_data:
        dialogue_id = item["dialogue_id"]
        for turn_idx, ann in enumerate(item.get("annotations", [])):
            lbl = ann[1] if isinstance(ann, list) and len(ann) > 1 else ann
            labels = [s.strip() for s in lbl.split(",")]
            if len(labels) != 1:
                continue
            strategy = labels[0]
            if strategy not in ALL_STRATEGIES:
                continue
            text = ann[0] if isinstance(ann, list) else ""
            if not text or not text.strip():
                continue
            ctx = build_context(item, turn_idx, context_turns=6)
            samples.append({
                "dialogue_id": dialogue_id,
                "turn_idx": turn_idx,
                "gold_strategy": strategy,
                "context": ctx,
                "utterance": text.strip(),
            })

    if max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    print(f"Valid set single-label turns: {len(samples)} (evaluating up to {max_samples or len(samples)})")
    label_dist = Counter(s["gold_strategy"] for s in samples)
    for s in ALL_STRATEGIES:
        print(f"  {s:<25} {label_dist.get(s, 0):>4}")
    print()

    # 分类
    correct = 0
    total = 0
    null = 0
    per_strat = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = defaultdict(int)

    for i, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{STRATEGY_DEFINITIONS}\n\n{FEW_SHOT_TEMPLATE.format(few_shot=fs_text)}\n\n{CLASSIFICATION_PROMPT.format(context=sample['context'][:1500], utterance=sample['utterance'][:500])}"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True)
        pred = extract_strategy(response)

        sample["pred"] = pred

        total += 1
        gold = sample["gold_strategy"]
        per_strat[gold]["total"] += 1
        if pred is None:
            null += 1
        elif pred == gold:
            correct += 1
            per_strat[gold]["correct"] += 1
        else:
            confusion[f"{gold} → {pred}"] += 1

        if i < 5:
            print(f"[sample {i}] gold={gold} | pred={pred} | raw={repr(response[:200])}")
        if (i + 1) % 50 == 0:
            print(f"  progress: {i+1}/{len(samples)}, current acc={correct/total:.4f}")

    print(f"\n{'═' * 60}")
    print(f"  评估器校准结果")
    print(f"{'═' * 60}")
    print(f"  Total samples:       {total}")
    print(f"  Correct:             {correct}")
    print(f"  Null predictions:    {null}")
    print(f"  Evaluator accuracy:  {correct}/{total-null} = {correct/(total-null)*100:.1f}% (excl. null)")
    print(f"  Evaluator accuracy:  {correct}/{total} = {correct/total*100:.1f}% (incl. null)")
    print()
    print(f"  Per-strategy accuracy:")
    for s in ALL_STRATEGIES:
        st = per_strat[s]
        acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"    {s:<25} {st['correct']:>3}/{st['total']:>3} = {acc:.1f}%")
    print()
    print(f"  Top confusions:")
    for pair, cnt in sorted(confusion.items(), key=lambda x: -x[1])[:15]:
        print(f"    {pair}: {cnt}")

    # 保存逐条预测结果
    out_path = Path("output/evaluator_calibration_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for s in samples:
        results.append({
            "dialogue_id": s["dialogue_id"],
            "gold": s["gold_strategy"],
            "pred": s.get("pred"),
            "utterance": s["utterance"][:120],
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Per-sample results saved to {out_path}")


if __name__ == "__main__":
    main()
