#!/usr/bin/env python3
"""B1 baseline: frozen Qwen3-8B with prompt-based strategy control.
No training, no PEFT modules. Strategy specified via text instruction.
"""

from __future__ import annotations

import json, re, sys, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ALL_STRATEGIES = [
    "elicit-pref", "self-need", "other-need", "no-need",
    "promote-coordination", "showing-empathy", "small-talk",
    "uv-part", "vouch-fair",
]

STRATEGY_DESC = {
    "elicit-pref": "Discover the negotiation partner's preference order or item priorities",
    "self-need": "Establish your own personal need or reason for an item",
    "other-need": "Establish an item need for your children, family, friends, group, or companions",
    "no-need": "State that you do not need, have low need for, or have enough of an item",
    "promote-coordination": "Promote a trade, mutual concession, exchange, or joint deal",
    "showing-empathy": "Positively acknowledge the negotiation partner's personal context",
    "small-talk": "Use social conversation outside negotiation and item allocation",
    "uv-part": "Undervalue or question the negotiation partner's need for an item",
    "vouch-fair": "Appeal to fairness or call out an allocation imbalance",
}


def build_instruction_prompt(dialogue_context: str, strategy: str) -> str:
    desc = STRATEGY_DESC.get(strategy, strategy)
    return (
        f"You are participating in a negotiation dialogue.\n\n"
        f"Dialogue so far:\n{dialogue_context}\n\n"
        f"As the next speaker, use the following negotiation strategy: {strategy} ({desc}).\n"
        f"Reply with exactly one utterance."
    )


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    """Token-by-token greedy generation, same as B2-B9 greedy_generate."""
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    generated = []
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        token_id = int(next_token.item())
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        next_token_t = next_token.to(model.device)
        input_ids = torch.cat([input_ids, next_token_t.unsqueeze(0)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=model.device)],
            dim=1,
        )

    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Strip <think> tags (Qwen3 outputs these for instruction prompts)
    result = re.sub(r'<\s*think\s*>.*?<\s*/\s*think\s*>', '', result, flags=re.DOTALL | re.IGNORECASE).strip()
    return result if result else "[empty]"


def main():
    model_path = "/home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B"
    data_path = "augmented_data/split/casino_valid.json"
    output_path = "output/need/casino_augmented_fix_b6/b1_frozen_base/swap_samples_valid.jsonl"
    num_examples = 30

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading base model (no PEFT)...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded.\n")

    # Load validation data
    with open(data_path) as f:
        valid_data = json.load(f)

    # Build context + gold strategy pairs (single-label only)
    samples = []
    for item in valid_data:
        dialogue_id = item["dialogue_id"]
        chat_logs = item["chat_logs"]
        for turn_idx, ann in enumerate(item.get("annotations", [])):
            lbl = ann[1] if isinstance(ann, list) and len(ann) > 1 else ann
            labels = [s.strip() for s in lbl.split(",")]
            if len(labels) != 1 or labels[0] not in ALL_STRATEGIES:
                continue
            text = ann[0] if isinstance(ann, list) else ""
            if not text or not text.strip():
                continue
            # Build context from previous turns
            start = max(0, turn_idx - 6)
            ctx_lines = []
            for i in range(start, turn_idx):
                msg = chat_logs[i]
                speaker = msg.get("speaker", msg.get("role", "unknown"))
                t = msg.get("text", msg.get("content", ""))
                ctx_lines.append(f"{speaker}: {t}")
            ctx = "\n".join(ctx_lines)
            samples.append({
                "dialogue_id": dialogue_id,
                "turn_index": turn_idx,
                "gold_strategy": labels[0],
                "context": ctx,
                "gold_text": text.strip(),
            })

    samples = samples[:num_examples]
    print(f"Generating for {len(samples)} validation samples × 9 strategies = {len(samples)*9} responses\n")

    records = []
    for i, sample in enumerate(samples):
        generated_by_strategy = {}
        for strategy in ALL_STRATEGIES:
            prompt = build_instruction_prompt(sample["context"], strategy)
            gen = generate(model, tokenizer, prompt)
            generated_by_strategy[strategy] = gen

        records.append({
            "split": "valid",
            "dialogue_id": sample["dialogue_id"],
            "turn_index": sample["turn_index"],
            "gold_strategy": sample["gold_strategy"],
            "prompt": sample["context"],
            "generated_by_strategy": generated_by_strategy,
        })
        if (i + 1) % 5 == 0:
            print(f"  progress: {i+1}/{len(samples)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nSaved to {output_path}")

    # Save metrics stub
    import os
    metrics = {
        "history": [{"epoch": 0, "train_loss": 0, "valid_ppl": "N/A (frozen base)", "test_ppl": "N/A"}],
        "best_epoch": 0,
        "best_valid_loss": float("inf"),
        "note": "B1 frozen base model — no PEFT training",
    }
    mf = os.path.join(os.path.dirname(output_path), "metrics.json")
    with open(mf, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone. Now run LLM evaluation:")
    print(f"  python scripts/evaluate_strategy_control_llm.py \\")
    print(f"    --model-path /home/amax/PycharmProjects/AINegoProject/src/Models/LLM/Qwen3-8B \\")
    print(f"    --config configs/b2_lora_only.json \\")
    print(f"    --dataset-tag casino_augmented_fix_b6 \\")
    print(f"    --dataset-dir ./augmented_data \\")
    print(f"    --jsonl {output_path} \\")
    print(f"    --out {os.path.join(os.path.dirname(output_path), 'strategy_eval_llm.json')} \\")
    print(f"    --max-samples 270")


if __name__ == "__main__":
    main()
