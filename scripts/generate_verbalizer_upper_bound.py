"""Generate V0--V4 explicit-strategy verbalizer upper-bound outputs.

Uses the frozen base Qwen3-8B only: no Prefix and no LoRA checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

STRATEGIES = [
    "small-talk", "self-need", "other-need", "no-need", "uv-part",
    "elicit-pref", "showing-empathy", "promote-coordination", "vouch-fair",
]
DEFINITIONS = {
    "small-talk": "Use a social greeting or casual conversation rather than negotiating needs.",
    "self-need": "State the current speaker's own need or reason for wanting an item.",
    "other-need": "State or acknowledge the other participant's concrete need or reason.",
    "no-need": "State that the current speaker does not need or has low need for an item.",
    "uv-part": "Argue that an item or allocation is less valuable or necessary to the partner.",
    "elicit-pref": "Ask the partner about preferences, priorities, or reasons.",
    "showing-empathy": "Express understanding of or sympathy for the partner's situation.",
    "promote-coordination": "Encourage joint problem solving, compromise, or working together.",
    "vouch-fair": "Claim that an offer, split, or bargaining position is fair or balanced.",
}
EXAMPLES = {
    "small-talk": "Hi there, how is your camping trip going?",
    "self-need": "I need the water because my group must stay hydrated.",
    "other-need": "I understand that you need the food to feed your group.",
    "no-need": "I do not need much firewood, so you can have it.",
    "uv-part": "Since you already brought drinks, the water is less important for you.",
    "elicit-pref": "Which supplies matter most to you, and why?",
    "showing-empathy": "I understand why keeping your family warm is important to you.",
    "promote-coordination": "Let us work together on a split that covers both of our priorities.",
    "vouch-fair": "An even split of the remaining supplies would be fair to both of us.",
}


def extract_sections(prompt: str):
    profile_match = re.search(r"(Target speaker private profile:\n.*?)(?=\nDialogue history:)", prompt, re.S)
    history_match = re.search(r"Dialogue history:\n(.*?)(?=\n\nGenerate the target speaker)", prompt, re.S)
    if not history_match:
        raise ValueError("Cannot parse dialogue history from reference prompt")
    return (profile_match.group(1).strip() if profile_match else "", history_match.group(1).strip())


def instruction(condition: str, target: str, profile: str, history: str):
    sections = ["You are producing the next natural utterance in a campsite negotiation."]
    if condition in {"v3_profile_definition", "v4_profile_definition_example"}:
        sections += [profile]
    sections += [f"Dialogue history:\n{history}"]
    if condition == "v1_strategy_name":
        sections += [f"Target strategy: {target}"]
    elif condition in {"v2_strategy_definition", "v3_profile_definition", "v4_profile_definition_example"}:
        sections += [f"Target strategy definition:\n{DEFINITIONS[target]}"]
    if condition == "v4_profile_definition_example":
        sections += [f"Example of this strategy (do not copy its facts):\n{EXAMPLES[target]}"]
    if condition == "v0_context":
        sections += ["Respond naturally to continue the negotiation."]
    else:
        sections += ["Write a response that clearly follows the target strategy."]
    sections += ["Do not mention the strategy name. Do not invent facts. Output only one utterance."]
    return "\n\n".join(x for x in sections if x)


def batches(values, size):
    for i in range(0, len(values), size): yield values[i:i + size]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=40)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")

    reference = [json.loads(x) for x in args.reference.read_text(encoding="utf-8").splitlines() if x.strip()]
    conditions = ["v0_context", "v1_strategy_name", "v2_strategy_definition",
                  "v3_profile_definition", "v4_profile_definition_example"]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=quant,
                                                 device_map="auto", dtype=torch.bfloat16,
                                                 trust_remote_code=True).eval()
    device_map = getattr(model, "hf_device_map", {}) or {}
    if any(str(x) in {"cpu", "disk"} for x in device_map.values()):
        raise RuntimeError(f"CPU/disk offload detected: {device_map}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"device_map={device_map}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        tasks = []
        for row_index, record in enumerate(reference):
            profile, history = extract_sections(record["prompt"])
            for target in STRATEGIES:
                tasks.append((row_index, target, instruction(condition, target, profile, history)))
        outputs = []
        for batch in batches(tasks, args.batch_size):
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": x[2]}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) for x in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            prefix_len = enc.input_ids.shape[1]
            decoded = tokenizer.batch_decode(generated[:, prefix_len:], skip_special_tokens=True)
            outputs.extend(x.strip() for x in decoded)
            print(f"{condition}: {len(outputs)}/{len(tasks)}", flush=True)
        records = []
        cursor = 0
        for ref in reference:
            generated_by_strategy = {}
            for target in STRATEGIES:
                generated_by_strategy[target] = outputs[cursor]; cursor += 1
            records.append({**ref, "verbalizer_condition": condition,
                            "generated_by_strategy": generated_by_strategy})
        out = args.out_dir / f"{condition}.jsonl"
        out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
        print(f"saved={out}", flush=True)


if __name__ == "__main__": main()
