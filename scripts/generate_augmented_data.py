"""
Generate strategy-driven negotiation dialogues using Qwen3.5-9B.

Each turn is generated conditioned on a specific strategy, producing clean
single-label annotations for training DeSTRS.

Usage:
    python scripts/generate_augmented_data.py \
        --model-path /path/to/Qwen3.5-9B \
        --casino-train CaSiNo-main/data/split/casino_train.json \
        --casino-valid CaSiNo-main/data/split/casino_valid.json \
        --casino-test CaSiNo-main/data/split/casino_test.json \
        --output-dir ./augmented_data/split \
        --target-per-strategy 500 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Strategy definitions ─────────────────────────────────────────────────

STRATEGIES = {
    "small-talk": {
        "definition": "Casual conversation, greetings, or chit-chat unrelated to the negotiation itself.",
        "examples": [
            "Hi there! Hope your day is going well.",
            "Me too. I hope you have a wonderful camping trip!",
            "Doing great! How about yourself?",
        ],
        "guidance": "Keep it friendly and brief. Do NOT mention items or priorities.",
    },
    "showing-empathy": {
        "definition": "Express understanding, support, or emotional connection with the other party's situation.",
        "examples": [
            "I understand your concern. Let me see how I can help.",
            "That sounds tough. I'd like to find a solution that works for both of us.",
            "I totally get it — being out in nature without enough supplies can be stressful.",
        ],
        "guidance": "Acknowledge the other party's feelings or situation. Show genuine care.",
    },
    "elicit-pref": {
        "definition": "Ask about the other party's preferences, priorities, or situation to gather information.",
        "examples": [
            "What items are most important to you for this trip?",
            "Do you have a strong preference for any particular resource?",
            "What reasons do you have for needing those?",
        ],
        "guidance": "Ask a question about what the other person wants or needs. Do NOT state your own needs.",
    },
    "self-need": {
        "definition": "Express or emphasize your own needs, wants, or requirements.",
        "examples": [
            "I really need extra water because we'll be hiking in the heat all day.",
            "Firewood is critical for me — it gets freezing cold at night where I'm camping.",
            "I need all the food I can get, I have a large group to feed.",
        ],
        "guidance": "State why YOU need a specific item. Be specific about your situation.",
    },
    "other-need": {
        "definition": "Acknowledge, discuss, or accommodate the other party's needs.",
        "examples": [
            "I see you need water for your group. That makes sense.",
            "Since you mentioned you have kids, I understand you'd want extra food.",
            "You're right that firewood is important for staying warm.",
        ],
        "guidance": "Reference what the OTHER person needs. Show you listened to them.",
    },
    "no-need": {
        "definition": "Downplay or deny needing something to signal flexibility or enable a trade.",
        "examples": [
            "I don't really need water — there's a stream near my campsite.",
            "Firewood isn't a priority for me, so I'm flexible on that.",
            "We brought enough food, so I can be generous with those packages.",
        ],
        "guidance": "State that you do NOT need much of a specific item. Explain why briefly.",
    },
    "uv-part": {
        "definition": "Emphasize the unique value of items or justify why something matters specifically to you.",
        "examples": [
            "Having extra firewood means I can cook meals AND stay warm — it serves double duty.",
            "Water is irreplaceable out here. You can find food in nature, but clean water is essential.",
            "The firewood here is premium dry oak — it burns much longer than regular wood.",
        ],
        "guidance": "Highlight what makes an item especially valuable or versatile. Go beyond basic need.",
    },
    "promote-coordination": {
        "definition": "Propose collaboration, compromise, or working together toward a deal.",
        "examples": [
            "How about we split the water evenly and figure out the rest from there?",
            "I'd like to make a deal that works for both of us. What if I give you all the food?",
            "Let's try to find a middle ground — maybe we can each get our top priority.",
        ],
        "guidance": "Suggest a trade, propose a split, or invite the other party to negotiate together.",
    },
    "vouch-fair": {
        "definition": "Appeal to fairness, equity, or balanced outcomes in the negotiation.",
        "examples": [
            "Since you're getting all the water, I think it's fair that I get most of the firewood.",
            "I want to make sure we both walk away with a good deal here.",
            "How about a more even split? You take 2 food and I take 2 water, and we split the firewood.",
        ],
        "guidance": "Reference fairness explicitly. Argue that the proposed deal is balanced for both sides.",
    },
}

ALL_STRATEGIES = list(STRATEGIES.keys())

ISSUES = ["Food", "Water", "Firewood"]
PRIORITIES = ["High", "Medium", "Low"]

# ── Dialogue flow templates ──────────────────────────────────────────────

FLOW_TEMPLATES = [
    ["small-talk", "elicit-pref", "self-need", "other-need", "uv-part",
     "promote-coordination", "self-need", "vouch-fair", "showing-empathy",
     "promote-coordination"],
    ["small-talk", "showing-empathy", "elicit-pref", "self-need", "no-need",
     "other-need", "promote-coordination", "vouch-fair", "uv-part",
     "promote-coordination"],
    ["small-talk", "elicit-pref", "self-need", "uv-part", "other-need",
     "no-need", "promote-coordination", "showing-empathy", "vouch-fair",
     "promote-coordination"],
    ["showing-empathy", "small-talk", "elicit-pref", "self-need", "self-need",
     "other-need", "uv-part", "vouch-fair", "no-need", "promote-coordination"],
    ["small-talk", "elicit-pref", "other-need", "self-need", "showing-empathy",
     "uv-part", "no-need", "promote-coordination", "vouch-fair",
     "promote-coordination"],
    ["elicit-pref", "self-need", "showing-empathy", "other-need", "uv-part",
     "no-need", "vouch-fair", "promote-coordination", "small-talk",
     "promote-coordination"],
]

SCENARIO = (
    "Two participants are negotiating how to divide extra camping supplies: "
    "3 packages of Food, 3 packages of Water, and 3 packages of Firewood. "
    "Each participant has different priorities for these items."
)

# ── Reason pool (sampled from CaSiNo) ───────────────────────────────────

REASON_POOL = {
    "Food": {
        "High": [
            "I have a large group to feed and we need plenty of food.",
            "I'm diabetic and need to keep my blood sugar stable with regular meals.",
            "A bear broke into our camp and ate all our food supplies.",
            "We have kids with us and they need to eat regularly.",
            "I'm planning to do a lot of hiking and need energy from food.",
        ],
        "Medium": [
            "Having some food would be nice for snacks during the day.",
            "I'd like to cook a good dinner at the campsite.",
            "Food is important but we brought some of our own.",
            "I want to make sure we have enough for a couple of meals.",
        ],
        "Low": [
            "We brought plenty of our own food already.",
            "I can forage and fish for food if needed.",
            "Food isn't my top concern since we packed well.",
            "We can manage with less food for this short trip.",
        ],
    },
    "Water": {
        "High": [
            "I need lots of water for hiking in the heat all day.",
            "We have no water source near our campsite.",
            "Staying hydrated is my top priority in this dry climate.",
            "I have medical conditions that require me to drink extra water.",
            "My group is large and we need water for drinking and cooking.",
        ],
        "Medium": [
            "Some extra water would be helpful for the hikes we have planned.",
            "Water is important but there's a stream nearby we can use.",
            "I'd like to have backup water in case the nearby source is contaminated.",
            "We need water for cooking as well as drinking.",
        ],
        "Low": [
            "There's a clean stream right next to our campsite.",
            "I don't need much water since I'm not doing strenuous activities.",
            "Water isn't a big concern since we brought our own bottles.",
            "I can boil stream water if I have firewood.",
        ],
    },
    "Firewood": {
        "High": [
            "It gets freezing cold at night and I need firewood to stay warm.",
            "I want to cook all our meals over the campfire.",
            "The area has been picked clean of natural firewood by other campers.",
            "Firewood is essential for both warmth and cooking for my group.",
            "We're camping in a cold mountain area and need fires every night.",
        ],
        "Medium": [
            "A campfire would be nice in the evening for warmth.",
            "I'd like firewood for cooking at least one big meal.",
            "Having a fire would help keep insects away at night.",
            "We enjoy sitting around the campfire, so some firewood would be great.",
        ],
        "Low": [
            "I don't need much firewood since the weather is warm.",
            "We have a portable stove, so firewood isn't critical.",
            "I'm not planning to have a campfire every night.",
            "Firewood is nice to have but not essential for us.",
        ],
    },
}


def generate_profile(rng: random.Random) -> Dict:
    perm = rng.sample(ISSUES, k=3)
    value2issue = dict(zip(PRIORITIES, perm))
    value2reason = {}
    for priority in PRIORITIES:
        issue = value2issue[priority]
        pool = REASON_POOL[issue][priority]
        value2reason[priority] = rng.choice(pool)
    return {"value2issue": value2issue, "value2reason": value2reason}


def count_existing_strategies(casino_train: List[Dict]) -> Counter:
    counts = Counter()
    for d in casino_train:
        for ann in d.get("annotations") or []:
            if isinstance(ann, list) and len(ann) == 2:
                labels = [
                    l.strip()
                    for l in str(ann[1]).split(",")
                    if l.strip() and l.strip() != "non-strategic"
                ]
                if len(labels) == 1:
                    counts[labels[0]] += 1
    return counts


def plan_strategy_sequence(
    deficit: Counter, rng: random.Random, min_turns: int = 10, max_turns: int = 14
) -> List[str]:
    template = rng.choice(FLOW_TEMPLATES).copy()
    num_turns = rng.randint(min_turns, max_turns)

    needed = [s for s in ALL_STRATEGIES if deficit.get(s, 0) > 0]
    if needed:
        extras = rng.choices(needed, weights=[deficit[s] for s in needed],
                             k=max(0, num_turns - len(template)))
        template.extend(extras)

    return template[:num_turns]


def build_generation_prompt(
    history: List[Dict],
    speaker_id: str,
    speaker_profile: Dict,
    partner_profile: Dict,
    strategy: str,
) -> str:
    strat_info = STRATEGIES[strategy]

    profile_text = f"Your priorities: "
    v2i = speaker_profile["value2issue"]
    v2r = speaker_profile["value2reason"]
    profile_parts = []
    for priority in PRIORITIES:
        issue = v2i[priority]
        reason = v2r[priority]
        profile_parts.append(f"{issue} ({priority} priority — {reason})")
    profile_text += "; ".join(profile_parts) + "."

    history_text = ""
    if history:
        lines = []
        for turn in history[-6:]:
            role = "You" if turn["id"] == speaker_id else "Partner"
            lines.append(f"{role}: {turn['text']}")
        history_text = "\n".join(lines)
    else:
        history_text = "(conversation just started)"

    examples_text = "\n".join(f'  - "{ex}"' for ex in strat_info["examples"])

    prompt = f"""You are participating in a camping supplies negotiation. {SCENARIO}

{profile_text}

Dialogue so far:
{history_text}

Your task: Generate your next utterance using the "{strategy}" strategy.

Strategy definition: {strat_info["definition"]}
Guidance: {strat_info["guidance"]}
Examples of this strategy:
{examples_text}

Rules:
- Write ONLY the utterance itself, nothing else
- Keep it natural and conversational (1-3 sentences)
- Do NOT include any labels, prefixes, or meta-commentary
- Stay in character as a camping negotiator

Your utterance:"""

    return prompt


def generate_utterance(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant participating in a negotiation dialogue. Respond with only the utterance, no explanations."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True
    ).strip()

    response = response.split("\n")[0].strip()
    for prefix in ["You:", "Partner:", "Utterance:", "Response:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    response = response.strip('"').strip("'").strip()

    return response


def quality_check(text: str, history: List[Dict]) -> bool:
    if not text or len(text) < 5:
        return False
    if len(text) > 500:
        return False
    if history:
        last_text = history[-1].get("text", "")
        if text.lower() == last_text.lower():
            return False
    return True


def generate_dialogue(
    model,
    tokenizer,
    strategy_sequence: List[str],
    profile_a: Dict,
    profile_b: Dict,
    dialogue_id: int,
    max_retries: int = 3,
) -> Optional[Dict]:
    agent_a, agent_b = "agent_1", "agent_2"
    chat_logs = []
    annotations = []

    for turn_idx, strategy in enumerate(strategy_sequence):
        speaker = agent_a if turn_idx % 2 == 0 else agent_b
        partner = agent_b if speaker == agent_a else agent_a
        speaker_profile = profile_a if speaker == agent_a else profile_b
        partner_profile = profile_b if speaker == agent_a else profile_a

        prompt = build_generation_prompt(
            history=chat_logs,
            speaker_id=speaker,
            speaker_profile=speaker_profile,
            partner_profile=partner_profile,
            strategy=strategy,
        )

        utterance = None
        for attempt in range(max_retries):
            candidate = generate_utterance(model, tokenizer, prompt)
            if quality_check(candidate, chat_logs):
                utterance = candidate
                break

        if utterance is None:
            return None

        chat_logs.append({"text": utterance, "task_data": {}, "id": speaker})
        annotations.append([utterance, strategy])

    return {
        "dialogue_id": dialogue_id,
        "chat_logs": chat_logs,
        "participant_info": {
            agent_a: profile_a,
            agent_b: profile_b,
        },
        "annotations": annotations,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate augmented negotiation dialogues")
    parser.add_argument("--model-path", required=True, help="Path to Qwen3.5-9B")
    parser.add_argument("--casino-train", required=True, help="CaSiNo training split JSON")
    parser.add_argument("--casino-valid", required=True, help="CaSiNo valid split JSON")
    parser.add_argument("--casino-test", required=True, help="CaSiNo test split JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for augmented data")
    parser.add_argument("--target-per-strategy", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--max-dialogues", type=int, default=400,
                        help="Safety cap on total dialogues to generate")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load existing data
    casino_train = json.loads(Path(args.casino_train).read_text(encoding="utf-8"))
    casino_valid = json.loads(Path(args.casino_valid).read_text(encoding="utf-8"))
    casino_test = json.loads(Path(args.casino_test).read_text(encoding="utf-8"))

    existing = count_existing_strategies(casino_train)
    deficit = Counter()
    for s in ALL_STRATEGIES:
        gap = args.target_per_strategy - existing.get(s, 0)
        if gap > 0:
            deficit[s] = gap
    total_needed = sum(deficit.values())

    print("=" * 60)
    print("Strategy augmentation plan:")
    print(f"{'Strategy':<25} {'Existing':>8} {'Target':>8} {'Need':>8}")
    print("-" * 60)
    for s in ALL_STRATEGIES:
        need = deficit.get(s, 0)
        print(f"{s:<25} {existing.get(s, 0):>8} {args.target_per_strategy:>8} {need:>8}")
    print("-" * 60)
    print(f"Total to generate: {total_needed}")
    print("=" * 60)

    if total_needed == 0:
        print("All strategies already meet the target. Nothing to generate.")
        return

    # Load model
    print(f"\nLoading model from {args.model_path}...")
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
    if hasattr(model, "generation_config"):
        model.generation_config.enable_thinking = False
    model.eval()
    print("Model loaded.\n")

    # Generate dialogues
    generated_dialogues = []
    generated_counts = Counter()
    dialogue_id = 10001
    failed = 0

    while sum(deficit.values()) > 0 and len(generated_dialogues) < args.max_dialogues:
        profile_a = generate_profile(rng)
        profile_b = generate_profile(rng)
        strategy_seq = plan_strategy_sequence(deficit, rng)

        dialogue = generate_dialogue(
            model, tokenizer, strategy_seq, profile_a, profile_b, dialogue_id
        )

        if dialogue is None:
            failed += 1
            if failed > 50:
                print(f"Too many failures ({failed}), stopping.")
                break
            continue

        generated_dialogues.append(dialogue)
        dialogue_id += 1

        for ann in dialogue["annotations"]:
            s = ann[1]
            generated_counts[s] += 1
            if deficit[s] > 0:
                deficit[s] -= 1

        if len(generated_dialogues) % 10 == 0:
            remaining = sum(deficit.values())
            print(f"  Generated {len(generated_dialogues)} dialogues, "
                  f"{sum(generated_counts.values())} turns, "
                  f"{remaining} still needed, {failed} failures")

    print(f"\nGeneration complete: {len(generated_dialogues)} dialogues, "
          f"{sum(generated_counts.values())} turns, {failed} failures")

    # Split generated data 90/10
    rng.shuffle(generated_dialogues)
    split_idx = max(1, int(len(generated_dialogues) * 0.9))
    gen_train = generated_dialogues[:split_idx]
    gen_test = generated_dialogues[split_idx:]

    # Assemble output
    combined_train = casino_train + gen_train
    rng.shuffle(combined_train)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [
        ("casino_train.json", combined_train),
        ("casino_valid.json", casino_valid),
        ("casino_test.json", casino_test),
        ("generated_test.json", gen_test),
    ]:
        path = out_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Stats
    final_counts = count_existing_strategies(combined_train)
    stats = {
        "generated_dialogues": len(generated_dialogues),
        "generated_turns": sum(generated_counts.values()),
        "failed_dialogues": failed,
        "gen_train_dialogues": len(gen_train),
        "gen_test_dialogues": len(gen_test),
        "combined_train_dialogues": len(combined_train),
        "strategy_counts_before": dict(existing),
        "strategy_counts_generated": dict(generated_counts),
        "strategy_counts_final_train": dict(final_counts),
        "target_per_strategy": args.target_per_strategy,
    }
    stats_path = out_dir / "generation_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Final training set strategy distribution:")
    print(f"{'Strategy':<25} {'Before':>8} {'Generated':>10} {'Final':>8}")
    print("-" * 60)
    for s in ALL_STRATEGIES:
        print(f"{s:<25} {existing.get(s, 0):>8} {generated_counts.get(s, 0):>10} "
              f"{final_counts.get(s, 0):>8}")
    print("=" * 60)
    print(f"\nOutput written to {out_dir}")
    print(f"  casino_train.json: {len(combined_train)} dialogues (original + generated)")
    print(f"  casino_valid.json: {len(casino_valid)} dialogues (unchanged)")
    print(f"  casino_test.json:  {len(casino_test)} dialogues (unchanged)")
    print(f"  generated_test.json: {len(gen_test)} dialogues (generated hold-out)")


if __name__ == "__main__":
    main()
