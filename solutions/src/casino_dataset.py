from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .config import TrainConfig


# ── 策略文本描述（用于 inject_strategy_text） ──
STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "elicit-pref": "Discover the negotiation partner's preference order or item priorities",
    "self-need": "Establish the current speaker's own personal need or reason for an item",
    "other-need": "Establish an item need for someone associated with the speaker other than the speaker",
    "no-need": "State that the current speaker does not need, has low need for, or has enough of an item",
    "promote-coordination": "Promote a trade, mutual concession, exchange, or joint effort to reach a deal",
    "showing-empathy": "Positively acknowledge the negotiation partner's personal context",
    "small-talk": "Use social conversation outside negotiation and item allocation to build rapport",
    "uv-part": "Undervalue or question the negotiation partner's need or need strength for an item",
    "vouch-fair": "Appeal to fairness or call out an allocation imbalance for personal benefit",
}


def _make_strategy_instruction(strategy_name: str) -> str:
    """构建单条策略文本指令。"""
    desc = STRATEGY_DESCRIPTIONS.get(strategy_name, strategy_name)
    return f"Use the following negotiation strategy: {desc}."


OFFICIAL_SPLIT_FILES = {
    "train": "casino_train.json",
    "valid": "casino_valid.json",
    "test": "casino_test.json",
}


@dataclass
class StrategyExample:
    dialogue_id: int
    turn_index: int
    speaker_id: str
    primary_strategy: str
    all_strategies: List[str]
    prompt: str
    target: str


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\n", " ").split()).strip()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_split_path(cfg: TrainConfig, split: str) -> str:
    explicit = {
        "train": cfg.train_file,
        "valid": cfg.valid_file,
        "test": cfg.test_file,
    }[split]
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"Split file does not exist: {explicit}")
        return explicit

    direct = os.path.join(cfg.dataset_dir, OFFICIAL_SPLIT_FILES[split])
    nested = os.path.join(cfg.dataset_dir, "split", OFFICIAL_SPLIT_FILES[split])
    if os.path.exists(direct):
        return direct
    if os.path.exists(nested):
        return nested
    raise FileNotFoundError(
        f"在 {cfg.dataset_dir} 下找不到 {OFFICIAL_SPLIT_FILES[split]}。"
        "期望路径为 dataset_dir/casino_train.json 或 dataset_dir/split/casino_train.json"
    )


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_profile(participant: Dict, max_reason_chars: int) -> str:
    value2issue = participant.get("value2issue", {})
    value2reason = participant.get("value2reason", {})
    lines = ["Target speaker private profile:"]
    for priority in ["High", "Medium", "Low"]:
        issue = value2issue.get(priority, "未知")
        reason = clip_text(value2reason.get(priority, ""), max_reason_chars)
        lines.append(f"- {priority} priority item: {issue}")
        if reason:
            lines.append(f"  Reason: {reason}")
    return "\n".join(lines)


def build_prompt(
    history_turns: Sequence[Dict],
    target_speaker_id: str,
    participant_info: Dict,
    cfg: TrainConfig,
) -> str:
    sections: List[str] = [
        "You are generating the next utterance for a negotiation dialogue in the CaSiNo campsite bargaining task.",
    ]

    if cfg.include_profile:
        speaker_profile = participant_info.get(target_speaker_id, {})
        sections.append(format_profile(speaker_profile, cfg.max_reason_chars))

    history_lines = ["Dialogue history:"]
    if not history_turns:
        history_lines.append("(conversation start)")
    else:
        trimmed_history = list(history_turns[-cfg.context_turns :]) if cfg.context_turns > 0 else list(history_turns)
        for turn in trimmed_history:
            role = "You" if turn.get("id") == target_speaker_id else "Partner"
            text = normalize_text(turn.get("text", ""))
            history_lines.append(f"{role}: {text}")

    sections.append("\n".join(history_lines))
    sections.append("Generate the target speaker's next utterance.\nResponse:")
    return "\n\n".join(sections)


def choose_labels(labels: List[str], cfg: TrainConfig) -> List[str]:
    filtered = [lab for lab in labels if lab and lab not in set(cfg.exclude_labels)]
    if not filtered:
        return []
    policy = (cfg.multi_label_policy or "").strip().lower()
    if policy == "drop":
        return filtered if len(filtered) == 1 else []
    if policy == "first":
        return [filtered[0]]
    if policy == "duplicate":
        return filtered
    raise ValueError(
        f"不支持的 multi_label_policy={cfg.multi_label_policy!r}。"
        "可选值为：drop、first、duplicate"
    )


def build_examples_from_dialogues(dialogues: Sequence[Dict], cfg: TrainConfig) -> List[StrategyExample]:
    examples: List[StrategyExample] = []
    for dialogue in dialogues:
        annotations = dialogue.get("annotations") or []
        chat_logs = dialogue.get("chat_logs") or []
        if not annotations:
            continue

        usable_logs = chat_logs[: len(annotations)]
        if len(usable_logs) != len(annotations):
            continue

        for idx, ann in enumerate(annotations):
            if not isinstance(ann, list) or len(ann) != 2:
                continue
            ann_text, ann_labels = ann
            raw_labels = [lab.strip() for lab in str(ann_labels).split(",") if lab.strip()]
            chosen = choose_labels(raw_labels, cfg)
            if not chosen:
                continue

            log_turn = usable_logs[idx]
            target_text = normalize_text(log_turn.get("text", ""))
            if not target_text:
                continue

            # 对标注文本与真实对话文本做宽松一致性检查。
            if normalize_text(ann_text) and normalize_text(ann_text) != target_text:
                # 训练时保留 chat log 文本，因为它才是真实的序列来源。
                pass

            history_turns = usable_logs[:idx]
            speaker_id = log_turn.get("id", "unknown_speaker")
            prompt = build_prompt(
                history_turns=history_turns,
                target_speaker_id=speaker_id,
                participant_info=dialogue.get("participant_info", {}),
                cfg=cfg,
            )

            for strategy in chosen:
                examples.append(
                    StrategyExample(
                        dialogue_id=int(dialogue.get("dialogue_id", -1)),
                        turn_index=idx,
                        speaker_id=speaker_id,
                        primary_strategy=strategy,
                        all_strategies=raw_labels,
                        prompt=prompt,
                        target=target_text,
                    )
                )
    return examples


class StrategyLabelSpace:
    def __init__(self, labels: Sequence[str]):
        self.labels = list(labels)
        self.label_to_id = {label: idx for idx, label in enumerate(self.labels)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.labels)}

    @classmethod
    def fit(cls, examples: Sequence[StrategyExample]) -> "StrategyLabelSpace":
        unique = sorted({ex.primary_strategy for ex in examples})
        return cls(unique)

    def to_json(self) -> Dict:
        return {
            "labels": self.labels,
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
        }

    @classmethod
    def from_json(cls, payload: Dict) -> "StrategyLabelSpace":
        labels = payload.get("labels")
        if not labels:
            labels = [label for label, _ in sorted(payload["label_to_id"].items(), key=lambda x: x[1])]
        return cls(labels)


class CasinoStrategyDataset(Dataset):
    def __init__(self, examples: Sequence[StrategyExample], label_space: StrategyLabelSpace):
        self.examples = [ex for ex in examples if ex.primary_strategy in label_space.label_to_id]
        self.label_space = label_space

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        return {
            "dialogue_id": ex.dialogue_id,
            "turn_index": ex.turn_index,
            "speaker_id": ex.speaker_id,
            "primary_strategy": ex.primary_strategy,
            "strategy_id": self.label_space.label_to_id[ex.primary_strategy],
            "all_strategies": ex.all_strategies,
            "prompt": ex.prompt,
            "target": ex.target,
        }


class StrategyDataCollator:
    def __init__(self, tokenizer, cfg: TrainConfig):
        self.tokenizer = tokenizer
        self.cfg = cfg
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_prompt_target(self, prompt: str, target: str) -> Tuple[List[int], List[int], List[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_text = target + (self.tokenizer.eos_token or "")
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]

        max_len = self.cfg.max_length
        if len(target_ids) >= max_len:
            target_ids = target_ids[: max_len - 1]
        available_for_prompt = max_len - len(target_ids)
        if available_for_prompt < 1:
            available_for_prompt = 1
        prompt_ids = prompt_ids[-available_for_prompt:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, labels

    def __call__(self, batch: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 构建 prompt（可选注入策略文本指令）
        if self.cfg.inject_strategy_text:
            prompts = [
                _make_strategy_instruction(item["primary_strategy"]) + "\n\n" + item["prompt"]
                for item in batch
            ]
        else:
            prompts = [item["prompt"] for item in batch]

        encoded = [self.encode_prompt_target(p, item["target"]) for p, item in zip(prompts, batch)]
        max_seq = max(len(x[0]) for x in encoded)
        pad_id = self.tokenizer.pad_token_id

        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        strategy_ids = []
        meta = []
        for item, (input_ids, attention_mask, labels) in zip(batch, encoded):
            pad_len = max_seq - len(input_ids)
            input_ids_list.append(input_ids + [pad_id] * pad_len)
            attention_mask_list.append(attention_mask + [0] * pad_len)
            labels_list.append(labels + [-100] * pad_len)
            strategy_ids.append(item["strategy_id"])
            meta.append(
                {
                    "dialogue_id": item["dialogue_id"],
                    "turn_index": item["turn_index"],
                    "speaker_id": item["speaker_id"],
                    "primary_strategy": item["primary_strategy"],
                    "all_strategies": item["all_strategies"],
                    "prompt": item["prompt"],
                    "target": item["target"],
                }
            )

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long),
            "strategy_id": torch.tensor(strategy_ids, dtype=torch.long),
            "meta": meta,
        }


def maybe_limit(examples: List[StrategyExample], max_samples: Optional[int], seed: int) -> List[StrategyExample]:
    if max_samples is None or max_samples >= len(examples):
        return examples
    rng = random.Random(seed)
    kept = list(examples)
    rng.shuffle(kept)
    return kept[:max_samples]


def load_split_examples(cfg: TrainConfig, split: str) -> List[StrategyExample]:
    split_path = resolve_split_path(cfg, split)
    dialogues = load_json(split_path)
    examples = build_examples_from_dialogues(dialogues, cfg)
    if split == "train":
        return maybe_limit(examples, cfg.max_train_samples, cfg.seed)
    if split == "valid":
        return maybe_limit(examples, cfg.max_valid_samples, cfg.seed)
    if split == "test":
        return maybe_limit(examples, cfg.max_test_samples, cfg.seed)
    return examples
