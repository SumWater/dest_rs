#!/usr/bin/env python3
"""Re-split CaSiNo dataset using only annotated dialogues with stratified sampling.

The official CaSiNo split randomly divides all 1030 dialogues, but only 396 have
strategy annotations. This leads to a valid set with just 7 annotated dialogues
(68 examples) and missing strategy coverage. This script:
  1. Filters to the 396 annotated dialogues
  2. Stratifies by the rarest strategy present in each dialogue
  3. Splits 80/10/10 with seed=42
  4. Backs up originals to split/original_split/
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

EXCLUDE_LABELS = {"non-strategic"}
SEED = 42
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10


def get_strategies(dialogue: Dict) -> Set[str]:
    strats: Set[str] = set()
    for ann in dialogue.get("annotations") or []:
        if isinstance(ann, list) and len(ann) == 2:
            for lab in str(ann[1]).split(","):
                lab = lab.strip()
                if lab and lab not in EXCLUDE_LABELS:
                    strats.add(lab)
    return strats


def count_examples(dialogues: List[Dict]) -> int:
    count = 0
    for d in dialogues:
        annotations = d.get("annotations") or []
        chat_logs = d.get("chat_logs") or []
        if not annotations:
            continue
        usable = chat_logs[: len(annotations)]
        if len(usable) != len(annotations):
            continue
        for ann in annotations:
            if not isinstance(ann, list) or len(ann) != 2:
                continue
            labels = [l.strip() for l in str(ann[1]).split(",") if l.strip()]
            filtered = [l for l in labels if l not in EXCLUDE_LABELS]
            if filtered:
                count += len(filtered)
    return count


def main():
    project_dir = Path(__file__).resolve().parent.parent
    casino_path = project_dir / "CaSiNo-main" / "data" / "casino.json"
    split_dir = project_dir / "CaSiNo-main" / "data" / "split"

    all_dialogues = json.loads(casino_path.read_text(encoding="utf-8"))
    annotated = [d for d in all_dialogues if d.get("annotations")]
    print(f"Total dialogues: {len(all_dialogues)}, annotated: {len(annotated)}")

    strat_freq = Counter()
    for d in annotated:
        for s in get_strategies(d):
            strat_freq[s] += 1

    # Assign each dialogue to its rarest strategy for stratification
    for d in annotated:
        strats = get_strategies(d)
        d["_strat_key"] = min(strats, key=lambda s: strat_freq[s]) if strats else ""

    # Group by stratification key
    groups: Dict[str, List[Dict]] = {}
    for d in annotated:
        groups.setdefault(d["_strat_key"], []).append(d)

    rng = random.Random(SEED)
    train, valid, test = [], [], []

    for key in sorted(groups.keys()):
        bucket = groups[key]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, round(n * TRAIN_RATIO))
        n_valid = max(1, round(n * VALID_RATIO))
        n_test = n - n_train - n_valid
        if n_test < 1:
            n_train -= 1
            n_test = 1
        train.extend(bucket[:n_train])
        valid.extend(bucket[n_train : n_train + n_valid])
        test.extend(bucket[n_train + n_valid :])

    # Clean up temporary key
    for d in train + valid + test:
        d.pop("_strat_key", None)

    # Backup originals
    backup_dir = split_dir / "original_split"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True)
        for name in ["casino_train.json", "casino_valid.json", "casino_test.json"]:
            src = split_dir / name
            if src.exists():
                shutil.copy2(src, backup_dir / name)
        print(f"Original split backed up to {backup_dir}")

    # Write new splits
    for name, data in [
        ("casino_train.json", train),
        ("casino_valid.json", valid),
        ("casino_test.json", test),
    ]:
        path = split_dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Report
    print()
    print("=" * 70)
    print(f"{'Split':<8} {'Dialogues':>10} {'Examples(dup)':>15}")
    print("-" * 70)
    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        print(f"{name:<8} {len(data):>10} {count_examples(data):>15}")
    print("=" * 70)

    print()
    all_strats = sorted(strat_freq.keys())
    header = f"{'Strategy':<22}" + "".join(f"{s:>8}" for s in ["train", "valid", "test"])
    print(header)
    print("-" * len(header))
    for strat in all_strats:
        counts = []
        for data in [train, valid, test]:
            c = sum(1 for d in data if strat in get_strategies(d))
            counts.append(c)
        row = f"{strat:<22}" + "".join(f"{c:>8}" for c in counts)
        missing = " *** MISSING" if any(c == 0 for c in counts) else ""
        print(row + missing)

    print()
    print("Done. All 9 strategies should appear in every split.")


if __name__ == "__main__":
    main()
