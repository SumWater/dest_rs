from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FILES = {"train": "casino_train.json", "valid": "casino_valid.json", "test": "casino_test.json"}


def resolve(dataset_dir: Path, split: str) -> Path:
    direct = dataset_dir / FILES[split]
    nested = dataset_dir / "split" / FILES[split]
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"在 {dataset_dir} 下找不到 {FILES[split]}")


def labels(annotation) -> list[str]:
    if not isinstance(annotation, list) or len(annotation) != 2:
        return []
    return [x.strip() for x in str(annotation[1]).split(",") if x.strip()]


def choose(raw: list[str], policy: str, exclude: set[str]) -> list[str]:
    kept = [x for x in raw if x not in exclude]
    if not kept:
        return []
    if policy == "drop" and len(kept) != 1:
        return []
    if policy == "first":
        return [kept[0]]
    if policy == "duplicate":
        return kept
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--policy", default="drop", choices=["drop", "first", "duplicate"])
    parser.add_argument("--exclude-label", action="append", default=["non-strategic"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    exclude = set(args.exclude_label or [])
    for split in ["train", "valid", "test"]:
        path = resolve(Path(args.dataset_dir), split)
        dialogues = json.loads(path.read_text(encoding="utf-8"))
        counter: Counter[str] = Counter()
        total_turns = 0
        used_turns = 0
        for dialogue in dialogues:
            for ann in dialogue.get("annotations") or []:
                total_turns += 1
                chosen = choose(labels(ann), args.policy, exclude)
                if chosen:
                    used_turns += 1
                    counter.update(chosen)
        for label, count in sorted(counter.items()):
            row = {
                "split": split,
                "label": label,
                "count": count,
                "total_turns": total_turns,
                "used_turns": used_turns,
                "share": count / used_turns if used_turns else 0.0
            }
            rows.append(row)
            print(f"{split:5s} {label:<24s} {count:5d} share={row['share']:.3f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "label", "count", "total_turns", "used_turns", "share"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
