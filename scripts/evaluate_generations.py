from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def tokens(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def ngrams(items: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(items[i : i + n]) for i in range(max(0, len(items) - n + 1))]


def distinct(texts: list[str], n: int) -> float:
    grams = []
    for text in texts:
        grams.extend(ngrams(tokens(text), n))
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_rate(texts: list[str], n: int = 4) -> float:
    repeated = 0
    counted = 0
    for text in texts:
        grams = ngrams(tokens(text), n)
        if not grams:
            continue
        counted += 1
        if any(value > 1 for value in Counter(grams).values()):
            repeated += 1
    return repeated / counted if counted else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = []
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            for strategy, text in (record.get("generated_by_strategy") or {}).items():
                rows.append(
                    {
                        "dialogue_id": record.get("dialogue_id"),
                        "turn_index": record.get("turn_index"),
                        "gold_strategy": record.get("gold_strategy"),
                        "generated_strategy": strategy,
                        "text": text or ""
                    }
                )

    texts = [row["text"] for row in rows]
    lengths = [len(tokens(text)) for text in texts]
    stats = {
        "generations": len(rows),
        "avg_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "distinct_1": distinct(texts, 1),
        "distinct_2": distinct(texts, 2),
        "repetition_rate_4gram": repetition_rate(texts, 4)
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dialogue_id", "turn_index", "gold_strategy", "generated_strategy", "text"]
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
