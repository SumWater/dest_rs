from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from train_strategy_evaluator import macro_f1, vectorize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = torch.load(args.evaluator, map_location="cpu")
    vocab = payload["vocab"]
    labels = payload["labels"]
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    model = nn.Linear(len(vocab), len(labels))
    model.load_state_dict(payload["state_dict"])
    model.eval()

    rows = []
    with Path(args.jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt = record.get("prompt") or ""
            for target_strategy, generated in (record.get("generated_by_strategy") or {}).items():
                if target_strategy not in label_to_id:
                    continue
                x = vectorize(prompt + "\n" + (generated or ""), vocab).unsqueeze(0)
                with torch.no_grad():
                    pred_id = int(model(x).argmax(dim=-1).item())
                rows.append(
                    {
                        "dialogue_id": record.get("dialogue_id"),
                        "turn_index": record.get("turn_index"),
                        "target_strategy": target_strategy,
                        "pred_strategy": labels[pred_id],
                        "correct": int(pred_id == label_to_id[target_strategy]),
                        "text": generated or "",
                    }
                )

    gold = torch.tensor([label_to_id[row["target_strategy"]] for row in rows], dtype=torch.long)
    pred = torch.tensor([label_to_id[row["pred_strategy"]] for row in rows], dtype=torch.long)
    metrics = {
        "strategy_accuracy": (pred == gold).float().mean().item() if len(rows) else 0.0,
        "macro_f1": macro_f1(pred, gold, len(labels)) if len(rows) else 0.0,
        "samples": len(rows),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["dialogue_id", "turn_index", "target_strategy", "pred_strategy", "correct", "text"]
            )
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
