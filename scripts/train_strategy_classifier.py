"""C1/C2/C3 strategy-predictability diagnostics using a hashed linear classifier.

No pretrained language model or sklearn dependency is required. Splits are read
directly from the existing dialogue-level CaSiNo JSON files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn

STRATEGIES = [
    "small-talk", "self-need", "other-need", "no-need", "uv-part",
    "elicit-pref", "showing-empathy", "promote-coordination", "vouch-fair",
]
INDEX = {x: i for i, x in enumerate(STRATEGIES)}


def annotation_labels(annotation):
    if not isinstance(annotation, (list, tuple)) or len(annotation) < 2:
        return []
    raw = annotation[1]
    values = raw if isinstance(raw, list) else re.split(r"[,;|]", str(raw))
    return [str(x).strip() for x in values if str(x).strip() in INDEX]


def build_examples(path: Path, mode: str, context_turns: int):
    dialogues = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for dialogue in dialogues:
        logs = dialogue.get("chat_logs") or []
        annotations = dialogue.get("annotations") or []
        for turn, annotation in enumerate(annotations[:len(logs)]):
            labels = annotation_labels(annotation)
            if not labels:
                continue
            response = str(logs[turn].get("text", "")).strip()
            history = []
            for i in range(max(0, turn - context_turns), turn):
                role = "same-speaker" if logs[i].get("id") == logs[turn].get("id") else "partner"
                history.append(f"{role}: {logs[i].get('text', '')}")
            context = "\n".join(history) if history else "[conversation-start]"
            if mode == "context": text = context
            elif mode == "response": text = response
            else: text = f"[CONTEXT]\n{context}\n[RESPONSE]\n{response}"
            target = [0.0] * len(STRATEGIES)
            for label in labels: target[INDEX[label]] = 1.0
            examples.append({
                "dialogue_id": dialogue.get("dialogue_id"), "turn_index": turn,
                "text": text, "context": context, "response": response,
                "labels": labels, "target": target,
            })
    return examples


def features(text: str, dimension: int):
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    grams = tokens + [tokens[i] + "_" + tokens[i + 1] for i in range(len(tokens) - 1)]
    counts = Counter()
    for gram in grams:
        digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
        counts[int.from_bytes(digest, "little") % dimension] += 1
    if not counts:
        return [], []
    norm = math.sqrt(sum(v * v for v in counts.values()))
    return list(counts), [v / norm for v in counts.values()]


def tensorize(examples, dimension, mode):
    total_dimension = dimension * 2 if mode == "context_response" else dimension
    x = torch.zeros((len(examples), total_dimension), dtype=torch.float32)
    y = torch.tensor([e["target"] for e in examples], dtype=torch.float32)
    for row, example in enumerate(examples):
        if mode == "context_response":
            context_indices, context_values = features(example["context"], dimension)
            response_indices, response_values = features(example["response"], dimension)
            if context_indices: x[row, context_indices] = torch.tensor(context_values)
            if response_indices:
                shifted = [i + dimension for i in response_indices]
                x[row, shifted] = torch.tensor(response_values)
        else:
            indices, values = features(example["text"], dimension)
            if indices: x[row, indices] = torch.tensor(values)
    return x, y


def metrics(logits, targets, examples, threshold):
    probs = torch.sigmoid(logits).cpu()
    truth = targets.bool().cpu()
    pred = probs >= threshold
    # Always emit at least the argmax label, avoiding empty predictions.
    empty = pred.sum(1) == 0
    pred[empty, probs[empty].argmax(1)] = True
    per_class, f1s = {}, []
    for i, strategy in enumerate(STRATEGIES):
        tp = int((pred[:, i] & truth[:, i]).sum())
        fp = int((pred[:, i] & ~truth[:, i]).sum())
        fn = int((~pred[:, i] & truth[:, i]).sum())
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        support = int(truth[:, i].sum())
        per_class[strategy] = {"precision": precision, "recall": recall, "f1": f1,
                               "support": support, "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    exact = float((pred == truth).all(1).float().mean())
    single = truth.sum(1) == 1
    primary_true = truth.float().argmax(1)
    primary_pred = probs.argmax(1)
    accuracy = float((primary_true[single] == primary_pred[single]).float().mean()) if single.any() else 0
    confusion = [[0] * len(STRATEGIES) for _ in STRATEGIES]
    for t, p in zip(primary_true[single], primary_pred[single]): confusion[int(t)][int(p)] += 1
    prediction_distribution = Counter()
    details = []
    for i, example in enumerate(examples):
        predicted = [STRATEGIES[j] for j in range(len(STRATEGIES)) if pred[i, j]]
        prediction_distribution.update(predicted)
        details.append({"dialogue_id": example["dialogue_id"], "turn_index": example["turn_index"],
                        "gold_labels": example["labels"], "predicted_labels": predicted,
                        "primary_prediction": STRATEGIES[int(primary_pred[i])]})
    return {"samples": len(examples), "single_label_samples": int(single.sum()),
            "single_label_accuracy": accuracy, "multilabel_exact_match": exact,
            "macro_f1": sum(f1s) / len(f1s), "per_class": per_class,
            "confusion_matrix_labels": STRATEGIES, "confusion_matrix": confusion,
            "prediction_distribution": dict(prediction_distribution), "details": details}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("CaSiNo-main/data/split"))
    p.add_argument("--mode", choices=["context", "response", "context_response"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dimension", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--context-turns", type=int, default=6)
    args = p.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    files = {"train": "casino_train.json", "dev": "casino_valid.json", "test": "casino_test.json"}
    examples = {s: build_examples(args.data_dir / f, args.mode, args.context_turns) for s, f in files.items()}
    tensors = {s: tensorize(examples[s], args.dimension, args.mode) for s in files}
    input_dimension = args.dimension * 2 if args.mode == "context_response" else args.dimension
    model = nn.Linear(input_dimension, len(STRATEGIES))
    positives = tensors["train"][1].sum(0)
    negatives = len(examples["train"]) - positives
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives.clamp_min(1)).clamp(max=20))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state, best_dev, history = None, -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(examples["train"])); total = 0.0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = model(tensors["train"][0][idx]); loss = loss_fn(logits, tensors["train"][1][idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step(); total += float(loss) * len(idx)
        model.eval()
        with torch.no_grad(): dev = metrics(model(tensors["dev"][0]), tensors["dev"][1], examples["dev"], args.threshold)
        history.append({"epoch": epoch, "train_loss": total / len(order), "dev_macro_f1": dev["macro_f1"]})
        if dev["macro_f1"] > best_dev:
            best_dev = dev["macro_f1"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    results = {"mode": args.mode, "seed": args.seed, "data_dir": str(args.data_dir),
               "config": vars(args) | {"out": str(args.out), "data_dir": str(args.data_dir)},
               "history": history}
    with torch.no_grad():
        for split in ("dev", "test"):
            results[split] = metrics(model(tensors[split][0]), tensors[split][1], examples[split], args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    compact = {s: {k: results[s][k] for k in ("samples", "single_label_samples", "single_label_accuracy", "multilabel_exact_match", "macro_f1")} for s in ("dev", "test")}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__": main()
