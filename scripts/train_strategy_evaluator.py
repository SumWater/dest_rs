from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.casino_dataset import StrategyLabelSpace, load_split_examples


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in text.replace("\n", " ").split() if tok.strip()]


def build_vocab(texts: list[str], max_features: int, min_freq: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {"<unk>": 0}
    for token, count in counts.most_common(max_features - 1):
        if count < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def vectorize(text: str, vocab: dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for token in tokenize(text):
        idx = vocab.get(token, 0)
        vec[idx] += 1.0
    total = vec.sum().clamp_min(1.0)
    return vec / total


def make_xy(examples, label_space: StrategyLabelSpace, vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for ex in examples:
        text = ex.prompt + "\n" + ex.target
        xs.append(vectorize(text, vocab))
        ys.append(label_space.label_to_id[ex.primary_strategy])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def macro_f1(pred: torch.Tensor, gold: torch.Tensor, num_labels: int) -> float:
    scores = []
    for label in range(num_labels):
        tp = ((pred == label) & (gold == label)).sum().item()
        fp = ((pred == label) & (gold != label)).sum().item()
        fn = ((pred != label) & (gold == label)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            logits = model(x[start : start + batch_size])
            preds.append(logits.argmax(dim=-1))
    pred = torch.cat(preds)
    acc = (pred == y).float().mean().item()
    return {"accuracy": acc, "macro_f1": macro_f1(pred, y, int(y.max().item()) + 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="outputs/strategy_evaluator.pt")
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_examples = load_split_examples(cfg, "train")
    valid_examples = load_split_examples(cfg, "valid")
    test_examples = load_split_examples(cfg, "test")
    label_space = StrategyLabelSpace.fit(train_examples)

    vocab = build_vocab([ex.prompt + "\n" + ex.target for ex in train_examples], args.max_features, args.min_freq)
    x_train, y_train = make_xy(train_examples, label_space, vocab)
    x_valid, y_valid = make_xy(valid_examples, label_space, vocab)
    x_test, y_test = make_xy(test_examples, label_space, vocab)

    model = nn.Linear(len(vocab), len(label_space.labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_valid = -1.0
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            loss = F.cross_entropy(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        valid_metrics = evaluate(model, x_valid, y_valid, args.batch_size)
        if valid_metrics["macro_f1"] > best_valid:
            best_valid = valid_metrics["macro_f1"]
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print(f"epoch={epoch + 1} valid_acc={valid_metrics['accuracy']:.4f} valid_macro_f1={valid_metrics['macro_f1']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "labels": label_space.labels,
        "valid": evaluate(model, x_valid, y_valid, args.batch_size),
        "test": evaluate(model, x_test, y_test, args.batch_size),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({"已保存": str(out), "验证集": payload["valid"], "测试集": payload["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
