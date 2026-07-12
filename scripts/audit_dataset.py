"""Audit original and class-balanced CaSiNo data without modifying it."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def labels(annotation):
    if not isinstance(annotation, (list, tuple)) or len(annotation) < 2:
        return []
    value = annotation[1]
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    # CaSiNo sometimes serializes multiple labels using comma/semicolon.
    return [x.strip() for x in re.split(r"[,;|]", str(value)) if x.strip()]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def dialogue_records(dialogues, split, source, original_ids):
    out = []
    for d in dialogues:
        did = str(d.get("dialogue_id"))
        anns = d.get("annotations") or []
        logs = d.get("chat_logs") or []
        for i, ann in enumerate(anns[: len(logs)]):
            text = str(logs[i].get("text", ""))
            labs = labels(ann)
            speaker = logs[i].get("id")
            profile = (d.get("participant_info") or {}).get(speaker, {})
            out.append({
                "split": split, "source": source, "dialogue_id": did,
                "turn_index": i, "text": text, "labels": labs,
                "profile_present": bool(profile.get("value2issue")),
                "reasons_present": bool(profile.get("value2reason")),
                "is_augmented": did not in original_ids,
            })
    return out


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original-dir", type=Path, default=Path("CaSiNo-main/data/split"))
    p.add_argument("--augmented-dir", type=Path, default=Path("augmented_data/split"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = p.parse_args()
    splits = {"train": "casino_train.json", "dev": "casino_valid.json", "test": "casino_test.json"}
    original = {s: load(args.original_dir / f) for s, f in splits.items()}
    augmented = {s: load(args.augmented_dir / f) for s, f in splits.items()}
    original_ids = {str(d.get("dialogue_id")) for ds in original.values() for d in ds}
    rows = []
    for s in splits:
        rows += dialogue_records(original[s], s, "D0_original", original_ids)
        rows += dialogue_records(augmented[s], s, "D1_balanced", original_ids)

    # D1 contains D0 in its unchanged splits, so class counts distinguish real/synthetic.
    count_rows = []
    for dataset in ["D0_original", "D1_balanced"]:
        subset = [r for r in rows if r["source"] == dataset]
        for split in splits:
            for origin in ["real", "augmented"]:
                c = Counter(l for r in subset if r["split"] == split and ("augmented" if r["is_augmented"] else "real") == origin for l in r["labels"])
                for label, n in sorted(c.items()):
                    count_rows.append({"dataset": dataset, "split": split, "origin": origin, "strategy": label, "count": n})

    co = Counter()
    for r in rows:
        if r["source"] != "D1_balanced": continue
        for a, b in combinations(sorted(set(r["labels"])), 2): co[(a, b)] += 1
    co_rows = [{"strategy_a": a, "strategy_b": b, "count": n} for (a, b), n in sorted(co.items())]

    # Exact duplicate utterances; report cross-label and cross-split occurrences.
    groups = defaultdict(list)
    for r in rows:
        if r["source"] == "D1_balanced" and norm(r["text"]): groups[norm(r["text"])].append(r)
    dup_rows = []
    for text, items in groups.items():
        if len(items) < 2: continue
        dup_rows.append({
            "normalized_text": text, "occurrences": len(items),
            "dialogue_ids": "|".join(sorted({x["dialogue_id"] for x in items})),
            "splits": "|".join(sorted({x["split"] for x in items})),
            "labels": "|".join(sorted({l for x in items for l in x["labels"]})),
        })

    id_sets = {name: {str(d.get("dialogue_id")) for d in augmented[name]} for name in splits}
    leakage = []
    for a, b in combinations(splits, 2):
        for did in sorted(id_sets[a] & id_sets[b]): leakage.append({"split_a": a, "split_b": b, "dialogue_id": did})
    test_ids = id_sets["test"]
    aug_train_ids = {str(d.get("dialogue_id")) for d in augmented["train"] if str(d.get("dialogue_id")) not in original_ids}
    formal_test_overlap = sorted(aug_train_ids & test_ids)

    d1 = [r for r in rows if r["source"] == "D1_balanced"]
    label_cardinality = Counter(len(r["labels"]) for r in d1)
    by_origin = Counter("augmented" if r["is_augmented"] else "real" for r in d1)
    profile_missing = sum(not r["profile_present"] for r in d1)
    reasons_missing = sum(not r["reasons_present"] for r in d1)
    avg_len = {lab: mean([len(r["text"].split()) for r in d1 if lab in r["labels"]]) for lab in sorted({l for r in d1 for l in r["labels"]})}
    report = {
        "inputs": {"original_dir": str(args.original_dir), "augmented_dir": str(args.augmented_dir)},
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*(args.original_dir / f for f in splits.values()), *(args.augmented_dir / f for f in splits.values())]},
        "dialogues": {"D0": {s: len(original[s]) for s in splits}, "D1": {s: len(augmented[s]) for s in splits}},
        "utterances_D1": len(d1), "origin_D1": dict(by_origin),
        "label_cardinality_D1": {str(k): v for k, v in sorted(label_cardinality.items())},
        "multi_label_utterances_D1": sum(v for k, v in label_cardinality.items() if k > 1),
        "profile_missing_D1": profile_missing, "reasons_missing_D1": reasons_missing,
        "dialogue_split_leakage_count": len(leakage),
        "augmented_train_vs_formal_test_dialogue_overlap": formal_test_overlap,
        "exact_duplicate_groups_D1": len(dup_rows),
        "average_response_words_by_strategy_D1": avg_len,
        "limitations": [
            "Near-duplicate/template similarity is not inferred from exact matching and requires a second-stage semantic review.",
            "Whether a generator saw test context cannot be proven from data alone; generation script provenance was inspected separately.",
            "Strategy mixing beyond supplied labels requires a multi-label judge or human annotation.",
        ],
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "data_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(args.reports_dir / "strategy_counts.csv", count_rows, ["dataset", "split", "origin", "strategy", "count"])
    write_csv(args.reports_dir / "strategy_cooccurrence.csv", co_rows, ["strategy_a", "strategy_b", "count"])
    write_csv(args.reports_dir / "duplicate_samples.csv", dup_rows, ["normalized_text", "occurrences", "dialogue_ids", "splits", "labels"])
    write_csv(args.reports_dir / "dialogue_split_leakage.csv", leakage, ["split_a", "split_b", "dialogue_id"])
    md = ["# 数据审计（自动检查）", "", f"- D1 对话数：{report['dialogues']['D1']}", f"- D1 标注发言数：{len(d1)}；真实/增强：{dict(by_origin)}", f"- 单/多标签基数：{dict(label_cardinality)}；多标签发言：{report['multi_label_utterances_D1']}", f"- 缺少 speaker profile / reason：{profile_missing} / {reasons_missing}", f"- dialogue_id 跨正式 split 泄漏：{len(leakage)}", f"- 增强训练 dialogue 与正式 test 重合：{len(formal_test_overlap)}", f"- 完全重复文本组：{len(dup_rows)}", "", "注意：语义近重复、未标注的策略混入与生成器是否曾读取测试上下文，不能仅凭静态数据完全证明，需结合生成日志与人工/多标签 judge。", ""]
    (args.reports_dir / "data_audit.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
