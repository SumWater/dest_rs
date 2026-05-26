from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_key(record: dict) -> tuple[str, str]:
    return str(record.get("dialogue_id")), str(record.get("turn_index"))


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 B4/B5 swap 样例的人工或 LLM judge 表格。")
    parser.add_argument("--b4-jsonl", required=True, help="B4 的 swap_samples jsonl 路径。")
    parser.add_argument("--b5-jsonl", required=True, help="B5 的 swap_samples jsonl 路径。")
    parser.add_argument("--out", default="outputs/judge_b4_vs_b5.csv", help="输出 CSV 路径。")
    parser.add_argument("--max-prompts", type=int, default=5, help="最多导出多少个 prompt。")
    parser.add_argument(
        "--strategies",
        default="elicit-pref,self-need,other-need,promote-coordination,vouch-fair",
        help="逗号分隔的目标策略列表。",
    )
    args = parser.parse_args()

    b4_records = read_jsonl(Path(args.b4_jsonl))
    b5_records = read_jsonl(Path(args.b5_jsonl))
    b5_by_key = {record_key(record): record for record in b5_records}
    strategies = [x.strip() for x in args.strategies.split(",") if x.strip()]

    rows = []
    prompt_count = 0
    for b4 in b4_records:
        key = record_key(b4)
        b5 = b5_by_key.get(key)
        if b5 is None:
            continue
        prompt_count += 1
        if prompt_count > args.max_prompts:
            break

        b4_generated = b4.get("generated_by_strategy") or {}
        b5_generated = b5.get("generated_by_strategy") or {}
        for strategy in strategies:
            rows.append(
                {
                    "dialogue_id": b4.get("dialogue_id"),
                    "turn_index": b4.get("turn_index"),
                    "gold_strategy": b4.get("gold_strategy"),
                    "target_strategy": strategy,
                    "prompt": b4.get("prompt") or "",
                    "gold_target": b4.get("gold_target") or "",
                    "b4_response": b4_generated.get(strategy, ""),
                    "b5_response": b5_generated.get(strategy, ""),
                    "b4_match": "",
                    "b5_match": "",
                    "which_better": "",
                    "judge_note": "",
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dialogue_id",
        "turn_index",
        "gold_strategy",
        "target_strategy",
        "prompt",
        "gold_target",
        "b4_response",
        "b5_response",
        "b4_match",
        "b5_match",
        "which_better",
        "judge_note",
    ]
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已写入 {len(rows)} 行到 {out.resolve()}")
    print("建议标注规则：b4_match/b5_match 填 1 或 0，which_better 填 b4/b5/tie。")


if __name__ == "__main__":
    main()
