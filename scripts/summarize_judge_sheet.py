from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_binary(value: str) -> int | None:
    text = (value or "").strip().lower()
    if text in {"1", "yes", "y", "true", "是", "符合"}:
        return 1
    if text in {"0", "no", "n", "false", "否", "不符合"}:
        return 0
    return None


def mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def summarize(rows: list[dict]) -> dict:
    valid_rows = []
    for row in rows:
        b4 = parse_binary(row.get("b4_match", ""))
        b5 = parse_binary(row.get("b5_match", ""))
        if b4 is None or b5 is None:
            continue
        copied = dict(row)
        copied["_b4"] = b4
        copied["_b5"] = b5
        copied["_which"] = (row.get("which_better") or "").strip().lower()
        valid_rows.append(copied)

    b4_values = [row["_b4"] for row in valid_rows]
    b5_values = [row["_b5"] for row in valid_rows]
    wins = Counter(row["_which"] for row in valid_rows if row["_which"])
    total = len(valid_rows)

    return {
        "total_labeled": total,
        "b4_match_rate": mean(b4_values),
        "b5_match_rate": mean(b5_values),
        "b5_minus_b4": mean(b5_values) - mean(b4_values),
        "b4_win_rate": wins.get("b4", 0) / total if total else 0.0,
        "b5_win_rate": wins.get("b5", 0) / total if total else 0.0,
        "tie_rate": wins.get("tie", 0) / total if total else 0.0,
        "valid_rows": valid_rows,
    }


def print_overall(summary: dict) -> None:
    print("=" * 80)
    print("整体标注结果")
    print("=" * 80)
    print(f"有效标注行数：{summary['total_labeled']}")
    print(f"B4 策略命中率：{pct(summary['b4_match_rate'])}")
    print(f"B5 策略命中率：{pct(summary['b5_match_rate'])}")
    print(f"B5 - B4 命中率差值：{pct(summary['b5_minus_b4'])}")
    print(f"B4 更优比例：{pct(summary['b4_win_rate'])}")
    print(f"B5 更优比例：{pct(summary['b5_win_rate'])}")
    print(f"两者相当比例：{pct(summary['tie_rate'])}")


def print_by_strategy(valid_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in valid_rows:
        grouped[row.get("target_strategy", "")].append(row)

    table = []
    print()
    print("=" * 80)
    print("按目标策略分组")
    print("=" * 80)
    print("strategy,b4_match,b5_match,b5_minus_b4,b4_win,b5_win,tie,count")
    for strategy in sorted(grouped):
        rows = grouped[strategy]
        b4_values = [row["_b4"] for row in rows]
        b5_values = [row["_b5"] for row in rows]
        wins = Counter(row["_which"] for row in rows if row["_which"])
        count = len(rows)
        item = {
            "target_strategy": strategy,
            "b4_match_rate": mean(b4_values),
            "b5_match_rate": mean(b5_values),
            "b5_minus_b4": mean(b5_values) - mean(b4_values),
            "b4_win_rate": wins.get("b4", 0) / count if count else 0.0,
            "b5_win_rate": wins.get("b5", 0) / count if count else 0.0,
            "tie_rate": wins.get("tie", 0) / count if count else 0.0,
            "count": count,
        }
        table.append(item)
        print(
            f"{strategy},"
            f"{pct(item['b4_match_rate'])},"
            f"{pct(item['b5_match_rate'])},"
            f"{pct(item['b5_minus_b4'])},"
            f"{pct(item['b4_win_rate'])},"
            f"{pct(item['b5_win_rate'])},"
            f"{pct(item['tie_rate'])},"
            f"{count}"
        )
    return table


def write_strategy_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_strategy",
        "b4_match_rate",
        "b5_match_rate",
        "b5_minus_b4",
        "b4_win_rate",
        "b5_win_rate",
        "tie_rate",
        "count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 B4/B5 judge 标注表。")
    parser.add_argument("--csv", required=True, help="已标注的 judge CSV 文件。")
    parser.add_argument("--out-by-strategy", default=None, help="可选：输出按策略分组统计 CSV。")
    args = parser.parse_args()

    with Path(args.csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    summary = summarize(rows)
    print_overall(summary)
    by_strategy = print_by_strategy(summary["valid_rows"])

    skipped = len(rows) - summary["total_labeled"]
    if skipped:
        print()
        print(f"提示：有 {skipped} 行未统计，因为 b4_match 或 b5_match 为空/非法。")

    if args.out_by_strategy:
        write_strategy_summary(Path(args.out_by_strategy), by_strategy)
        print(f"按策略分组结果已写入：{args.out_by_strategy}")


if __name__ == "__main__":
    main()
