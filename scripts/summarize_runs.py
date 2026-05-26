from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "run_id",
    "adapter_mode",
    "best_valid_loss",
    "last_train_loss",
    "last_gen_loss",
    "last_orth_loss",
    "last_orth_local_loss",
    "last_orth_global_loss",
    "last_cls_loss",
    "last_valid_loss",
    "last_valid_ppl",
    "last_test_loss",
    "last_test_ppl",
    "lambda_orth",
    "lambda_cls",
    "orth_alpha",
    "num_virtual_tokens",
    "lora_r",
    "seed",
    "output_dir"
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> dict:
    metrics = read_json(run_dir / "metrics.json")
    cfg = read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
    history = metrics.get("history") or []
    last = history[-1] if history else {}
    return {
        "run_id": run_dir.name,
        "adapter_mode": cfg.get("adapter_mode"),
        "best_valid_loss": metrics.get("best_valid_loss"),
        "last_train_loss": last.get("train_loss"),
        "last_gen_loss": last.get("train_gen_loss"),
        "last_orth_loss": last.get("train_orth_loss"),
        "last_orth_local_loss": last.get("train_orth_local_loss"),
        "last_orth_global_loss": last.get("train_orth_global_loss"),
        "last_cls_loss": last.get("train_cls_loss"),
        "last_valid_loss": last.get("valid_loss"),
        "last_valid_ppl": last.get("valid_ppl"),
        "last_test_loss": last.get("test_loss"),
        "last_test_ppl": last.get("test_ppl"),
        "lambda_orth": cfg.get("lambda_orth"),
        "lambda_cls": cfg.get("lambda_cls"),
        "orth_alpha": cfg.get("orth_alpha"),
        "num_virtual_tokens": cfg.get("num_virtual_tokens"),
        "lora_r": cfg.get("lora_r"),
        "seed": cfg.get("seed"),
        "output_dir": str(run_dir)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", default="run_summary.csv")
    args = parser.parse_args()

    rows = [summarize_run(path.parent) for path in sorted(Path(args.runs_root).glob("*/metrics.json"))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已写入 {len(rows)} 行到 {out.resolve()}")


if __name__ == "__main__":
    main()
