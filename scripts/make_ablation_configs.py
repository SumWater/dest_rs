from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


RUNS = {
    "b2_lora_only": {"adapter_mode": "lora_only", "lambda_orth": 0.0, "lambda_cls": 0.0},
    "b3_prefix_only": {"adapter_mode": "prefix_only", "lambda_orth": 0.0, "lambda_cls": 0.0},
    "b4_prefix_lora": {"adapter_mode": "prefix_lora", "lambda_orth": 0.0, "lambda_cls": 0.0},
    "b5_prefix_lora_orth": {"adapter_mode": "prefix_lora_orth", "lambda_orth": 0.05, "lambda_cls": 0.0},
    "b6_dest_rs": {"adapter_mode": "dest_rs", "lambda_orth": 0.05, "lambda_cls": 0.2}
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="configs/b6_dest_rs.json")
    parser.add_argument("--out-dir", default="configs/generated")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-root", default="./outputs")
    args = parser.parse_args()

    template = json.loads(Path(args.template).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for run_id, updates in RUNS.items():
        cfg = deepcopy(template)
        cfg.update(updates)
        cfg["output_dir"] = str(Path(args.output_root) / run_id).replace("\\", "/")
        if args.model_path:
            cfg["model_name_or_path"] = args.model_path
        if args.dataset_dir:
            cfg["dataset_dir"] = args.dataset_dir
        path = out_dir / f"{run_id}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
