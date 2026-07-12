from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train import build_optimizer
from src.config import load_config, resolve_warm_start_dir
from src.modeling import assert_trainable_partition, build_hybrid_model, freeze_for_adapter_mode
from src.strategy_labels import load_canonical_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="reports/remote_checks/trainable_params_check.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.warm_start_dir = args.checkpoint
    cfg.warm_start_prefix = cfg.warm_start_lora = True
    cfg.require_warm_start_prefix = cfg.require_warm_start_lora = True
    cfg.train_prefix, cfg.train_lora, cfg.train_classifier, cfg.freeze_prefix = True, False, False, False
    resolve_warm_start_dir(cfg)
    labels = load_canonical_labels(cfg.strategy_label_space_path)
    hybrid, _, catcher = build_hybrid_model(cfg, len(labels), expected_labels=labels)
    catcher.remove()
    freeze_for_adapter_mode(hybrid, cfg)
    counts = assert_trainable_partition(hybrid, base=0, lora=0, classifier=0, prefix_positive=True)
    optimizer = build_optimizer(hybrid, cfg)
    payload = {
        "status": "PASS",
        "counts": counts,
        "all_trainable_parameter_names": [name for name, p in hybrid.named_parameters() if p.requires_grad],
        "optimizer_parameter_names": [
            name for name, p in hybrid.named_parameters()
            if any(p is item for group in optimizer.param_groups for item in group["params"])
        ],
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
