from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config, resolve_warm_start_dir
from src.modeling import build_hybrid_model, freeze_for_adapter_mode, trainable_partition
from src.strategy_labels import load_canonical_labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", default="reports/remote_checks/b9_checkpoint_check.json")
    args = parser.parse_args()

    report: dict = {"status": "FAIL", "errors": []}
    try:
        cfg = load_config(args.config)
        cfg.seed = args.seed
        cfg.warm_start_dir = args.checkpoint
        cfg.warm_start_prefix = True
        cfg.warm_start_lora = True
        cfg.require_warm_start_prefix = True
        cfg.require_warm_start_lora = True
        cfg.train_prefix = True
        cfg.train_lora = False
        cfg.train_classifier = False
        cfg.freeze_prefix = False
        resolve_warm_start_dir(cfg)
        checkpoint = Path(cfg.warm_start_dir).resolve()
        labels = load_canonical_labels(cfg.strategy_label_space_path)
        files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
        report.update({
            "config_path": str(Path(args.config).resolve()),
            "checkpoint_path": str(checkpoint),
            "seed": cfg.seed,
            "base_model_path": cfg.model_name_or_path,
            "canonical_labels": labels,
            "files": [{"path": str(path.relative_to(checkpoint)), "sha256": sha256(path)} for path in files],
        })
        prefix_payload = torch.load(checkpoint / "prefix_bank.pt", map_location="cpu")
        prefix = prefix_payload["prefix_bank"]
        report["prefix"] = {
            "parameter_names": ["prefix_bank"],
            "shape": list(prefix.shape),
            "dtype": str(prefix.dtype),
            "norm": float(prefix.float().norm().item()),
            "labels": prefix_payload.get("labels"),
        }
        hybrid, _, catcher = build_hybrid_model(cfg, len(labels), expected_labels=labels)
        catcher.remove()
        freeze_for_adapter_mode(hybrid, cfg)
        partitions = trainable_partition(hybrid)
        report["trainable"] = {
            key: {"count": sum(p.numel() for _, p in values), "names": [name for name, _ in values]}
            for key, values in partitions.items()
        }
        lora = [(name, param) for name, param in hybrid.named_parameters() if "lora_" in name.lower()]
        report["lora"] = {
            "adapter_names": sorted({name.split(".lora_")[0] for name, _ in lora}),
            "parameter_names": [name for name, _ in lora],
            "parameter_count": sum(param.numel() for _, param in lora),
            "norm": float(torch.sqrt(sum(param.detach().float().pow(2).sum() for _, param in lora)).item()),
        }
        report["missing_keys"] = None
        report["unexpected_keys"] = None
        report["load_key_report_note"] = (
            "PeftModel.from_pretrained completed successfully but does not expose an incompatible-keys "
            "object through this call; file completeness, adapter parameter inventory and counts are checked instead."
        )
        counts = {key: value["count"] for key, value in report["trainable"].items()}
        if counts != {"prefix": prefix.numel(), "lora": 0, "classifier": 0, "base": 0}:
            raise RuntimeError(f"Unexpected M-series trainable partition: {counts}")
        report["status"] = "PASS"
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
