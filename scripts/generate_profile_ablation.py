"""Generate the P0 no-profile side of the B9 profile ablation.

P1 is the already-saved B9 seed-42 swap file. P0 uses the identical checkpoint,
examples, strategy IDs and greedy decoding, changing only include_profile=False.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.casino_dataset import CasinoStrategyDataset, StrategyLabelSpace, load_split_examples
from src.config import load_config, resolve_warm_start_dir
from src.evaluate import save_swap_samples
from src.modeling import build_hybrid_model, freeze_for_adapter_mode


def keys(path: Path):
    result = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            result.append((item.get("dialogue_id"), item.get("turn_index")))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--reference-p1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-examples", type=int, default=30)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for B9 generation")
    cfg = load_config(args.config)
    cfg.warm_start_dir = args.checkpoint
    cfg.warm_start_prefix = True
    cfg.warm_start_lora = True
    cfg.include_profile = False
    cfg.demo_num_examples = args.num_examples
    cfg.demo_temperature = 0.0
    resolve_warm_start_dir(cfg)

    checkpoint = Path(cfg.warm_start_dir)
    required = [checkpoint / "prefix_bank.pt", checkpoint / "lora_adapter" / "adapter_model.safetensors"]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete B9 checkpoint: {missing}")

    label_space = StrategyLabelSpace.from_json(json.loads(Path(args.label_map).read_text(encoding="utf-8")))
    examples = load_split_examples(cfg, "valid")
    dataset = CasinoStrategyDataset(examples, label_space)
    expected = [(dataset[i]["dialogue_id"], dataset[i]["turn_index"]) for i in range(min(args.num_examples, len(dataset)))]
    reference = keys(args.reference_p1)
    if expected != reference[:len(expected)]:
        raise RuntimeError("P0/P1 example keys differ; refusing an unpaired ablation")

    print(f"checkpoint={checkpoint}")
    print("include_profile=False")
    print(f"paired_contexts={len(expected)}")
    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    catcher.remove(); freeze_for_adapter_mode(hybrid, cfg); hybrid.eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        save_swap_samples(hybrid, tokenizer, dataset, label_space, cfg, str(args.out),
                          args.num_examples, "valid")
    # Verify that the private profile block is really absent.
    with args.out.open(encoding="utf-8") as f:
        first = json.loads(next(f))
    if "Target speaker private profile:" in first.get("prompt", ""):
        raise RuntimeError("Profile removal failed")
    print(f"saved={args.out}")


if __name__ == "__main__": main()
