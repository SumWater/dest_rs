#!/usr/bin/env python3
"""
S1: Reverse Curriculum — Train LoRA first, then freeze + train Prefix.

Training order:
  Stage 1: LoRA only (no Prefix) → learns domain-adapted attention projections
  Stage 2: Freeze LoRA + Base model → train Prefix only from scratch

Key hypothesis: Prefix trained on LoRA-modified attention space will learn
to navigate the modified Q/K/V/O projections natively, avoiding the
"environment change" interference discovered in B9.

Usage (from project root):
  python solutions/scripts/run_s1_reverse.py [--stage1-only] [--stage2-only]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SOLUTIONS_OUT = ROOT / "solutions" / "output"
TRAIN_SCRIPT = ROOT / "train.py"
OUTPUT_OTHER = SOLUTIONS_OUT / "other"   # S1 checkpoints under solutions/output/other/
OUTPUT_NEED  = SOLUTIONS_OUT / "need"    # S1 eval results under solutions/output/need/

STAGE1_CFG = ROOT / "solutions" / "configs" / "s1_reverse.json"
# Stage 2 reuses the same config but with different adapter settings


def run_stage1(dataset_tag: str = "casino_augmented"):
    """
    Stage 1: Train LoRA only (no Prefix).
    """
    exp_name = "s1_clean_stage1_lora_only"

    # Build a minimal LoRA-only config by modifying the S1 config
    with open(STAGE1_CFG) as f:
        cfg = json.load(f)

    cfg["experiment_name"] = exp_name
    cfg["output_root"] = "solutions/output"
    cfg["adapter_mode"] = "lora_only"
    cfg["dataset_tag"] = dataset_tag
    cfg["enable_prefix"] = False
    cfg["enable_lora"] = True
    cfg["inject_strategy_text"] = False
    cfg["num_epochs"] = 2
    cfg["warm_start_dir"] = None  # Stage 1: no warm start

    # Save temp config
    tmp_cfg_path = SOLUTIONS_OUT / "s1_clean_stage1_config.json"
    SOLUTIONS_OUT.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[S1 Stage 1] Training LoRA only → {exp_name}")
    print(f"  Config: {tmp_cfg_path}")
    print(f"  Dataset: {dataset_tag}")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(tmp_cfg_path),
        "--dataset-tag", dataset_tag,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    # Return checkpoint path for Stage 2
    other_dir = OUTPUT_OTHER / dataset_tag / exp_name
    return other_dir if other_dir.exists() else None


def run_stage2(
    stage1_checkpoint_dir: Path,
    dataset_tag: str = "casino_augmented",
):
    """
    Stage 2: Freeze LoRA + Base, train Prefix only.

    Loads Stage 1 LoRA checkpoint, keeps it frozen, and trains
    strategy-specific Prefix embeddings from scratch.
    """
    exp_name = "s1_clean_stage2_prefix_on_frozen_lora"

    with open(STAGE1_CFG) as f:
        cfg = json.load(f)

    cfg["experiment_name"] = exp_name
    cfg["output_root"] = "solutions/output"
    cfg["adapter_mode"] = "prefix_only"
    cfg["dataset_tag"] = dataset_tag
    cfg["enable_prefix"] = True
    cfg["enable_lora"] = True       # Keep LoRA active in forward pass (frozen for training)
    cfg["train_prefix"] = True      # Explicit: Prefix is trainable
    cfg["train_lora"] = False       # Explicit: LoRA is frozen
    cfg["train_classifier"] = False
    cfg["inject_strategy_text"] = False
    cfg["num_epochs"] = 2

    # Warm-start: load Stage 1 LoRA but NOT its random prefix
    cfg["warm_start_dir"] = str(stage1_checkpoint_dir)
    cfg["warm_start_lora"] = True
    cfg["warm_start_prefix"] = False

    tmp_cfg_path = SOLUTIONS_OUT / "s1_clean_stage2_config.json"
    with open(tmp_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n[S1 Stage 2] Training Prefix on frozen LoRA → {exp_name}")
    print(f"  Config: {tmp_cfg_path}")
    print(f"  Warm-start LoRA from: {stage1_checkpoint_dir}")

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--config", str(tmp_cfg_path),
        "--dataset-tag", dataset_tag,
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    need_dir = OUTPUT_NEED / dataset_tag / exp_name
    return need_dir


def main():
    parser = argparse.ArgumentParser(description="S1: Reverse Curriculum Training")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--dataset-tag", default="casino_augmented")
    parser.add_argument("--stage1-checkpoint", type=str, default=None,
                        help="Path to Stage 1 checkpoint (for --stage2-only)")
    args = parser.parse_args()

    if args.stage2_only:
        if not args.stage1_checkpoint:
            print("ERROR: --stage2-only requires --stage1-checkpoint")
            sys.exit(1)
        run_stage2(Path(args.stage1_checkpoint), args.dataset_tag)
        return

    # Stage 1
    ckpt = run_stage1(args.dataset_tag)
    if ckpt is None:
        print("ERROR: Stage 1 failed to produce checkpoint")
        sys.exit(1)

    if args.stage1_only:
        print(f"\n[S1] Stage 1 complete. Checkpoint: {ckpt}")
        return

    # Stage 2
    run_stage2(ckpt, args.dataset_tag)
    print("\n[S1] Both stages complete.")


if __name__ == "__main__":
    main()
