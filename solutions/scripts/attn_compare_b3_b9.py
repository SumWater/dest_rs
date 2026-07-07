#!/usr/bin/env python3
"""
B3 vs B9 Attention Pattern Comparison (S3 prerequisite).

Extracts and compares attention weights from B3 (Prefix-only) and B9
(Frozen Prefix + LoRA) on the same test samples. Diagnoses whether
LoRA compresses, redistributes, or amplifies Prefix attention weights.

This analysis determines whether S3 (attention gating) is the right fix:
  - Compressed → S3 gating amplifies → likely to help
  - Redistributed → S3 gating won't help (issue is WHERE, not HOW MUCH)
  - Amplified → interference is not via attention compression at all

Usage (from project root):
  python solutions/scripts/attn_compare_b3_b9.py \
      --b3-checkpoint solutions/output/other/casino_augmented/b3_prefix_only \
      --b9-checkpoint solutions/output/other/casino_augmented/b9_prefix_then_lora \
      --num-samples 20 \
      --output solutions/output/attn_comparison.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
SOLUTIONS_OUT = ROOT / "solutions" / "output"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_hybrid_model(checkpoint_dir: str, label_space=None, model_path: str = None):
    """Load HybridStrategyModel from a checkpoint directory.

    Args:
        checkpoint_dir: path to output/other/{dataset_tag}/{exp_name}/
        label_space: StrategyLabelSpace (auto-loaded from prefix_bank.pt if None)
        model_path: optional override for base model path

    Returns:
        (model, tokenizer, label_space, adapter_mode)
    """
    checkpoint_dir = Path(checkpoint_dir)

    prefix_path = checkpoint_dir / "prefix_bank.pt"
    if not prefix_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {prefix_path}\n"
            f"Expected: {checkpoint_dir}/\n"
            f"  ├── prefix_bank.pt\n"
            f"  ├── lora_adapter/\n"
            f"  └── tokenizer/\n"
            f"Run training first to produce this checkpoint."
        )

    # Load prefix bank to get labels
    ckpt = torch.load(prefix_path, map_location="cpu", weights_only=False)
    labels = ckpt.get("labels", [])
    adapter_mode = ckpt.get("adapter_mode", "prefix_only")

    from src.casino_dataset import StrategyLabelSpace
    if label_space is None:
        label_space = StrategyLabelSpace(labels)

    # Infer model path (use override if provided)
    resolved_model_path = model_path or _infer_model_path(checkpoint_dir)

    from src.config import TrainConfig
    cfg = TrainConfig()
    cfg.model_name_or_path = resolved_model_path
    cfg.adapter_mode = adapter_mode

    # Build model
    from src.modeling import build_hybrid_model, load_tokenizer

    model, tokenizer, _ = build_hybrid_model(cfg, num_strategies=len(label_space.labels))

    # Load saved prefix_bank
    saved_prefix = ckpt["prefix_bank"]
    model.prefix_bank.data.copy_(saved_prefix.to(model.prefix_bank.device))

    # Load LoRA adapter if exists
    lora_path = checkpoint_dir / "lora_adapter"
    if lora_path.exists():
        from peft import PeftModel
        try:
            model.peft_model = PeftModel.from_pretrained(
                model.peft_model, str(lora_path),
                is_trainable=False,  # inference only
            )
            print(f"  Loaded LoRA adapter from {lora_path}")
        except Exception as e:
            print(f"  Warning: Could not load LoRA adapter: {e}")

    # Load tokenizer
    tok_path = checkpoint_dir / "tokenizer"
    if tok_path.exists():
        try:
            tokenizer = type(tokenizer).from_pretrained(str(tok_path))
        except Exception:
            pass

    model.eval()
    return model, tokenizer, label_space, adapter_mode


def _infer_model_path(checkpoint_dir: Path) -> str:
    """Infer base model path from saved configs or common locations."""
    import glob

    # Try run_config.json in corresponding need/ directory
    parts = checkpoint_dir.parts
    for i, part in enumerate(parts):
        if part == "other":
            need_parts = list(parts[:i]) + ["need"] + list(parts[i+1:])
            need_dir = Path(*need_parts)
            config_path = need_dir / "run_config.json"
            if config_path.exists():
                cfg = json.load(open(config_path, "r", encoding="utf-8"))
                if cfg.get("model_name_or_path"):
                    return cfg["model_name_or_path"]

    # Search for any run_config.json in the project (old + new output roots)
    for search_root in ["output/need", "solutions/output/need"]:
        for config_path in ROOT.glob(f"{search_root}/*/*/run_config.json"):
            try:
                cfg = json.load(open(config_path, "r", encoding="utf-8"))
                if cfg.get("model_name_or_path"):
                    return cfg["model_name_or_path"]
            except Exception:
                pass

    # Try common paths
    candidates = [
        os.path.expanduser("~/models/Qwen2.5-8B-Instruct"),
        os.path.expanduser("~/models/Qwen2.5-7B-Instruct"),
        "/replace/with/your/local/qwen/path",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        "Cannot find base model. Check run_config.json in "
        "output/need/*/*/ or solutions/output/need/*/*/, "
        "or pass --model-path explicitly."
    )


def compare_attention(
    b3_checkpoint_dir: str,
    b9_checkpoint_dir: str,
    dataset_tag: str = "casino_augmented_new_fix_seed42",
    num_samples: int = 20,
    num_layers: int = 32,
    output_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict:
    """Compare B3 vs B9 attention patterns on test samples.

    Returns per-layer statistics and a global conclusion about whether
    LoRA compresses or redistributes Prefix attention weights.
    """
    print("=" * 60)
    print("B3 vs B9 Attention Pattern Comparison")
    print("=" * 60)

    # Load test data
    from src.casino_dataset import load_split_examples, CasinoStrategyDataset
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.dataset_dir = str(ROOT / "augmented_data")
    if model_path:
        cfg.model_name_or_path = model_path
    cfg.dataset_tag = dataset_tag

    test_examples = load_split_examples(cfg, "test")
    print(f"[data] Loaded {len(test_examples)} test examples")

    # Load B3
    print(f"\n[model] Loading B3 from {b3_checkpoint_dir}...")
    model_b3, tokenizer, label_space, mode_b3 = load_hybrid_model(
        b3_checkpoint_dir, model_path=model_path
    )
    prefix_b3 = model_b3.prefix_bank.detach()

    # Load B9
    print(f"[model] Loading B9 from {b9_checkpoint_dir}...")
    model_b9, _, _, mode_b9 = load_hybrid_model(
        b9_checkpoint_dir, label_space=label_space, model_path=model_path
    )
    prefix_b9 = model_b9.prefix_bank.detach()

    # Build dataset for consistent sample selection
    dataset = CasinoStrategyDataset(test_examples, label_space)

    # Select samples
    if num_samples and num_samples < len(dataset):
        step = max(1, len(dataset) // num_samples)
        indices = list(range(0, len(dataset), step))[:num_samples]
    else:
        indices = list(range(len(dataset)))

    print(f"\n[compare] Analyzing {len(indices)} samples across {num_layers} layers...")

    from solutions.src.attention_utils import extract_attention_weights

    device_b3 = next(model_b3.parameters()).device
    device_b9 = next(model_b9.parameters()).device

    # Accumulate per-layer prefix attention weights
    per_layer_b3 = {i: [] for i in range(num_layers)}
    per_layer_b9 = {i: [] for i in range(num_layers)}

    # We'll analyze only a subset of layers to save memory/time
    # Sample every 4th layer plus first and last
    target_layers = sorted(set(
        [0, num_layers - 1] +
        list(range(4, num_layers - 4, 4))
    ))

    for idx_idx, idx in enumerate(indices):
        sample = dataset[idx]
        input_ids = tokenizer(
            sample["prompt"], return_tensors="pt", add_special_tokens=False
        )["input_ids"]
        attn_mask = torch.ones_like(input_ids)
        strategy_id = sample["strategy_id"]

        prefix_emb = prefix_b3[strategy_id].to(device_b3)

        # Extract B3 attention (only target layers)
        try:
            with torch.no_grad():
                attn_b3 = extract_attention_weights(
                    model_b3,
                    input_ids.to(device_b3),
                    attn_mask.to(device_b3),
                    prefix_emb,
                    target_layers=target_layers,
                )
        except Exception as e:
            print(f"  Warning: B3 attention extraction failed for sample {idx}: {e}")
            continue

        # Extract B9 attention
        prefix_emb_b9 = prefix_b9[strategy_id].to(device_b9)
        try:
            with torch.no_grad():
                attn_b9 = extract_attention_weights(
                    model_b9,
                    input_ids.to(device_b9),
                    attn_mask.to(device_b9),
                    prefix_emb_b9,
                    target_layers=target_layers,
                )
        except Exception as e:
            print(f"  Warning: B9 attention extraction failed for sample {idx}: {e}")
            continue

        # Accumulate per-layer prefix attention weights
        for layer_idx in target_layers:
            key = f"layer_{layer_idx}_prefix_weight"
            if key in attn_b3:
                # Mean prefix attention weight across heads and text positions
                mean_w = attn_b3[key].mean().item()
                per_layer_b3[layer_idx].append(mean_w)
            else:
                per_layer_b3[layer_idx].append(0.0)

            if key in attn_b9:
                mean_w = attn_b9[key].mean().item()
                per_layer_b9[layer_idx].append(mean_w)
            else:
                per_layer_b9[layer_idx].append(0.0)

        if (idx_idx + 1) % 5 == 0:
            print(f"  Progress: {idx_idx + 1}/{len(indices)} samples")

    # Compute summary statistics
    summary = []
    for layer_idx in target_layers:
        b3_vals = per_layer_b3[layer_idx]
        b9_vals = per_layer_b9[layer_idx]

        if not b3_vals or not b9_vals:
            continue

        b3_mean = sum(b3_vals) / len(b3_vals)
        b9_mean = sum(b9_vals) / len(b9_vals)

        if b3_mean > 0:
            ratio = b9_mean / b3_mean
        else:
            ratio = float('inf') if b9_mean > 0 else 1.0

        if b9_mean < b3_mean * 0.8:
            conclusion = "compressed"
        elif abs(b9_mean - b3_mean) < b3_mean * 0.2:
            conclusion = "redistributed"
        else:
            conclusion = "amplified"

        summary.append({
            "layer": layer_idx,
            "b3_mean_prefix_attn": round(b3_mean, 6),
            "b9_mean_prefix_attn": round(b9_mean, 6),
            "ratio_b9_to_b3": round(ratio, 4),
            "conclusion": conclusion,
        })

        conf_str = {"compressed": "↓", "redistributed": "→", "amplified": "↑"}[conclusion]
        print(f"  Layer {layer_idx:2d}: B3={b3_mean:.4f} B9={b9_mean:.4f} "
              f"ratio={ratio:.3f} {conf_str} {conclusion}")

    # Global conclusion
    ratios = [s["ratio_b9_to_b3"] for s in summary]
    if not ratios:
        global_conclusion = "INCONCLUSIVE: No data collected"
    else:
        avg_ratio = sum(ratios) / len(ratios)
        compressed_count = sum(1 for s in summary if s["conclusion"] == "compressed")
        redistributed_count = sum(1 for s in summary if s["conclusion"] == "redistributed")
        amplified_count = sum(1 for s in summary if s["conclusion"] == "amplified")

        total = len(summary)
        if compressed_count > total * 0.6:
            global_conclusion = (
                f"GLOBALLY COMPRESSED: {compressed_count}/{total} layers show "
                f"compressed Prefix attention (avg ratio={avg_ratio:.3f}). "
                f"S3 gating is likely to help by restoring Prefix attention weight."
            )
        elif redistributed_count > total * 0.6:
            global_conclusion = (
                f"GLOBALLY REDISTRIBUTED: {redistributed_count}/{total} layers show "
                f"redistributed attention (avg ratio={avg_ratio:.3f}). "
                f"S3 gating may NOT help — the issue is WHERE attention goes, "
                f"not how much. Consider S2 (parameter orth) or S4 (CPD) instead."
            )
        elif amplified_count > total * 0.6:
            global_conclusion = (
                f"GLOBALLY AMPLIFIED: {amplified_count}/{total} layers show "
                f"amplified Prefix attention (avg ratio={avg_ratio:.3f}). "
                f"Interference is NOT via attention compression. "
                f"Focus on S2 (parameter orthogonality) or S4 (CPD)."
            )
        else:
            global_conclusion = (
                f"MIXED PATTERN: {compressed_count}c/{redistributed_count}r/"
                f"{amplified_count}a across {total} layers "
                f"(avg ratio={avg_ratio:.3f}). Layer-specific analysis needed."
            )

    print(f"\n{'='*60}")
    print(f"CONCLUSION: {global_conclusion}")
    print(f"{'='*60}")

    results = {
        "b3_checkpoint": str(b3_checkpoint_dir),
        "b9_checkpoint": str(b9_checkpoint_dir),
        "num_samples": len(indices),
        "target_layers": target_layers,
        "summary": summary,
        "layer_counts": {
            "compressed": sum(1 for s in summary if s["conclusion"] == "compressed"),
            "redistributed": sum(1 for s in summary if s["conclusion"] == "redistributed"),
            "amplified": sum(1 for s in summary if s["conclusion"] == "amplified"),
        },
        "global_conclusion": global_conclusion,
        "recommendation": _get_recommendation(summary),
    }

    # Save
    if output_path is None:
        output_path = SOLUTIONS_OUT / "attn_comparison.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    return results


def _get_recommendation(summary: List[Dict]) -> str:
    """Generate actionable recommendation based on attention comparison results."""
    compressed = sum(1 for s in summary if s["conclusion"] == "compressed")
    redistributed = sum(1 for s in summary if s["conclusion"] == "redistributed")
    amplified = sum(1 for s in summary if s["conclusion"] == "amplified")
    total = len(summary)

    if total == 0:
        return "No data available. Ensure checkpoints exist and attention extraction succeeds."

    if compressed > total * 0.5:
        return (
            "S3 (attention gating) is the PRIMARY recommendation. "
            "LoRA compresses Prefix attention weights, and gating can "
            "restore them. Also try S4 (CPD) which operates at logit-level "
            "and may complement gating."
        )
    elif redistributed > total * 0.5:
        return (
            "S2 (parameter orthogonality) is the PRIMARY recommendation. "
            "LoRA redistributes attention without compressing it — the issue "
            "is in the Q/K/V/O projection space, not attention weights. "
            "S4 (CPD) is a secondary option for inference-only mitigation."
        )
    elif amplified > total * 0.5:
        return (
            "S2 (parameter orthogonality) is the PRIMARY recommendation. "
            "Attention is amplified, suggesting the interference is at the "
            "projection level (Q/K/V/O matrices). S4 (CPD) may help as "
            "a lightweight inference-only alternative."
        )
    else:
        return (
            "MIXED recommendation: run S2 (parameter orth) first, then S3 "
            "(gating) if compression pattern emerges after training. "
            "S4 (CPD) as a zero-cost diagnostic baseline."
        )


def main():
    parser = argparse.ArgumentParser(
        description="B3 vs B9 Attention Pattern Comparison"
    )
    parser.add_argument(
        "--b3-checkpoint", type=str, required=True,
        help="Path to B3 checkpoint directory (solutions/output/other/.../b3_prefix_only)"
    )
    parser.add_argument(
        "--b9-checkpoint", type=str, required=True,
        help="Path to B9 checkpoint directory (solutions/output/other/.../b9_prefix_then_lora)"
    )
    parser.add_argument(
        "--dataset-tag", type=str, default="casino_augmented_new_fix_seed42",
        help="Dataset tag for loading test data"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of test samples to analyze (default: 20)"
    )
    parser.add_argument(
        "--num-layers", type=int, default=32,
        help="Number of transformer layers in the model (default: 32)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: solutions/output/attn_comparison.json)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Base model path (auto-detected if not provided)"
    )
    args = parser.parse_args()

    # Validate checkpoint paths
    for label, path in [("B3", args.b3_checkpoint), ("B9", args.b9_checkpoint)]:
        p = Path(path)
        if not (p / "prefix_bank.pt").exists():
            print(f"ERROR: {label} checkpoint not found at {p}")
            print(f"  Expected: {p / 'prefix_bank.pt'}")
            print(f"  Make sure you've run training for {label} first.")
            print(f"  Checkpoint is saved to solutions/output/other/{{dataset_tag}}/{{exp_name}}/")
            sys.exit(1)

    compare_attention(
        b3_checkpoint_dir=args.b3_checkpoint,
        b9_checkpoint_dir=args.b9_checkpoint,
        dataset_tag=args.dataset_tag,
        num_samples=args.num_samples,
        num_layers=args.num_layers,
        output_path=args.output,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
