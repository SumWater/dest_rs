#!/usr/bin/env python3
"""
S4: Contrastive Prefix Decoding — Zero-training inference enhancement.

Evaluates whether Contrastive Prefix Decoding (CPD) can recover strategy
control accuracy without any additional training. At each decode step,
amplifies the strategy-specific logit component by subtracting a neutral
prefix distribution:

    logit_final = logit_target + alpha * (logit_target - logit_neutral)

Key hypothesis: If LoRA's "environment change" is a uniform distribution
shift, subtracting the mean (neutral) prefix distribution will isolate the
strategy-specific signal that was being drowned.

Requires: Existing checkpoint with trained prefix_bank.pt (B3, B4, B7, or B9).

Usage (from project root):
  python solutions/scripts/run_s4_cpd_eval.py \
      --checkpoint solutions/output/other/casino_augmented/b3_prefix_only \
      --dataset-tag casino_augmented_new_fix_seed42 \
      --max-samples 50
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
SOLUTIONS_OUT = ROOT / "solutions" / "output"

# Add project root to path for reusing existing infrastructure
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_model_and_tokenizer(checkpoint_dir: str, model_path: Optional[str] = None):
    """Load a saved HybridStrategyModel + tokenizer from checkpoint directory.

    Args:
        checkpoint_dir: path to checkpoint directory
        model_path: optional override for base model path
    """
    checkpoint_dir = Path(checkpoint_dir)

    prefix_path = checkpoint_dir / "prefix_bank.pt"
    if not prefix_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {prefix_path}\n"
            f"Expected directory structure: {checkpoint_dir}/\n"
            f"  ├── prefix_bank.pt\n"
            f"  ├── lora_adapter/\n"
            f"  └── tokenizer/\n"
            f"Run training first to generate checkpoints."
        )

    # Load tokenizer
    from src.modeling import load_tokenizer
    from src.config import TrainConfig

    # Minimal config for model building
    cfg = TrainConfig()
    cfg.model_name_or_path = model_path or _infer_model_path(checkpoint_dir)

    tokenizer = load_tokenizer(cfg)
    if (checkpoint_dir / "tokenizer").exists():
        tokenizer = tokenizer.from_pretrained(str(checkpoint_dir / "tokenizer"))

    # Load prefix bank
    ckpt = torch.load(prefix_path, map_location="cpu", weights_only=False)
    prefix_bank = ckpt["prefix_bank"]  # [S, K, d]
    labels = ckpt.get("labels", [])
    adapter_mode = ckpt.get("adapter_mode", "prefix_only")

    # Build label map
    from src.casino_dataset import StrategyLabelSpace
    label_space = StrategyLabelSpace(labels)

    # Load model (full model is expensive; for CPD we mainly need prefix_bank + tokenizer)
    # We attempt to load the actual model if lora_adapter exists
    model = None
    lora_adapter_path = checkpoint_dir / "lora_adapter"
    if lora_adapter_path.exists():
        try:
            from src.modeling import build_hybrid_model
            model, _, _ = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
            model.prefix_bank.data.copy_(prefix_bank.to(model.prefix_bank.device))

            # Load trained LoRA weights (build_hybrid_model creates the LoRA
            # structure but with random init — we must load the saved adapter)
            from peft import PeftModel
            model.peft_model = PeftModel.from_pretrained(
                model.peft_model, str(lora_adapter_path),
                is_trainable=False,  # inference only
            )
            model.eval()
            print(f"[S4] Loaded model with LoRA adapter from {lora_adapter_path}")
            print(f"     {prefix_bank.size(0)} strategies × "
                  f"{prefix_bank.size(1)} virtual tokens")
        except Exception as e:
            print(f"[S4] Warning: Could not load full model ({e}). "
                  f"Using prefix_bank only for logit analysis.")
            model = None
    else:
        print(f"[S4] No LoRA adapter found. Using prefix_bank only for logit analysis "
              f"(CPD via _forward_with_prefix with raw base model).")
        try:
            from src.modeling import build_hybrid_model
            model, _, _ = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
            model.prefix_bank.data.copy_(prefix_bank.to(model.prefix_bank.device))
            model.eval()
        except Exception as e:
            print(f"[S4] Warning: Could not load base model ({e}). "
                  f"Falling back to prefix-only analysis.")
            model = None

    return {
        "model": model,
        "tokenizer": tokenizer,
        "prefix_bank": prefix_bank,
        "label_space": label_space,
        "adapter_mode": adapter_mode,
    }


def _infer_model_path(checkpoint_dir: Path) -> str:
    """Try to infer the base model path from saved configs."""
    config_path = checkpoint_dir.parent.parent / "need" / \
        checkpoint_dir.parent.name / checkpoint_dir.name / "run_config.json"
    # Walk up to find run_config.json
    for ancestor in [checkpoint_dir] + list(checkpoint_dir.parents):
        for candidate in [
            ancestor / "run_config.json",
            Path(str(ancestor).replace("other", "need")) / "run_config.json",
        ]:
            if candidate.exists():
                cfg = json.load(open(candidate, "r", encoding="utf-8"))
                if "model_name_or_path" in cfg:
                    return cfg["model_name_or_path"]

    # Default: try common Qwen3-8B paths
    candidates = [
        "/replace/with/your/local/qwen/path",
        os.path.expanduser("~/models/Qwen2.5-8B-Instruct"),
        os.path.expanduser("~/models/Qwen2.5-7B-Instruct"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        "Cannot find base model. Please set model_name_or_path in your config "
        "or pass --model-path explicitly."
    )


def run_cpd_evaluation(
    checkpoint_dir: str,
    dataset_tag: str,
    max_samples: int = 50,
    alphas: Optional[List[float]] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
) -> Dict:
    """Run CPD evaluation and return results dict."""
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

    # Load checkpoint
    assets = load_model_and_tokenizer(checkpoint_dir, model_path=model_path)
    prefix_bank = assets["prefix_bank"]
    label_space = assets["label_space"]
    tokenizer = assets["tokenizer"]
    model = assets["model"]

    num_strategies = prefix_bank.size(0)
    K = prefix_bank.size(1)

    # Load test data
    from src.casino_dataset import load_split_examples, CasinoStrategyDataset
    from src.config import TrainConfig

    cfg = TrainConfig()
    cfg.dataset_dir = str(ROOT / "augmented_data")
    if model_path:
        cfg.model_name_or_path = model_path
    cfg.dataset_tag = dataset_tag

    test_examples = load_split_examples(cfg, "test")
    dataset = CasinoStrategyDataset(test_examples, label_space)

    # Limit samples
    if max_samples and max_samples < len(dataset):
        indices = list(range(0, len(dataset), max(1, len(dataset) // max_samples)))[:max_samples]
    else:
        indices = list(range(len(dataset)))

    print(f"[S4] Evaluating CPD on {len(indices)} test samples "
          f"(alpha={alphas}, strategies={num_strategies})")

    # Import CPD
    from solutions.src.inference_cpd import generate_with_cpd, compute_cpd_logit_overlap

    # For each alpha, we need the prefix_bank on the correct device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is not None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
    prefix_bank = prefix_bank.to(device)

    # Results container
    results = {
        "checkpoint": str(checkpoint_dir),
        "dataset_tag": dataset_tag,
        "num_samples": len(indices),
        "alphas": alphas,
        "per_alpha": {str(a): {"correct": 0, "total": 0, "generations": []} for a in alphas},
    }

    # Optional: compute logit overlap diagnostic (on first 5 samples)
    if model is not None and len(indices) >= 2:
        print("[S4] Computing logit overlap diagnostic...")
        sample = dataset[indices[0]]
        input_ids = tokenizer(
            sample["prompt"], return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)
        target_id = sample["strategy_id"]
        wrong_ids = [i for i in range(num_strategies) if i != target_id]
        if wrong_ids:
            wrong_id = wrong_ids[0]
            try:
                overlap = compute_cpd_logit_overlap(
                    model, input_ids, prefix_bank,
                    target_id, wrong_id, top_k=10,
                )
                results["logit_overlap_top10"] = overlap
                print(f"  Logit overlap (top-10, target vs wrong): {overlap:.3f}")
                if overlap > 0.8:
                    print(f"  WARNING: High overlap → CPD may produce noise, not signal")
            except Exception as e:
                print(f"  Could not compute overlap: {e}")

    # Run CPD generation
    for idx_idx, idx in enumerate(indices):
        sample = dataset[idx]
        prompt = sample["prompt"]
        target_id = sample["strategy_id"]
        gold = sample["target"]

        input_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)

        for alpha in alphas:
            try:
                with torch.no_grad():
                    gen_ids = generate_with_cpd(
                        model=model if model is not None else _get_base_model_for_cpd(cfg),
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        prefix_bank=prefix_bank,
                        target_strategy_id=target_id,
                        alpha=alpha,
                        max_new_tokens=40,
                        temperature=0.0,
                        neutral_mode="mean",
                    )
                utterance = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            except Exception as e:
                utterance = f"[ERROR: {e}]"

            results["per_alpha"][str(alpha)]["generations"].append({
                "dialogue_id": sample.get("dialogue_id", -1),
                "turn_index": sample.get("turn_index", -1),
                "strategy": sample["primary_strategy"],
                "gold_utterance": gold,
                "generated": utterance,
            })

            # We can't automatically evaluate strategy accuracy without LLM-as-judge.
            # Mark for external evaluation.
            results["per_alpha"][str(alpha)]["total"] += 1

        if (idx_idx + 1) % 10 == 0:
            print(f"  Progress: {idx_idx + 1}/{len(indices)} samples")

    # Save results
    if output_dir is None:
        output_dir = SOLUTIONS_OUT / "s4_cpd"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "cpd_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also save a human-readable summary
    summary_path = output_dir / "cpd_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CPD Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {checkpoint_dir}\n")
        f.write(f"Samples: {len(indices)}\n")
        f.write(f"Alphas: {alphas}\n")
        if "logit_overlap_top10" in results:
            f.write(f"Logit overlap (top-10): {results['logit_overlap_top10']:.3f}\n")
        f.write(f"\nGenerated files for external LLM evaluation:\n")
        for alpha in alphas:
            gen_file = output_dir / f"generations_alpha{alpha}.jsonl"
            with open(gen_file, "w", encoding="utf-8") as gf:
                for gen in results["per_alpha"][str(alpha)]["generations"]:
                    gf.write(json.dumps(gen, ensure_ascii=False) + "\n")
            f.write(f"  alpha={alpha}: {gen_file}\n")
        f.write(f"\nFull results: {output_path}\n")

    print(f"\n[S4] Results saved to {output_dir}")
    print(f"  Full JSON: {output_path}")
    print(f"  Summary:   {summary_path}")
    print(f"\n[S4] To evaluate strategy accuracy, run strategy_eval.py on the")
    print(f"  generated .jsonl files for each alpha value.")

    return results


def _get_base_model_for_cpd(cfg):
    """Load base model without PEFT for pure prefix-only CPD testing."""
    from transformers import AutoModelForCausalLM
    from src.modeling import load_tokenizer

    tokenizer = load_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=torch.bfloat16 if cfg.use_bfloat16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="S4: Contrastive Prefix Decoding evaluation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint directory (containing prefix_bank.pt)"
    )
    parser.add_argument(
        "--dataset-tag", type=str, default="casino_augmented_new_fix_seed42",
        help="Dataset tag for loading test data"
    )
    parser.add_argument(
        "--max-samples", type=int, default=50,
        help="Maximum number of test samples (default: 50)"
    )
    parser.add_argument(
        "--alphas", type=str, default="0.0,0.5,1.0,2.0,3.0,5.0",
        help="Comma-separated alpha values for CPD (default: 0.0,0.5,1.0,2.0,3.0,5.0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: solutions/output/s4_cpd/)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Base model path (auto-detected if not provided)"
    )
    parser.add_argument(
        "--diagnostic-only", action="store_true",
        help="Only run logit overlap diagnostic, skip full generation"
    )
    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    print("=" * 60)
    print("S4: Contrastive Prefix Decoding Evaluation")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Dataset:    {args.dataset_tag}")
    print(f"  Samples:    {args.max_samples}")
    print(f"  Alphas:     {alphas}")
    print("=" * 60)

    if args.diagnostic_only:
        # Quick diagnostic mode
        assets = load_model_and_tokenizer(args.checkpoint, model_path=args.model_path)
        prefix_bank = assets["prefix_bank"]
        model = assets["model"]
        tokenizer = assets["tokenizer"]
        label_space = assets["label_space"]

        if model is None:
            print("ERROR: Cannot run diagnostic without a loaded model")
            sys.exit(1)

        from solutions.src.inference_cpd import compute_cpd_logit_overlap
        from src.casino_dataset import load_split_examples, CasinoStrategyDataset
        from src.config import TrainConfig

        cfg = TrainConfig()
        cfg.dataset_dir = str(ROOT / "augmented_data")
        if args.model_path:
            cfg.model_name_or_path = args.model_path

        test_examples = load_split_examples(cfg, "test")
        dataset = CasinoStrategyDataset(test_examples, label_space)

        device = next(model.parameters()).device
        prefix_bank = prefix_bank.to(device)

        num_strategies = prefix_bank.size(0)
        results = []

        for idx in range(min(10, len(dataset))):
            sample = dataset[idx]
            input_ids = tokenizer(
                sample["prompt"], return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(device)
            target_id = sample["strategy_id"]

            for wrong_id in range(num_strategies):
                if wrong_id == target_id:
                    continue
                overlap = compute_cpd_logit_overlap(
                    model, input_ids, prefix_bank,
                    target_id, wrong_id, top_k=10,
                )
                results.append({
                    "dialogue_id": sample.get("dialogue_id", -1),
                    "target": label_space.id_to_label[target_id],
                    "wrong": label_space.id_to_label[wrong_id],
                    "overlap_top10": overlap,
                })

        avg_overlap = sum(r["overlap_top10"] for r in results) / max(1, len(results))
        print(f"\nDiagnostic Results:")
        print(f"  Samples analyzed: {len(results)}")
        print(f"  Avg logit overlap: {avg_overlap:.3f}")
        print(f"  Interpretation: ", end="")
        if avg_overlap > 0.8:
            print("High overlap → CPD unlikely to work well")
        elif avg_overlap > 0.5:
            print("Moderate overlap → CPD may help marginally")
        else:
            print("Low overlap → CPD has room to amplify strategy signal")

        diag_path = SOLUTIONS_OUT / "s4_cpd" / "cpd_diagnostic.json"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump({
                "avg_overlap_top10": avg_overlap,
                "num_comparisons": len(results),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {diag_path}")
        return

    run_cpd_evaluation(
        checkpoint_dir=args.checkpoint,
        dataset_tag=args.dataset_tag,
        max_samples=args.max_samples,
        alphas=alphas,
        output_dir=args.output_dir,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
