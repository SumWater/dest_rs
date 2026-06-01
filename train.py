from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainConfig, load_config, save_config
from src.casino_dataset import (
    CasinoStrategyDataset,
    StrategyDataCollator,
    StrategyLabelSpace,
    load_split_examples,
)
from src.evaluate import evaluate_generation_loss, save_swap_samples
from src.losses import compute_training_losses
from src.modeling import (
    build_hybrid_model,
    freeze_for_adapter_mode,
    load_tokenizer,
    print_trainable_parameters,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "lora_adapter"), exist_ok=True)


def build_dataloaders(cfg: TrainConfig, tokenizer):
    train_examples = load_split_examples(cfg, "train")
    valid_examples = load_split_examples(cfg, "valid")
    test_examples = load_split_examples(cfg, "test")

    label_space = StrategyLabelSpace.fit(train_examples)
    print("=" * 88)
    print(f"[data] train examples: {len(train_examples)}")
    print(f"[data] valid examples: {len(valid_examples)}")
    print(f"[data] test examples:  {len(test_examples)}")
    print(f"[data] strategies: {label_space.labels}")
    print("=" * 88)

    train_dataset = CasinoStrategyDataset(train_examples, label_space)
    valid_dataset = CasinoStrategyDataset(valid_examples, label_space)
    test_dataset = CasinoStrategyDataset(test_examples, label_space)
    collator = StrategyDataCollator(tokenizer, cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    return train_loader, valid_loader, test_loader, label_space


def save_label_map(label_space: StrategyLabelSpace, output_dir: str) -> None:
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_space.to_json(), f, ensure_ascii=False, indent=2)


def build_optimizer(hybrid, cfg: TrainConfig):
    prefix_params, lora_params, cls_params = freeze_for_adapter_mode(hybrid, cfg.adapter_mode, cfg.freeze_prefix)
    print_trainable_parameters(hybrid)
    groups = []
    if prefix_params:
        groups.append({"params": prefix_params, "lr": cfg.prefix_lr, "name": "prefix"})
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lr, "name": "lora"})
    if cls_params:
        groups.append({"params": cls_params, "lr": cfg.lr, "name": "cls"})
    if not groups:
        raise ValueError(f"当前 adapter_mode={cfg.adapter_mode} 没有可训练参数")
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    print("=" * 88)
    for idx, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{idx}")
        count = sum(p.numel() for p in group["params"])
        print(f"[optim] {name:<8} lr={group['lr']:.2e} params={count:,}")
    print("=" * 88)
    return optimizer


def save_checkpoint(hybrid, tokenizer, cfg: TrainConfig, label_space: StrategyLabelSpace, output_dir: str) -> None:
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    hybrid.peft_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
    torch.save(
        {
            "prefix_bank": hybrid.prefix_bank.detach().cpu(),
            "strategy_classifier": hybrid.strategy_classifier.state_dict(),
            "num_virtual_tokens": cfg.num_virtual_tokens,
            "labels": label_space.labels,
            "adapter_mode": cfg.adapter_mode,
        },
        os.path.join(output_dir, "prefix_bank.pt"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    ensure_output_dir(cfg.output_dir)
    save_config(cfg, os.path.join(cfg.output_dir, "run_config.json"))

    print("=" * 88)
    print("[config]")
    print(json.dumps(cfg.__dict__, ensure_ascii=False, indent=2))
    print("=" * 88)

    tokenizer = load_tokenizer(cfg)
    train_loader, valid_loader, test_loader, label_space = build_dataloaders(cfg, tokenizer)
    save_label_map(label_space, cfg.output_dir)

    hybrid, tokenizer, catcher = build_hybrid_model(cfg, num_strategies=len(label_space.labels))
    optimizer = build_optimizer(hybrid, cfg)

    best_valid_loss = float("inf")
    history = []
    global_step = 0
    prefix_norm_before = float(hybrid.prefix_bank.norm().item())

    print("[train] starting training")
    for epoch in range(cfg.num_epochs):
        hybrid.train()
        running_gen = 0.0
        running_orth = 0.0
        running_orth_local = 0.0
        running_orth_global = 0.0
        running_cls = 0.0
        running_total = 0.0
        batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}", leave=True)
        for batch in pbar:
            global_step += 1
            losses = compute_training_losses(
                hybrid=hybrid,
                catcher=catcher,
                batch=batch,
                cfg=cfg,
                global_step=global_step,
            )
            gen_loss = losses["gen_loss"]
            orth_loss = losses["orth_loss"]
            orth_local_loss = losses["orth_local_loss"]
            orth_global_loss = losses["orth_global_loss"]
            cls_loss = losses["cls_loss"]
            total_loss = losses["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in hybrid.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
            optimizer.step()

            running_gen += gen_loss.item()
            running_orth += orth_loss.item()
            running_orth_local += orth_local_loss.item()
            running_orth_global += orth_global_loss.item()
            running_cls += cls_loss.item()
            running_total += total_loss.item()
            batches += 1

            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                gen=f"{gen_loss.item():.4f}",
                orth=f"{orth_loss.item():.4f}",
                cls=f"{cls_loss.item():.4f}",
                orth_on="Y" if losses["used_orth"] else "N",
            )

        epoch_record: Dict[str, float] = {
            "epoch": epoch + 1,
            "train_loss": running_total / max(1, batches),
            "train_gen_loss": running_gen / max(1, batches),
            "train_orth_loss": running_orth / max(1, batches),
            "train_orth_local_loss": running_orth_local / max(1, batches),
            "train_orth_global_loss": running_orth_global / max(1, batches),
            "train_cls_loss": running_cls / max(1, batches),
        }

        if cfg.eval_every_epoch:
            valid_metrics = evaluate_generation_loss(hybrid, valid_loader, cfg)
            test_metrics = evaluate_generation_loss(hybrid, test_loader, cfg)
            epoch_record.update(
                {
                    "valid_loss": valid_metrics["loss"],
                    "valid_ppl": valid_metrics["perplexity"],
                    "test_loss": test_metrics["loss"],
                    "test_ppl": test_metrics["perplexity"],
                }
            )
            print(
                f"[eval] epoch={epoch + 1} valid_loss={valid_metrics['loss']:.4f} valid_ppl={valid_metrics['perplexity']:.2f} "
                f"test_loss={test_metrics['loss']:.4f} test_ppl={test_metrics['perplexity']:.2f}"
            )

            if valid_metrics["loss"] < best_valid_loss:
                best_valid_loss = valid_metrics["loss"]
                save_checkpoint(hybrid, tokenizer, cfg, label_space, cfg.output_dir)
                print(f"[save] 新的最佳 checkpoint 已保存到 {cfg.output_dir}")
        history.append(epoch_record)

    prefix_norm_after = float(hybrid.prefix_bank.norm().item())
    print(
        f"[train] finished | prefix_norm_before={prefix_norm_before:.4f} "
        f"prefix_norm_after={prefix_norm_after:.4f}"
    )

    metrics_payload = {
        "history": history,
        "best_valid_loss": best_valid_loss,
        "prefix_norm_before": prefix_norm_before,
        "prefix_norm_after": prefix_norm_after,
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    save_checkpoint(hybrid, tokenizer, cfg, label_space, cfg.output_dir)
    save_swap_samples(
        hybrid=hybrid,
        tokenizer=tokenizer,
        dataset=valid_loader.dataset,
        label_space=label_space,
        cfg=cfg,
        output_path=os.path.join(cfg.output_dir, "swap_samples_valid.jsonl"),
        num_examples=cfg.demo_num_examples,
        split_name="valid",
    )

    catcher.remove()
    print(f"[done] 所有产物已保存到 {cfg.output_dir}")


if __name__ == "__main__":
    main()
