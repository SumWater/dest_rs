from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Sequence

import torch

from .config import TrainConfig
from .modeling import HybridStrategyModel, get_embed_device, lora_disabled_ctx
from .losses import align_labels_and_response_mask, response_nll_stats, cfg_uses_lora, cfg_uses_prefix


@torch.no_grad()
def evaluate_generation_loss(
    hybrid: HybridStrategyModel,
    dataloader,
    cfg: TrainConfig,
) -> Dict[str, float]:
    hybrid.eval()
    total_nll = 0.0
    total_tokens = 0
    device_in = get_embed_device(hybrid.peft_model)
    prefix_on = cfg_uses_prefix(cfg)
    lora_on = cfg_uses_lora(cfg)

    for batch in dataloader:
        if lora_on:
            outputs, extended_labels, _ = hybrid(
                input_ids=batch["input_ids"].to(device_in),
                attention_mask=batch["attention_mask"].to(device_in),
                strategy_ids=batch["strategy_id"].to(device_in),
                labels=batch["labels"].to(device_in),
                prefix_on=prefix_on,
                prefix_scale=cfg.prefix_scale_eval,
                use_cache=False,
            )
        else:
            with lora_disabled_ctx(hybrid.peft_model):
                outputs, extended_labels, _ = hybrid(
                    input_ids=batch["input_ids"].to(device_in),
                    attention_mask=batch["attention_mask"].to(device_in),
                    strategy_ids=batch["strategy_id"].to(device_in),
                    labels=batch["labels"].to(device_in),
                    prefix_on=prefix_on,
                    prefix_scale=cfg.prefix_scale_eval,
                    use_cache=False,
                )
        aligned_labels, aligned_mask = align_labels_and_response_mask(
            batch["labels"].to(outputs.logits.device),
            batch["response_mask"].to(outputs.logits.device),
            num_virtual_tokens=cfg.num_virtual_tokens if prefix_on else 0,
            logits_sequence_length=outputs.logits.size(1),
        )
        nll_sums, counts = response_nll_stats(outputs.logits, aligned_labels, aligned_mask)
        total_nll += float(nll_sums.sum().item())
        total_tokens += int(counts.sum().item())

    if total_tokens == 0:
        raise ValueError("Evaluation contains zero response tokens")
    mean_loss = total_nll / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "response_tokens": total_tokens,
        "nll_sum": total_nll,
    }


@torch.no_grad()
def greedy_generate(
    hybrid: HybridStrategyModel,
    tokenizer,
    prompt: str,
    strategy_id: int,
    cfg: TrainConfig,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    prefix_override: torch.Tensor | None = None,
    strategy_name: str | None = None,
) -> str:
    hybrid.eval()
    device_in = get_embed_device(hybrid.peft_model)
    max_new_tokens = max_new_tokens or cfg.demo_max_new_tokens
    temperature = cfg.demo_temperature if temperature is None else temperature
    prefix_on = cfg_uses_prefix(cfg)
    lora_on = cfg_uses_lora(cfg)

    # 若需要注入策略文本指令，拼接后再编码
    if cfg.inject_strategy_text and strategy_name:
        from .casino_dataset import _make_strategy_instruction
        prompt = _make_strategy_instruction(strategy_name) + "\n\n" + prompt

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device_in)
    attention_mask = enc["attention_mask"].to(device_in)
    strategy_ids = torch.tensor([strategy_id], dtype=torch.long, device=device_in)

    generated = []
    for _ in range(max_new_tokens):
        if lora_on:
            outputs, _, _ = hybrid(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=None,
                prefix_on=prefix_on,
                prefix_scale=cfg.prefix_scale_eval,
                prefix_override=prefix_override,
                use_cache=False,
            )
        else:
            with lora_disabled_ctx(hybrid.peft_model):
                outputs, _, _ = hybrid(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    strategy_ids=strategy_ids,
                    labels=None,
                    prefix_on=prefix_on,
                    prefix_scale=cfg.prefix_scale_eval,
                    prefix_override=prefix_override,
                    use_cache=False,
                )
        next_token_logits = outputs.logits[:, -1, :]
        if temperature and temperature > 0.0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)
        token_id = int(next_token.item())
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        next_token = next_token.to(device_in)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device_in)], dim=1
        )

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def save_swap_samples(
    hybrid: HybridStrategyModel,
    tokenizer,
    dataset,
    label_space,
    cfg: TrainConfig,
    output_path: str,
    num_examples: int,
    split_name: str,
) -> None:
    num_examples = min(num_examples, len(dataset))
    records: List[Dict] = []
    for idx in range(num_examples):
        item = dataset[idx]
        prompt = item["prompt"]
        gold_strategy = item["primary_strategy"]
        gold_id = item["strategy_id"]

        generated_by_strategy = {}
        for strategy_id, strategy_name in label_space.id_to_label.items():
            generated_by_strategy[strategy_name] = greedy_generate(
                hybrid=hybrid,
                tokenizer=tokenizer,
                prompt=prompt,
                strategy_id=strategy_id,
                cfg=cfg,
                strategy_name=strategy_name,
            )

        records.append(
            {
                "split": split_name,
                "dialogue_id": item["dialogue_id"],
                "turn_index": item["turn_index"],
                "gold_strategy": gold_strategy,
                "gold_target": item["target"],
                "all_strategies": item["all_strategies"],
                "prompt": prompt,
                "generated_by_strategy": generated_by_strategy,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
